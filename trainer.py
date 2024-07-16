import torch
import math
import torch.nn as nn
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader

import os
import json
import wandb
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from accelerate import Accelerator
from accelerate.logging import get_logger

L = get_logger("dropfree", log_level="DEBUG")

from data import process_batch

import tempfile

from argparse import Namespace

from torch.utils.data import IterableDataset

def collate_and_process(x, tokenizer, device):
    batch = [i["text"] for i in x]
    inputs, labels = process_batch(batch, tokenizer, device)

    return { **inputs, "labels": labels }

class Trainer:
    def __init__(self, config):

        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="dropfree", 
            config=vars(config),
            init_kwargs={"wandb": {"entity": "jemoka",
                                   "mode": None if config.wandb else "disabled",
                                   "name": config.experiment}},
        )

        self.model_config = AutoConfig.from_pretrained(config.base)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base)

        dataset = load_dataset(config.dataset, streaming=True, split="train")
        self.loader = DataLoader(dataset, 
                                 collate_fn=lambda x: collate_and_process(x, self.tokenizer, self.device), 
                                 batch_size=config.batch_size)

        self.training_config = config

        if not config.dropout:
            self.model_config.attention_probs_dropout_prob = 0
            self.model_config.hidden_dropout_prob = 0

        self.model = AutoModelForMaskedLM.from_config(self.model_config)

        self.optim = AdamW(self.model.parameters(), lr=config.lr, betas=(0.9,0.999), eps=1e-6, weight_decay=0.01)

        scheduler1 = LinearLR(self.optim, start_factor=1e-20, end_factor=1, total_iters=config.warmup_steps)
        scheduler2 = LinearLR(self.optim, start_factor=1, end_factor=0, total_iters=14.9e6/config.batch_size) # todo stop hardcoding
        self.scheduler = SequentialLR(self.optim, schedulers=[scheduler1, scheduler2], milestones=[config.warmup_steps])

        self.global_step_counter_ = 0
        self.best_val_loss_ = float("+inf")

        self.save_dir = os.path.join(config.save_dir, config.experiment, "checkpoint")
        self.best_dir = os.path.join(config.save_dir, config.experiment, "best")
        (self.model, self.optim, self.scheduler, self.loader) = self.accelerator.prepare(
             self.model, self.optim, self.scheduler, self.loader)

        if os.path.exists(os.path.join(self.save_dir, "config.json")):
            L.info(f"loading existing weights at {self.save_dir}")
            self.load(self.save_dir)
            dataset = dataset.skip(config.batch_size*self.global_step_counter_*self.accelerator.state.num_processes)
            self.loader = DataLoader(dataset, 
                                     collate_fn=lambda x: collate_and_process(x, self.tokenizer, self.device), 
                                     batch_size=config.batch_size)

            self.loader = self.accelerator.prepare(self.loader)

        
    def train(self):
        if self.accelerator.is_main_process:
            wandb.watch(self.model)

        config = self.training_config

        for indx, batch in enumerate(iter(self.loader)):
            if indx % 1024 == 0:
                # we can do this because we are not training more than
                # one epoch
                with torch.inference_mode():
                    outputs = self.model(**batch)
                loss = self.accelerator.gather(outputs.loss).mean().cpu().item()
                ppl = math.exp(loss)
                if loss < self.best_val_loss_:
                    self.best_val_loss_ = loss
                    self.save(self.best_dir)

                self.save(self.save_dir)
                self.accelerator.log({"validation/loss": loss, "validation/ppl": ppl},
                                    step=self.global_step_counter_)
                L.info(f"validation | loss {round(loss, 3)} | ppl {round(ppl, 3)}", main_process_only=True)

                continue

            outputs = self.model(**batch)

            self.accelerator.backward(outputs.loss)
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            if indx % 64 == 0:
                loss = self.accelerator.gather(outputs.loss).mean().cpu().item()

                L.info(f"training | batch {indx} | loss {round(loss, 3)}", main_process_only=True)
                self.accelerator.log({"training/loss": loss,
                                      "training/lr": self.optim.param_groups[0]["lr"]},
                                     step=self.global_step_counter_)

            self.global_step_counter_ += 1

        self.accelerator.end_training()

    def save(self, path):
        self.accelerator.save_state(path)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.training_config),
                "steps": self.global_step_counter_,
                "loss": self.best_val_loss_
            }, df)


    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.training_config = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_loss_ = data.get("loss", 0)

    @property
    def device(self):
        return self.accelerator.device
        
