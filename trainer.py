import torch
import math
import torch.nn as nn
from ray.train import get_context, report, Checkpoint, get_checkpoint, get_dataset_shard
import ray.train.torch as rt
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

class Trainer:
    def __init__(self, config):
        dataset = load_dataset(config.dataset, streaming=True, split="train")
        val_dataset = load_dataset(config.dataset, streaming=True, split="validation")
        self.loader = iter(DataLoader(dataset, batch_size=config.batch_size))
        self.val_loader = iter(DataLoader(val_dataset, batch_size=config.batch_size))

        self.model_config = AutoConfig.from_pretrained(config.base)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base)
        self.training_config = config

        if not config.dropout:
            self.attention_probs_dropout_prob = 0
            self.hidden_dropout_prob = 0

        self.model = AutoModelForMaskedLM.from_config(self.model_config)
        self.optim = AdamW(self.model.parameters(), lr=config.lr, betas=(0.9,0.999), eps=1e-6, weight_decay=0.01)

        scheduler1 = LinearLR(self.optim, start_factor=1e-6, end_factor=1, total_iters=config.warmup_steps)
        scheduler2 = LinearLR(self.optim, start_factor=1, end_factor=0, total_iters=1.5e6/config.batch_size) # todo stop hardcoding
        self.scheduler = SequentialLR(self.optim, schedulers=[scheduler1, scheduler2], milestones=[config.warmup_steps])

        self.global_step_counter_ = 0
        self.best_val_loss_ = float("+inf")

        self.save_dir = os.path.join(config.save_dir, config.experiment)
        self.accelerator = Accelerator()
        self.accelerator.init_trackers(
            project_name="dropfree", 
            config=vars(self.training_config),
            init_kwargs={"wandb": {"entity": "jemoka",
                                   "mode": None if config.wandb else "disabled",
                                   "name": config.experiment}},
        )

        (self.model, self.optim, self.scheduler
         self.loader, self.val_loader) = self.accelerator.prepare(
             self.model, self.optim, self.scheduler,
             self.loader, self.val_loader
         )

        self.loader = self.accelerator.skip_first_batches(self.loader,
                                                          self.global_step_counter_)
        

        if os.path.exists(os.path.join(self.save_dir, "config.json")):
            L.info(f"loading existing weights at {self.save_dir}")
            self.load(self.save_dir)


    def train(self):
        config = self.training_config

        for indx, batch in enumerate(self.loader):
            inputs, labels = process_batch(batch["text"], self.tokenizer, self.device)
            outputs = self.model(**inputs, labels=labels)

            self.accelerator.backward(outputs.loss)
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            if indx % 25:
                loss = self.accelerator.gather(outputs.loss).cpu().item()

                L.info(f"training | batch {indx} | loss {round(loss, 3)}", main_process_only=True)
                self.accelerator.log({"training/loss": loss,
                                      "training/lr": self.optim.param_groups[0]["lr"]},
                                     step=self.global_step_counter_)

            if indx % 256 == 0:
                self.val()

            self.global_step_counter_ += 1

        if self.is_headnode and config.wandb:
            wandb.finish()

    def val(self):
        loss = 0
        count = 0

        for indx, batch in enumerate(self.val_loader):
            with torch.inference_mode():
                inputs, labels = process_batch(batch["text"], self.tokenizer, self.device)

                outputs = self.model(**inputs, labels=labels)

            loss += self.accelerator.gather(outputs.loss)
            count += 1

            if indx == 100:
                break

        if self.accelerator.is_main_process:
            loss = loss.cpu().item()/count
            ppl = math.exp(loss)

            if loss < self.best_val_loss_:
                self.best_val_loss_ = loss

            self.save(self.save_dir)
            self.accelerator.log({"validation/loss": loss, "validation/ppl": ppl},
                                step=self.global_step_counter_)
            L.info(f"validation | loss {round(loss, 3)} | ppl {round(ppl, 3)}")

    def save(self, path):
        self.accelerator.save_state(path)
        with open(os.path.join(self.save_dir, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.training_config),
                "steps": self.global_step_counter_,
                "loss": self.best_val_loss_
            }, df)


    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(self.save_dir, "config.json"), 'r') as df:
            data = json.load(df)

        self.training_config = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_loss_ = data.get("loss", 0)

    @property
    def device(self):
        return self.accelerator.device
        
