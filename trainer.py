import torch
import math
import torch.nn as nn
from ray.train import get_context, report, Checkpoint, get_checkpoint
import ray.train.torch as rt
from datasets import load_dataset
from torch.utils.data import DataLoader

import os
import wandb
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from data import process_batch

import tempfile

from argparse import Namespace

class Trainer:
    def __init__(self, config, train_split="train", val_split="validation"):
        dataset = load_dataset(config.dataset, split=train_split, streaming=True)
        self.loader = DataLoader(dataset, batch_size=config.batch_size)

        val_dataset = load_dataset(config.dataset, split=val_split, streaming=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

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
        self.local_step_counter_ = 0

    @classmethod
    def execute(cls, *args, **kwargs):
        trainer = cls(*args, **kwargs)
        return trainer.train

    def train(self):
        config = self.training_config

        # load checkpoint, if exists
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                self.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

        if self.is_headnode and self.training_config.wandb:
            wandb.init(
                project="dropfree",
                entity="jemoka",
                name=config.experiment,
                config=vars(config)
            )

        self.model = rt.prepare_model(self.model)
        self.loader = rt.prepare_data_loader(self.loader)
        self.val_loader = rt.prepare_data_loader(self.val_loader)
        self.optim = rt.prepare_optimizer(self.optim)


        for indx, batch in enumerate(self.loader):
            if self.global_step_counter_ > self.local_step_counter_:
                self.local_step_counter_ += 1
                continue

            inputs, labels = process_batch(batch["text"], self.tokenizer)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(**inputs, labels=labels)

            outputs.loss.backward()
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            loss = outputs.loss.cpu().item()
            if self.is_headnode:
                print(f"trained batch {indx} | loss {round(loss, 3)}")
                if self.training_config.wandb:
                    wandb.log({"training/loss": loss,
                               "training/lr": self.optim.param_groups[0]["lr"]})
            if self.is_headnode and indx % 16 == 0:
                self.val()

            self.global_step_counter_ += 1
            self.local_step_counter_ += 1

        if self.is_headnode and config.wandb:
            wandb.finish()

    def val(self):
        loss = 0
        count = 0

        for indx, batch in enumerate(self.val_loader):
            with torch.inference_mode():
                inputs, labels = process_batch(batch["text"], self.tokenizer)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(**inputs, labels=labels)

            loss += outputs.loss
            count += 1

            if indx == 100:
                break

        loss = loss.cpu().item()/count
        ppl = math.exp(loss)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            self.save(os.path.join(temp_checkpoint_dir, "checkpoint.pt"))
            report({"loss": loss, "ppl": ppl}, 
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))

        if self.is_headnode and self.training_config.wandb:
            wandb.log({"validation/loss": loss,
                       "validation/ppl": ppl})

        print(f"validation: loss {round(loss, 3)} | ppl {round(ppl, 3)}")

    def save(self, path):
        torch.save({
            "scheduler": self.scheduler.state_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "config": vars(self.training_config),
            "steps": self.global_step_counter_
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.scheduler.load_state_dict(data.get("scheduler", {}))
        self.model.load_state_dict(data.get("model", {}))
        self.optimizer.load_state_dict(data.get("optimizer", {}))
        self.training_config = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.set("steps", 0)

    @property
    def is_headnode(self):
        return get_context().get_world_rank() == 0
    @property
    def world_size(self):
        return get_context().get_world_size()
    @property
    def device(self):
        return rt.get_device()


