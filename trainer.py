# common standard library utilities
import os
import sys
import time
import json
import math
import inspect
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from torch.amp import autocast

# huggingface
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# our stuff
from data import *

R = Random(7)

class Trainer:
    def __init__(self, args, accelerator=None):
        # set up the trainer
        self.args = args
        if not accelerator:
            self.accelerator = Accelerator(log_with="wandb")
        else:
            self.accelerator = accelerator
        self.accelerator.init_trackers(
            project_name="dropfree", 
            config=vars(args),
            init_kwargs={"wandb": {"mode": None if args.wandb else "disabled",
                                   "name": args.experiment}},
        )

        # ...and the output path
        save_dir = Path(args.out_dir) / args.experiment
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir / "checkpoint"
        self.best_dir = str(save_dir / "best")

        # set up models
        self.config = AutoConfig.from_pretrained(args.architecture)
        self.tokenizer = AutoTokenizer.from_pretrained(args.architecture)

        self.config.attention_dropout = args.dropout
        self.config.hidden_dropout = args.dropout

        # enable a selective amount of dropout
        # which is the main variable we are testing
        self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.train()

        # set up data
        # TODO hard coding (because the number of iters is hard coded)
        self.train_dl = get_dataloader(load_dataset("EleutherAI/the_pile_deduplicated", streaming=True)["train"],
                                       self.tokenizer,
                                       self.config,
                                       args.batch_size)

        # leave blank
        # this will exist if we are resuming from checkpoint
        self.train_dl_skipped = None 

        # optimizer
        self.optim = self.configure_optimizers(args.weight_decay, args.lr,
                                               (args.beta1, args.beta2),
                                               device_type="cuda" if torch.cuda.is_available() else "cpu")

        # scheduler
        # TODO hard coding (because the number of iters is hard coded)
        TOTAL_ITERS = (134318121 // args.batch_size)
        warmup_steps = int(args.warmup_pct*TOTAL_ITERS)

        scheduler1 = LinearLR(self.optim, start_factor=1e-20, end_factor=1, total_iters=warmup_steps)
        scheduler2 = CosineAnnealingLR(self.optim, TOTAL_ITERS,
                                       eta_min=args.lr*args.decay_target_pct) # todo stop hardcoding
        self.scheduler = SequentialLR(self.optim, schedulers=[scheduler1, scheduler2],
                                      milestones=[warmup_steps])


        # compute training size + the counter (useful for mid-checkpoint recovery) 
        self.total_batches = 134318121 // args.batch_size
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf") # "score" means higher is better 

        # check if dropout is disabled, this may get updated by
        # the trainer dynamically if we want to early dropout
        self.__dropout_disabled = (args.dropout == 0.0)

        # weeeeeeeeeeee
        (self.model, self.optim, self.scheduler, self.train_dl) = self.accelerator.prepare(
            self.model, self.optim, self.scheduler, self.train_dl)
        if self.accelerator.is_main_process:
            wandb.watch(self.model)

    def stop_dropout_(self):
        """stop dropout for the model, if any

        useful for "early dropout" type experiments.
        """

        logger.info("Stopping any configured dropout!!")

        # get underlying module in self.model, if exists
        # should handle FSDP, DDP, and Deepspeed
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
        model.gpt_neox.emb_dropout.p = 0.0
        for i in model.gpt_neox.layers:
            i.post_attention_dropout.p = 0.0
            i.post_mlp_dropout.p = 0.0
            i.attention.attention_dropout.p = 0.0

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer
        
    def train(self):
        for eid in range(self.args.epochs):
            if self.global_step_counter_ >= ((eid+1)*self.total_batches):
                logger.debug("SKIPPING EPOCH {} due to global step count...", eid)
                continue

            self.epoch()

        self.finish()

    def finish(self):
        self.accelerator.end_training()

    def val(self, batch):
        with torch.inference_mode():
            with autocast("cuda", dtype=torch.bfloat16):
                self.model.eval()

                result = self.model(**batch)
                loss = self.gather(result.loss).cpu().item()

                score = 1/loss  # because higher score is better
                metrics = { "val/loss": loss }

                self.model.train()

                return score, metrics

    def epoch(self):
        if self.accelerator.is_main_process:
            logger.info("BEGIN EPOCH")

        # because sometimes the load function may skip some epochs
        dl = self.train_dl if not self.train_dl_skipped else self.train_dl_skipped
        for indx, i in enumerate(dl):
            # save a checkpoint, if needed
            if indx % self.args.checkpoint_interval == 0 and indx != 0:
                self.save(str(self.save_dir/str(self.global_step_counter_)))

            # perform validation and save a checkpoint, if needed
            if indx % self.args.validation_interval == 0 and indx != 0:
                score, val_metrics = self.val(i)
                self.accelerator.log(val_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("VAL | {} | score {}", self.global_step_counter_, score)

                if score > self.best_val_score_:
                    logger.info("VAL | BEST SCORE | score {}", score)
                    self.best_val_score_ = score
                    self.save(self.best_dir)
                continue # skip the validation batch

            # take a step
            loss, train_metrics = self.step(i)
            train_metrics["train/lr"] = self.optim.param_groups[0]["lr"]

            # perform logging, and then increment
            # (we do this because global_step_counter_
            #  is useful not as the # of steps but how
            #  many we need to skip for warm start)
            if indx % self.args.report_interval == 0 and indx != 0:
                self.accelerator.log(train_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("TRAIN | {}/{} | loss {}", self.global_step_counter_,
                                self.total_batches*self.args.epochs, loss)
            self.global_step_counter_ += 1

            logger.debug("STEP | {} | {}", indx, train_metrics)

        # we are done using the skipped DL since we finished the remaining batch
        self.train_dl_skipped = None

    def step(self, batch):
        # check if dropout needs disabling, if it does, do it
        if (hasattr(self.args, "disable_dropout_steps") and
            self.global_step_counter_ > self.args.disable_dropout_steps and
            not self.__dropout_disabled):
            self.stop_dropout_()
            self.__dropout_disabled = True

        with autocast("cuda", dtype=torch.bfloat16):
            loss = self.model(**batch).loss
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            loss = self.gather(loss).cpu().item() 
            metrics = { "train/loss": loss }

        return loss, metrics
        

    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

        # skip batches
        # self.train_dl_skipped = self.accelerator.skip_first_batches(self.train_dl,
        #                                                             self.global_step_counter_ % self.total_batches)

    def save(self, path):
        if self.accelerator.is_main_process:
            logger.debug("CHECKPOINT | saving checkpoint at {}", path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(path)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.args),
                "steps": self.global_step_counter_,
                "score": self.best_val_score_
            }, df)

    @classmethod
    def from_pretrained(cls, path, disable_wandb=True, accelerator=None):
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)
        args = Namespace(**data.get("config", {}))
        args.wandb = False
        new = cls(args, accelerator)
        new.load(path)

        if disable_wandb:
            new.args.wandb = False

        return new

    @property
    def device(self):
        return self.accelerator.device

    def gather(self, n):
        result = self.accelerator.gather(n)
        if isinstance(result, list):
            return sum(result)/len(result)
        else:
            return result.mean()
    

