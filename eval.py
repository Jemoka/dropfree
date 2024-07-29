import torch
import math
import torch.nn as nn
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

import logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    handlers=[logging.StreamHandler()])


def collate_and_process(x, tokenizer, device):
    batch = [i["text"] for i in x]
    inputs, labels = process_batch(batch, tokenizer, device)

    return { **inputs, "labels": labels }

class Evaluator:
    def __init__(self, model, dataset="cerebras/SlimPajama-627B", batch_size=12, alt=None):

        self.accelerator = Accelerator()
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        dataset = load_dataset(dataset, streaming=True, split="test")
        self.loader = DataLoader(dataset, 
                                 collate_fn=lambda x: collate_and_process(x, self.tokenizer, self.device), 
                                 batch_size=batch_size)
        self.model, self.loader = self.accelerator.prepare(self.model, self.loader)
        self.model.eval()

        if alt:
            self.alt = self.accelerator.prepare(AutoModelForMaskedLM.from_pretrained(alt))
        else:
            self.alt = None

    def eval(self):
        total_loss = 0
        for indx, i in enumerate(self.loader):
            res = self.step(i).cpu().item()
            total_loss += res
            if indx >= 2048:
                break
            if indx % 32 == 0:
                L.info(f"TEST | {indx}/2048 | result: {round(res, 3)}")
        return total_loss/indx

    def step(self, step):
        # perform forward pass
        res = self.model(**step)
        mask_idx = (step["input_ids"] == self.tokenizer.mask_token_id)

        if self.alt != None:
            res_alt = self.alt(**step)
            pred_dists_alt = F.softmax(res_alt.logits[mask_idx], dim=1)
        
        # create distributions of the input
        pred_dists = F.softmax(res.logits[mask_idx], dim=1)
        label_dists = F.one_hot((step["labels"])[mask_idx], num_classes=res.logits.shape[-1])

        # create cross entropy NLL loss
        return ((-pred_dists.log())*label_dists).sum(dim=1).mean()
        
    
    @property
    def device(self):
        return self.accelerator.device

 
model = "./models/no_dropout"
ev = Evaluator(model)
res = ev.eval()
L.info(f"model: {model}, result: {round(res, 8)}")

