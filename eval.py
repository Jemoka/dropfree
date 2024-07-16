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

def collate_and_process(x, tokenizer, device):
    batch = [i["text"] for i in x]
    inputs, labels = process_batch(batch, tokenizer, device)

    return { **inputs, "labels": labels }

class Evaluator:
    def __init__(self, model, dataset="cerebras/SlimPajama-627B", batch_size=12):

        self.accelerator = Accelerator(cpu=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        dataset = load_dataset(dataset, streaming=True, split="test")
        self.loader = DataLoader(dataset, 
                                 collate_fn=lambda x: collate_and_process(x, self.tokenizer, self.device), 
                                 batch_size=batch_size)
        self.model, self.loader = self.accelerator.prepare(self.model, self.loader)

    def eval(self):
        total_loss = 0
        for indx, i in enumerate(self.loader):
            total_loss += self.step(i).cpu().item()
            if indx >= 2048:
                break
        return total_loss/indx

    def step(self, step):
        # perform forward pass
        res = self.model(**step)
        mask_idx = (step["input_ids"] == self.tokenizer.mask_token_id)
        
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
print(f"model: {model}, result: {round(res, 8)}")

