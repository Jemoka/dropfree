# common standard library utilities
import os
import sys
import glob
import time
import json
import math
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
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from torch.utils.data import DataLoader, Dataset

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# data utilities
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from datasets import load_dataset

tqdm.pandas()

from datasets import load_dataset
from functools import partial
from transformers import DataCollatorForLanguageModeling

def tokenize(tokenizer, config, example):
    return tokenizer(example["text"],
                     return_special_tokens_mask=True,
                     truncation=True,
                     padding=True,
                     max_length=config.max_position_embeddings)

def get_dataloader(dataset, tokenizer, config, batch_size=256):
    """gets dataloader for training from HF dataset

    Arguments
    ---------
        dataset: HF dataset
                dataset to get dataloader from, must have "text" column
        tokenizer: HF tokenizer
                tokenizer to use for tokenization
        config: HF config
                config for model, useful for `max_position_embeddings`
        batch_size: int
                batch size for dataloader

    Returns
    -------
        torch.utils.data.DataLoader
            dataloader for training
    """


    # apparently the newer LMs' tokenizer doesn't have a `pad_token`
    # so we need to add it
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokenized = dataset.map(partial(tokenize, tokenizer, config), batched=True)
    tokenized = tokenized.remove_columns(["text"])
    collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=16, mlm=False, return_tensors="pt")

    return DataLoader(tokenized, collate_fn=collator, batch_size=batch_size)


