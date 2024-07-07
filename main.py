import os
import torch
import random
from transformers import AutoConfig, AutoTokenizer, AutoModel
from trainer import Trainer

import datasets
datasets.config.STREAMING_READ_MAX_RETRIES = 20000
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10

from datasets import load_dataset

import argparse
import logging

import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger

L = get_logger("dropfree", log_level="DEBUG")

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    handlers=[logging.StreamHandler()])
L.setLevel(logging.DEBUG)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='bert')
    parser.add_argument("experiment", help="name for the experiment", type=str)
    parser.add_argument("save_dir", help="where to put logs and checkpoints to", type=str)
    parser.add_argument("--out_dir", default=None, type=str, help="path to export the model")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb", type=str, help="dataset")
    parser.add_argument("--batch_size", default=12, type=int, help="training batch size *PER WORKER*")
    parser.add_argument("--base", default="FacebookAI/xlm-roberta-large", type=str, help="base model configuration (and tokenizer) to use")
    parser.add_argument("--dropout", default=False, action="store_true", help="whether to enable dropout")
    parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="number of steps to warm up scheduler")
    args = parser.parse_args()

    trainer = Trainer(args)

    if args.experiment == "export":
        trainer.load(args.save_dir)
        trainer.model.save_pretrained(args.out_dir)
        trainer.tokenizer.save_pretrained(args.out_dir)
    else:
        trainer.train()
