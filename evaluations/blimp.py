"""
blimp.py
Scoring the BLiMP dataset using a given pretrained decoder LM
"""

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
    "<level>{level: ^8}</level>| "
    "<magenta>({name}:{line})</magenta> <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    enqueue=True
)


# from commands import configure

from trainer import Trainer
from datasets import load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
import click
import json

def prepare_indx(indx, ds, tokenizer, batch_size=16):
    """tokenize and prepare a this specific indicie of the dataset for LM/evals
    
    Arguments
    ----------
        indx : str
            The column of the dataset to prepare
        ds : Dataset
            The dataset to prepare
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer to use for data tokenization/coallation
        batch_size : optional, int
            The batch size to use for the DataLoader

    Returns
    ----------
        torch.utils.data.DataLoader
            The prepared dataset
    """

    res = ds.map(lambda x: tokenizer(x[indx])).select_columns(["input_ids", "attention_mask"])
    res = res.add_column("labels", res["input_ids"])
    res_dl = DataLoader(res, batch_size=batch_size, shuffle=False,
                        collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=16))

    return res_dl

criterion = torch.nn.CrossEntropyLoss(reduction='none')
def score_ppl(logits, labels):
    nll_sums = criterion(logits.permute(0,2,1)[:, :, :-1], labels[:,1:]).sum(dim=1)
    nll_means = nll_sums/(labels != -100).sum(dim=1)
    return torch.exp(nll_means)

def score_blimp(trainer, subset, batch_size=512):
    """Score the BLiMP dataset using the given trainer and subset

    The BLiMP dataset is a dataset of sentences with a good and bad version of each sentence
    through having some grammatical permutation.

    We compute the accuracy of the model by comparing the perplexity of the model's output on
    the good and bad sentences, and counting the number of times the model correctly identifies
    the better sentence through assigning it with a lower perplexity.

    Arguments
    ---------
        trainer : Trainer
            The trainer carrying the model, etc.
            to use for scoring
        subset : str
            The subset of the BLiMP dataset to score
        batch_size : optional, int
            The batch size to use for scoring

    Returns
    -------
        float
            The accuracy of the model on the BLiMP dataset
    """

    model = trainer.model
    tokenizer = trainer.tokenizer

    ds = load_dataset("nyu-mll/blimp", subset)["train"]

    good_dl = prepare_indx("sentence_good", ds, tokenizer, batch_size)
    bad_dl = prepare_indx("sentence_bad", ds, tokenizer, batch_size)

    correctness_contrib = 0.0 # the amount of 1 (Trues), summed
    total_count = 0 # the count, used for calculating true accuracy by divinging the top

    for good_batch, bad_batch in tqdm(zip(iter(good_dl), iter(bad_dl)), total=len(bad_dl)):
        with torch.inference_mode():
            good_out = model(**{i:j.to(trainer.device) for i,j in good_batch.items()})
            bad_out = model(**{i:j.to(trainer.device) for i,j in bad_batch.items()})

        ppl_good = score_ppl(good_out.logits, good_batch["labels"].to(trainer.device))
        ppl_bad = score_ppl(bad_out.logits, bad_batch["labels"].to(trainer.device))

        correctness_contrib += (ppl_good < ppl_bad).float().mean().item()*batch_size
        total_count += batch_size

    return correctness_contrib / total_count

@logger.catch()
@click.command()
@click.argument("weights", type=str)
@click.argument("output", type=click.File('w'))
@click.option("--subset", type=str, default=["all"], help="Which BLiMP subset to score?", multiple=True)
@click.option("--batch_size", type=int, default=512, help="How many samples to score at once?")
def blimp(weights, output, subset, batch_size):
    trainer = Trainer.from_pretrained(weights)
    subsets = get_dataset_config_names("nyu-mll/blimp")

    # validate that we are actually scoring a valid set
    if "all" in subset:
        subset = subsets
    for i in subset:
        if i not in subsets:
            raise ValueError(f"Unrecognized subset: {i}")

    # collate scores
    scores = {}
    for i in subset:
        scores[i] = score_blimp(trainer, i, batch_size=batch_size)

    json.dump(scores, output, indent=4)


if __name__ == "__main__":
    blimp()
