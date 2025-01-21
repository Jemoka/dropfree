"""
hellaswag.py
Runs HellaSwag evaluation on the model using likelyhood scoring
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
from torch.utils.data import DataLoader, TensorDataset
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

def score_hellaswag(trainer, slice="validation", batch_size=512):
    """Score the Hellaswag dataset using the given trainer and subset

    The Hellaswag dataset is a dataset of commonsense reasoning tasks, where the model
    must predict the correct ending to a given sentence.     

    We compute the accuracy of the model by comparing the perplexity of the model's output on
    the correct sentences vs the others, and counting the number of times the model correctly
    identifies the better sentence through assigning it with a lower perplexity.

    Arguments
    ---------
        trainer : Trainer
            The trainer carrying the model, etc.
            to use for scoring
        subset : str
            The subset of the HellaSwag dataset to score
        batch_size : optional, int
            The batch size to use for scoring

    Returns
    -------
        float
            The accuracy of the model on the BLiMP dataset
    """

    model = trainer.model
    tokenizer = trainer.tokenizer

    ds = load_dataset("Rowan/hellaswag")[slice]
    ds_mapped = ds.map(lambda x: {f"text_{i}": x["ctx"]+" "+x["endings"][i] for i in range(4)})
    labels = [int(i) for i in ds["label"]]

    dl_0 = prepare_indx("text_0", ds_mapped, tokenizer, batch_size=batch_size)
    dl_1 = prepare_indx("text_1", ds_mapped, tokenizer, batch_size=batch_size)
    dl_2 = prepare_indx("text_2", ds_mapped, tokenizer, batch_size=batch_size)
    dl_3 = prepare_indx("text_3", ds_mapped, tokenizer, batch_size=batch_size)
    dl_labels = DataLoader(
        TensorDataset(torch.tensor(labels)),
        shuffle=False,
        batch_size=batch_size
    )
    (dl_0, dl_1, dl_2, dl_3, dl_labels) = trainer.accelerator.prepare(
        dl_0, dl_1, dl_2, dl_3, dl_labels)

    total_count = 0
    total_true = 0

    for a,b,c,d,label in tqdm(zip(iter(dl_0), iter(dl_1),
                                  iter(dl_2), iter(dl_3),
                                  iter(dl_labels)), total=len(dl_0)):
        with torch.inference_mode():
            a_out = model(**a)
            b_out = model(**b)
            c_out = model(**c)
            d_out = model(**d)

        ppl_a = score_ppl(a_out.logits, a["labels"])
        ppl_b = score_ppl(b_out.logits, b["labels"])
        ppl_c = score_ppl(c_out.logits, c["labels"])
        ppl_d = score_ppl(d_out.logits, d["labels"])

        res = (torch.stack([ppl_a, ppl_b, ppl_c, ppl_d],
                        dim=1).argmin(dim=1) == label[0])
        total_count += res.size(0)
        total_true += res.sum().cpu().item()

    return total_true/total_count

@logger.catch()
@click.command()
@click.argument("weights", type=str)
@click.argument("output", type=click.File('w'))
@click.option("--subset", type=str, default="validation", help="Which HellaSwag slice to score?", multiple=True)
@click.option("--batch_size", type=int, default=512, help="How many samples to score at once?")
def blimp(weights, output, subset, batch_size):
    trainer = Trainer.from_pretrained(weights)

    # collate scores
    scores = {
        "slice": subset,
        "accuracy": score_blimp(trainer, subset, batch_size=batch_size)
    }
    json.dump(scores, output, indent=4)


if __name__ == "__main__":
    blimp()
