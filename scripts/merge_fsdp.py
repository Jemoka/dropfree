"""
merge_fsdp.py
since modes are largly saved as FSDP, we want to export the checkpoints into fully merged models
"""

import shutil
import click
from pathlib import Path
from glob import glob
from accelerate.utils import merge_fsdp_weights

def run(path, get_one=False):
    if not get_one:
        checkpoints = [Path(i) for i in glob(str(Path(path) / "checkpoint" / "**" / "pytorch_model_fsdp_0"))]
        optimizers = [Path(i) for i in glob(str(Path(path) / "checkpoint" / "**" / "optimizer_0"))]
    else:
        checkpoints = [Path(path) /  "pytorch_model_fsdp_0"]
        optimizers = [Path(path) / "optimizer_0"]


    for checkpoint,optimizer in zip(sorted(checkpoints), sorted(optimizers)):
        merge_fsdp_weights(checkpoint,
                        str(checkpoint.parent/"pytorch_model_merged"),
                        safe_serialization=True)
        (checkpoint.parent/"pytorch_model_merged"/ "model.safetensors").rename(checkpoint.parent/ "model.safetensors")
        (checkpoint.parent/"pytorch_model_merged").rmdir()
        shutil.rmtree(str(checkpoint.parent/"pytorch_model_fsdp_0"))
        merge_fsdp_weights(optimizer,
                        str(optimizer.parent/"optimizer_merged"),
                        safe_serialization=False)
        (optimizer.parent/"optimizer_merged"/ "pytorch_model.bin").rename(optimizer.parent/ "optimizer.bin")
        (optimizer.parent/"optimizer_merged").rmdir()
        shutil.rmtree(str(optimizer.parent/"optimizer_0"))

######


@click.group()
def merge_fsdp():
    """Merge all fsdp model shards and optimizer shards found in PATH."""

@merge_fsdp.command()
@click.argument("path", type=click.Path(exists=True))
def export(path):
    run(str(path))

@merge_fsdp.command()
@click.argument("path", type=click.Path(exists=True))
def get_one(path):
    run(str(path), get_one=True)


if __name__ == "__main__":
    merge_fsdp()


