"""
merge_fsdp.py
since modes are largly saved as FSDP, we want to export the checkpoints into fully merged models
"""

import click
from pathlib import Path
from glob import glob
from accelerate.utils import merge_fsdp_weights

def run(path):
    checkpoints = [Path(i) for i in glob(str(Path(path) / "checkpoint" / "**" / "pytorch_model_fsdp_0"))]
    optimizers = [Path(i) for i in glob(str(Path(path) / "checkpoint" / "**" / "optimizer_0"))]

    for i,j in zip(sorted(checkpoints), sorted(optimizers)):
        merge_fsdp_weights(checkpoint,
                        str(checkpoint.parent/"pytorch_model_merged"),
                        safe_serialization=True)
        (checkpoint.parent/"pytorch_model_merged"/ "model.safetensors").rename(checkpoint.parent/ "model.safetensors")
        (checkpoint.parent/"pytorch_model_merged").rmdir()
        merge_fsdp_weights(optimizer,
                        str(optimizer.parent/"optimizer_merged"),
                        safe_serialization=False)
        (optimizer.parent/"optimizer_merged"/ "pytorch_model.bin").rename(optimizer.parent/ "optimizer.bin")
        (optimizer.parent/"optimizer_merged").rmdir()

######


@click.command()
@click.argument("path", type=click.Path(exists=True))
def merge_fsdp(path):
    """Merge all fsdp model shards and optimizer shards found in PATH."""
    run(str(path))

if __name__ == "__main__":
    merge_fsdp()


