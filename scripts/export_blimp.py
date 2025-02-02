"""
export_blimp.py
Aggregate blimp data that has been evaluated into CSV files
"""

import re
import json
import seaborn
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def run_export(prefix, in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    data_files = [Path(i) for i in glob(str(in_path / (prefix+"*")))]
    data_files = sorted([(int(i.stem.split("_")[-1]), str(i)) for i in data_files],
                        key=lambda x:x[0])

    collated_keys = defaultdict(list)
    means = []

    for indx, j in data_files:
        with open(j) as df:
            data = json.load(df)

        for k,v in data.items():
            collated_keys[k].append(v)

        means.append(sum(data.values())/len(data))
    collated_keys["average"] = means

    df = pd.DataFrame(collated_keys)
    df.index = [i[0] for i in data_files]

    df.to_csv(out_path/"data.csv", index=True)

    with open(out_path/"metadata.json", 'w') as f:
        json.dump({
            "prefix": prefix,
            "dropout": float(re.search(r"d(\d+.?\d+)", prefix).group(1)),
            "checkpoints": [i[0] for i in data_files]
        }, f)

import click

@click.command()
@click.argument("prefix", type=str)
@click.argument("in_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def export_blimp(prefix, in_path, out_path):
    # prefix = "e1_1.4b-d0.0"
    # in_path = "./results/blimp"
    # out_path = "./results/blimp_e1.4b-d0.0"

    run_export(prefix, in_path, out_path)

if __name__ == "__main__":
    export_blimp()

