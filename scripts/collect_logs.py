import re
import click
import pandas as pd

@click.command()
@click.argument("log", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def collect_logs(log, output):
    with open(log, 'r') as df:
        data = df.readlines()

    data = [i for i in data if "TRAIN" in i]
    data = [i.split("|")[-2:] for i in data]

    steps = []
    losses = []
    for step, loss in data:
        step = int(step.strip().split("/")[0])
        loss = float(re.sub('[^0-9.]+', '', loss.replace("loss", "")))
        steps.append(step)
        losses.append(loss)

    df = pd.DataFrame({"steps": steps, "losses": losses})
    df.to_csv(output, index=False)

if __name__ == "__main__":
    collect_logs()

