#!/bin/bash

# this is just a dry run
uvrun -q jag -d "a6000" -g 4 -r 32G -c 4 "source .venv/bin/activate &&  accelerate launch main.py e0_dryrun_160m_0.0 --batch_size 16 --wandb"
uvrun -q jag -d "a6000" -g 4 -r 32G -c 4 "source .venv/bin/activate &&  accelerate launch main.py e0_dryrun_160m_0.5 --batch_size 16 --dropout 0.5 --wandb"
