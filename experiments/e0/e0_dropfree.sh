#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:4
#SBATCH --job-name=houjun-dropfree-e0_dropfree
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e0_dropfree.out
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

cd .
source .venv/bin/activate
accelerate launch main.py e0_dryrun_160m_0.0 --batch_size 16 --wandb
