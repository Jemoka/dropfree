#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a6000:4
#SBATCH --job-name=houjun-dropfree-e0_dryrun_160m_0.3
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e0_dryrun_160m_0.3.out
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

cd .
source .venv/bin/activate
accelerate launch main.py e0_dryrun_160m_0.3 --dropout 0.3 --batch_size 16 --wandb
