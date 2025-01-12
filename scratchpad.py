"""
scratchpad.py
A place to test code snippets and experiment with new ideas.
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

from trainer import Trainer
from commands import configure


config = configure(
    "test",
    lr=1.2e-4,
    batch_size=16,
    architecture="EleutherAI/pythia-160m",
    report_interval=1,
    checkpoint_interval=16,
    validation_interval=32
)
trainer = Trainer(config)
trainer.train()

