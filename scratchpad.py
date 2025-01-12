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


config = configure("test",
                   lr=1.2e-6,
                   batch_size=2,
                   architecture="EleutherAI/pythia-6.9b")
trainer = Trainer(config)
trainer.train()

