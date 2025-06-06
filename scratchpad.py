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


from commands import configure
from trainer import Trainer

# trainer = Trainer.from_pretrained("./output/e1_1.4b-d0.0/checkpoint/1179648")
# model = trainer.model
# tokenizer = trainer.tokenizer




