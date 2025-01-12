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

trainer_df = Trainer.from_pretrained("output/e0_dryrun_160m_0.0/best")
trainer_do = Trainer.from_pretrained("output/e0_dryrun_160m_0.3/best")

model_df = trainer_df.model
model_do = trainer_do.model

tokenizer = trainer_df.tokenizer

prompt = tokenizer("The capital of Canada is the city of", return_tensors="pt")
prompt

model_df_out = model_df.generate(prompt["input_ids"].cuda(), repetition_penalty=1.1)
model_do_out = model_do.generate(prompt["input_ids"].cuda(), repetition_penalty=1.1)

tokenizer.batch_decode(model_df_out)
tokenizer.batch_decode(model_do_out)

