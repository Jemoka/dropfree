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


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

trainer = Trainer.from_pretrained("./output/e0_dryrun_160m_0.0/best")

model = trainer.model
tokenizer = trainer.tokenizer

result = model.generate(**tokenizer(["this is surprisingly "], return_tensors="pt").to("cuda"),
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        max_length=256)
print(tokenizer.batch_decode(result)[0])





# import os
# os.listdir("./output")

# config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")

# config.attention_dropout = 0.7
# config.hidden_dropout = 0.7
# config.maximum_chicken = 0.1
# config
# config.max_position_embeddings

# print("ehwo")

# model = AutoModelForCausalLM.from_config(config)
# model


# from data import get_dataloader

# dataset = load_dataset("EleutherAI/the_pile_deduplicated", streaming=True)["train"]
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

# dl = get_dataloader(dataset, tokenizer, config, 3)
# tmp = next(iter(dl))
# tmp["input_ids"].shape


