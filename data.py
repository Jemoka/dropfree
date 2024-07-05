import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def process_batch(batch, tokenizer, device="cpu"):
    tokenized = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    inputs = tokenized["input_ids"]
    inputs_backup = torch.clone(tokenized["input_ids"])

    corrupt_candidates = tokenized["attention_mask"].nonzero()
    corrupt_candidates = corrupt_candidates[
        torch.randperm(corrupt_candidates.size(0)).to(device)
    ][:int(0.15*corrupt_candidates.size(0))]

    shuffle_size = int(0.1*corrupt_candidates.size(0))
    mask_size = int(0.8*corrupt_candidates.size(0))

    shuffle_candidates = corrupt_candidates[:shuffle_size]
    mask_candidates = corrupt_candidates[shuffle_size:shuffle_size+mask_size]

    inputs[mask_candidates[:,0], mask_candidates[:,1]] = tokenizer.mask_token_id
    inputs[shuffle_candidates[:,0], shuffle_candidates[:,1]] = torch.randperm(tokenizer.vocab_size)[:shuffle_candidates.size(0)].to(device)

    return tokenized, inputs_backup

# next = next(iter(dataloader))
# a,b = process_batch(next["text"], tokenizer)
# (a["input_ids"] != b).nonzero()
