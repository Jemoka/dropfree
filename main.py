import os
import ray
import ray.train.torch as rt
from transformers import AutoConfig, AutoTokenizer, AutoModel
from ray.train.torch import TorchTrainer, get_device
from trainer import Trainer

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='bert')
    parser.add_argument("experiment", help="name for the experiment", type=str)
    parser.add_argument("save_dir", help="path for ray to put logs and checkpoints to", type=str)
    parser.add_argument("--head_node", help="ip address of the headnode of the cluster", type=str, default=None)
    parser.add_argument("--checkpoints", default=3, type=int, help="how many checkpoints to keep")
    parser.add_argument("--dataset", default="cerebras/SlimPajama-627B", type=str, help="dataset")
    parser.add_argument("--batch_size", default=6, type=int, help="training batch size *PER WORKER*")
    parser.add_argument("--base", default="FacebookAI/xlm-roberta-large", type=str, help="base model configuration (and tokenizer) to use")
    parser.add_argument("--dropout", default=False, action="store_true", help="whether to enable dropout")
    parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="number of steps to warm up scheduler")
    args = parser.parse_args()
    
    if args.head_node:
        ray.init(args.head_node, num_cpus=1, num_gpus=1)
    else:
        ray.init(num_cpus=1, num_gpus=1, dashboard_host="0.0.0.0")
    scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=True)
    run_config = ray.train.RunConfig(storage_path="file://"+os.path.abspath(args.save_dir),
                                     name=args.experiment,
                                     checkpoint_config=ray.train.CheckpointConfig(
                                         num_to_keep=args.checkpoints,
                                         checkpoint_score_attribute="ppl",
                                         checkpoint_score_order="min"
                                     ))
    trainer = ray.train.torch.TorchTrainer(
        Trainer.execute(args),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    trainer.fit()
    ray.shutdown()

