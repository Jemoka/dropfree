import argparse

parser = argparse.ArgumentParser(prog='dropfree')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")

# intervals
parser.add_argument("--report_interval", default=64, type=int, help="save to wandb every this many steps")
parser.add_argument("--checkpoint_interval", default=8192, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=8192, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")

# model and data
parser.add_argument("--architecture", help="what model to start with?", type=str, default="EleutherAI/pythia-160m")

# hyperparameters
## training
parser.add_argument("--batch_size", help="batch size per GPU", type=int, default=32)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.0)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=1)

## optimizer
parser.add_argument("--lr", help="learning rate", type=float, default=0.0006)
parser.add_argument("--beta1", help="adam beta 1", type=float, default=0.9)
parser.add_argument("--beta2", help="adam beta 2", type=float, default=0.95)
parser.add_argument("--eps", help="adam epsilon", type=float, default=1.0e-8)
parser.add_argument("--weight_decay", help="AdamW weight decay", type=float, default=0.1)
parser.add_argument("--gradient_clip", help="AdamW gradient clip maximum", type=float, default=1.0)

## scheduler
parser.add_argument("--warmup_pct", help="learning rate warmup steps", type=float, default=0.01)
parser.add_argument("--decay_target_pct", help="learning rate decay target as a fraction of initial LR", type=float, default=0.1)



