from loguru import logger

from trainer import Trainer
from parameters import parser

from pathlib import Path

@logger.catch
def execute(args):
    if Path(str(args.warm_start)).exists():
        # by default, the from_pretrained function disables
        # whatever wandb settings was there b/c we usually
        # use this to load an existing model, but when we are
        # actually training, we want to actually enable it
        trainer = Trainer.from_pretrained(args.warm_start,
                                          disable_wandb=False)
    else:
        trainer = Trainer(args)

    trainer.train()

def configure(experiment, **kwargs):
    """configure a run from arguments

    Arguments
    ----------
        experiment : str
                experiment name
        kwargs : dict
                arguments to configure the run

    Returns
    -------
        SimpleNamespace
                configuration object
    """

    # listcomp grossery to parse input string into arguments that's
    # readable by argparse

    try:
        return parser.parse_args(([str(experiment)]+
        [j for k,v in kwargs.items() for j in ([f"--{k}", str(v)]
        if not isinstance(v, bool) else [f"--{k}"])]))
    except SystemExit as e:
        logger.error("unrecognized arguments found in configure: {}", kwargs)
        return None




