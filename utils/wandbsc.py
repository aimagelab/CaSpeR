import wandb
from argparse import Namespace
from utils import random_id


def innested_vars(args: Namespace):
    new_args = vars(args).copy()
    for key, value in new_args.items():
        if isinstance(value, Namespace):
            new_args[key] = innested_vars(value)
    return new_args


class WandbLogger:
    def __init__(self, args: Namespace, name=None):
        self.active = args.wandb
        self.run_id = random_id(5)
        if self.active:
            prj = args.wb_prj
            entity = args.wb_entity
            if name is not None:
                name += f'-{self.run_id}'
            wandb.init(project=prj, entity=entity, config=innested_vars(args), name=name)

    def __call__(self, obj: any, **kwargs):
        if wandb.run:
            wandb.log(obj, **kwargs)
