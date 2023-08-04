
from datetime import datetime
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, default='seq-cifar100',
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--lr_momentum', type=float, default=0,)
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='Learning rate decay.')
    parser.add_argument('--lr_decay_steps', type=lambda s: [] if s == '' else [int(v) for v in s.split(',')],
                        default='', help='Epochs at which lr is multiplied by lr_decay.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wb_prj', type=str, default='',
                        help='Wandb project')
    parser.add_argument('--wb_entity', type=str, default='',
                        help='Watdb entity')
    parser.add_argument('--custom_log', action='store_true',
                        help='Enable log (custom for each model, must be implemented)')
    parser.add_argument('--save_checks', action='store_true',
                        help='Save checkpoints')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--set_device', default=None, type=str)


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=None,
                        help='The batch size of the memory buffer.')

