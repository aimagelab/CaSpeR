

import importlib
import numpy as np
import os
import sys
import socket
import wandb

conf_path = os.getcwd()
sys.path.insert(0, conf_path)
# print(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.conf import set_random_seed
from utils import create_if_not_exists
import torch

import uuid
import datetime


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]

    if args.set_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.set_device

    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # TODO remove
    if 'effnet_variant' in vars(args) and args.effnet_variant is not None:
        os.environ['EFF_VAR'] = args.effnet_variant
        print("WILL USE VARIANT ", os.environ['EFF_VAR'])
    # ---

    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    # args.conf_git_commit = os.popen(r"git log | head -1 | sed s/'commit '//").read().strip()
    # if not os.path.isdir('gitdiffs'):
    #     create_if_not_exists("gitdiffs")
    # os.system('git diff > gitdiffs/diff_%s.txt' % args.conf_jobnum)
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    if socket.gethostname().startswith('go') or socket.gethostname() == 'jojo' or socket.gethostname() == 'yobama' or socket.gethostname() == 'dragon':
        import setproctitle
        setproctitle.setproctitle(
            '{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    train(model, dataset, args)


if __name__ == '__main__':
    main()
