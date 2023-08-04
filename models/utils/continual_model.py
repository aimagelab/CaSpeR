

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from datasets import get_dataset
from utils.wandbsc import WandbLogger, innested_vars
from utils.conf import base_path
import os
from torch.optim.lr_scheduler import MultiStepLR
import pickle


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.opt = SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.lr_momentum)
        self.scheduler = None

        dataset = get_dataset(args)
        self.N_TASKS = dataset.N_TASKS
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.dataset_name = dataset.NAME
        self.N_CLASSES = self.N_TASKS * self.N_CLASSES_PER_TASK

        self.args.name = self.get_name()
        self.wblogger = WandbLogger(self.args, name=self.args.name)
        self.log_results = []
        self.wb_log = {}
        self.task = 0

    def get_name(self):
        return self.NAME.capitalize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def end_task(self, dataset):
        self.task += 1

    def log_accs(self, accs):
        pass

    def save_checkpoint(self):
        log_dir = os.path.join(base_path(), 'checkpoints', self.dataset_name, f'{self.args.name}-{self.wblogger.run_id}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if self.task == 1:
            with open(os.path.join(log_dir, 'args.pyd'), 'w') as f:
                f.write(str(innested_vars(self.args)))
        torch.save(self.net.state_dict(), f'{log_dir}/task_{self.task}.pt')
        return log_dir

    def save_logs(self):
        log_dir = os.path.join(base_path(), 'logs', self.dataset_name, self.args.name)
        # obj = {**vars(self.args), 'results': self.log_results}
        # self.print_logs(log_dir, obj, name='results')
        obj = {**innested_vars(self.args), 'results': self.log_results}
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filename = f'{self.wblogger.run_id}.pyd'
        with open(os.path.join(log_dir, filename), 'a') as f:
            f.write(str(obj) + '\n')
        return log_dir

    def reset_scheduler(self):
        if len(self.args.lr_decay_steps) > 0 and self.args.n_epochs > 1:
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.lr_momentum)
            self.scheduler = MultiStepLR(self.opt, milestones=self.args.lr_decay_steps, gamma=self.args.lr_decay)

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()
