from copy import deepcopy

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomApply

from backbone.SupCon_Resnet import SupConResNet
from utils.augmentations import normalize
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.no_bn import bn_track_stats
import numpy as np

from utils.supconloss import SupConLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via SCR.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--temp', type=float, required=True,
                        help='Temperature for loss.')
    parser.add_argument('--head', type=str, required=False, default='mlp')
    parser.add_argument('--backbone', type=str, required=False, default='resnet18', choices=['resnet18', 'lopeznet',
                                                                                             'efficientnet'])
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--supcon_weight', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--trans_twice', type=int, default=0)
    
    return parser


input_size_match = {
    'seq-cifar100': [3, 32, 32],
}


class SCRDerpp(ContinualModel):
    NAME = 'scr_derpp'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        if args.minibatch_size is None:
            args.minibatch_size = args.batch_size
        backbone = SupConResNet(head=args.head, backbone=args.backbone)
        super(SCRDerpp, self).__init__(backbone, loss, args, transform)
        self.class_means = None
        self.dataset = get_dataset(args)

        self.denorm = self.dataset.get_denormalization_transform()
        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')
        self.transform_scr = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.args.dataset][1], input_size_match[self.args.dataset][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            # ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2)

        )

        self.temp = args.temp
        self.supconloss = SupConLoss(temperature=self.args.temp)
        self.current_task = 0

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        self.opt.zero_grad()
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        # cross entropy loss on the current task
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        # get different data from buffer for every regularization term
        if not self.buffer.is_empty():
            # supcon loss
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size)
            transformed_inputs = self.transform_scr(buf_inputs)
            if self.args.trans_twice:
                buf_inputs = self.transform_scr(buf_inputs)
            pred = torch.cat([self.net.forward_scr(buf_inputs).unsqueeze(1),
                              self.net.forward_scr(transformed_inputs).unsqueeze(1)],
                             dim=1)
            supconloss = self.supconloss(pred, buf_labels)
            self.wb_log['supcon_loss'] = supconloss.item()
            loss += supconloss * self.args.supcon_weight

            # derpp loss
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            self.wb_log['derpp_loss'] = derpp_loss.item()
            loss += derpp_loss
        
        loss.backward()
        self.opt.step()
        loss = loss.item()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                        labels=labels,
                                        logits=outputs.data)

        return loss

    def end_task(self, dataset) -> None:
        self.current_task += 1
        self.class_means = None
