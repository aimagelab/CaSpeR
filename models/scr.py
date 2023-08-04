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
    return parser


input_size_match = {
    'seq-cifar100': [3, 32, 32],
}


class SCR(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        if args.minibatch_size is None:
            args.minibatch_size = args.batch_size
        backbone = SupConResNet(head=args.head, backbone=args.backbone)
        super(SCR, self).__init__(backbone, loss, args, transform)
        self.class_means = None
        self.dataset = get_dataset(args)

        self.denorm = self.dataset.get_denormalization_transform()
        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.args.dataset][1], input_size_match[self.args.dataset][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            # ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2)

        )

        self.temp = args.temp
        self.loss = SupConLoss(temperature=self.args.temp)
        self.current_task = 0

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        x = torch.stack([self.denorm(_x) for _x in x])
        # feats = self.net.features(x).squeeze()
        feats = self.net.features(x).float().squeeze()

        feats = feats.reshape(feats.shape[0], -1)
        feats = F.normalize(feats, dim=1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())
        loss = 0
        self.class_means = None
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)
            comb_inputs = torch.cat((not_aug_inputs, buf_inputs))
            comb_transformed_inputs = self.transform(comb_inputs)
            comb_labels = torch.cat((labels, buf_labels))
            pred = torch.cat([self.net.forward_scr(comb_inputs).unsqueeze(1),
                              self.net.forward_scr(comb_transformed_inputs).unsqueeze(1)],
                             dim=1)
            loss = self.loss(pred, comb_labels)
            loss.backward()

            self.opt.step()
            loss = loss.item()
        self.buffer.add_data(not_aug_inputs, labels)

        return loss

    def end_task(self, dataset) -> None:
        self.current_task += 1
        self.class_means = None

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        class_means = []
        examples, labels = self.buffer.get_all_data(None)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                all_feats = self.net.features(x_buf).squeeze()
                if x_buf.shape[0] == 1:
                    all_feats = all_feats.unsqueeze(0)
                all_feats = all_feats / all_feats.norm(dim=1, keepdim=True)

                class_means.append(all_feats.mean(0).flatten())
            self.class_means = torch.stack(class_means)
            self.class_means = self.class_means / self.class_means.norm(dim=1, keepdim=True)
