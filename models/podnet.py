from copy import deepcopy
from utils.augmentations import normalize
import torch
from torch import nn
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.egap_model import CasperModel
from utils.no_bn import bn_track_stats
import math
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--wd_reg', type=float, default=1e-5,
                        help='L2 regularization applied to the parameters.')
    parser.add_argument('--lambda_c', type=float, default=0.05,
                        help='L2 regularization applied to the parameters.')
    parser.add_argument('--lambda_f', type=float, default=0.01,
                        help='L2 regularization applied to the parameters.')
    parser.add_argument('--delta', type=float, default=0.6, )
    parser.add_argument('--k', type=int, default=10, )
    parser.add_argument('--scheduler', default=None, choices=['cosine', 'none'])
    parser.add_argument('--eta', type=float, default=1.)
    parser.add_argument('--scaling', type=float, default=3.)

    CasperModel.add_replay_args(parser)

    return parser


class PodNetClassifier(nn.Module):
    def __init__(self, in_features, out_features, k, scaling=1, eta=None):
        super(PodNetClassifier, self).__init__()
        self.in_features = in_features
        self.k = k
        self.out_features = out_features
        self.eta = nn.Parameter(torch.ones(1)) if eta is None else eta
        self.theta = nn.Parameter(torch.empty(in_features, k, out_features))
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        self.t = 0
        self.scaling = scaling

    def expand(self, num_classes):
        self.t += num_classes
        assert self.t <= self.out_features, "Trying to expand more than the maximum number of classes"

    def imprint(self, backbone, dataloader):
        feats, labels = [], []
        for data in dataloader:
            x, y = data[0], data[1]
            x = x.to(self.theta.device)
            y = y.to(self.theta.device)
            with torch.no_grad():
                try:
                    x = backbone.features(x)
                except:
                    _, x = backbone(x, returnt='both')
                x = F.normalize(x, dim=1, p=2)
                feats.append(x)
                labels.append(y)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        print()
        for c in tqdm(labels.unique(), desc='Imprinting'):
            class_feats = feats[labels == c]
            kmeans = KMeans(n_clusters=self.k).fit(class_feats.cpu())
            self.theta[:, :, c] = torch.from_numpy(kmeans.cluster_centers_).to(self.theta.device).T
        print()

    def our_forward(self, x, with_eta=False):
        x = F.normalize(x, dim=1)
        theta = F.normalize(self.theta[:, :, :self.t], dim=0)

        c = torch.einsum('bl,lkc->bkc', x, theta)  # b k c-t

        s = F.softmax(c, dim=1)  # b k c-t
        y = (c * s).sum(dim=1)  # b c-t
        if not with_eta:
            return y
        else:
            return y, self.eta

    def forward(self, x, with_eta=False):
        """Computes the pairwise distance matrix with numerical stability."""
        theta = self.theta[:, :, :self.t].permute(0, 2, 1).reshape(self.in_features, -1).permute(1,
                                                                                                 0)  # c-t in_features
        theta = self.scaling * F.normalize(theta, dim=-1, p=2)
        a = self.scaling * F.normalize(x, dim=-1, p=2)
        mat = torch.cat([a, theta])

        pairwise_distances_squared = torch.add(
            mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
            torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
        ) - 2 * (torch.mm(mat, torch.t(mat)))

        # Deal with numerical inaccuracies. Set small negatives to zero.
        pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

        # Get the mask where the zero distances are at.
        error_mask = torch.le(pairwise_distances_squared, 0.0)

        # Optionally take the sqrt.

        pairwise_distances = pairwise_distances_squared

        # Undo conditionally adding 1e-16.
        pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

        # Explicitly set diagonals to zero.
        mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

        simi_per_class = -pairwise_distances[:a.shape[0], a.shape[0]:]  # todo - reshape
        simi_per_class = simi_per_class.view(x.shape[0], simi_per_class.shape[1] // self.k, self.k)

        gamma = 1
        attentions = F.softmax(gamma * simi_per_class, dim=-1)
        out = (simi_per_class * attentions).sum(-1)
        if not with_eta:
            return out
        else:
            return out, self.eta


def icarl_fill_buffer(self: CasperModel, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, = buf_x[idx], buf_y[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader  # --> not_aug_dataloader(self.args.batch_size)
    mean, std = dataset.get_denormalization_transform().mean, dataset.get_denormalization_transform().std
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        try:
            feats = self.net.features(normalize(not_norm_x, mean, std)).float()
            outs = self.net.classifier(feats)
        except:
            outs, feats = self.net(normalize(not_norm_x, mean, std), returnt='both')
        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in range(classes_start, classes_end):
        idx = (a_y == _y)
        _x, _y = a_x[idx], a_y[idx]
        feats = a_f[idx]
        feats = feats.reshape(len(feats), -1)
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device)
            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)


class PodNetLoss(nn.Module):
    def __init__(self, delta, num_classes, device):
        super(PodNetLoss, self).__init__()
        self.delta = delta
        self.eye = torch.eye(num_classes).to(device).bool()

    def forward(self, y_hat, y, eta):
        y_map = self.eye[:y_hat.shape[1], :y_hat.shape[1]][y]
        y_hat = eta * (y_hat - self.delta)
        y_hat = y_hat - y_hat.max(1)[0].view(-1, 1)  # for numerical stability
        gts = (y_hat[y_map].reshape(len(y_hat))).exp()
        ngts = (y_hat[~y_map].reshape(len(y_hat), -1) * eta).exp().sum(1)
        return F.relu(-torch.log(gts / ngts)).mean()


class PodNet(CasperModel):
    NAME = 'podnet'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(PodNet, self).__init__(backbone, loss, args, transform)
        if self.args.scheduler == 'none':
            self.args.scheduler = None
        self.dataset = get_dataset(args)

        # Instantiate buffers
        # self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.N_CLASSES_PER_TASK *
                             self.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.num_classes = self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS

        self.net.classifier = PodNetClassifier(
            self.net.classifier.in_features, self.num_classes, self.args.k, scaling=args.scaling, eta=args.eta)

        self.flatnorm = lambda x, axis: x.pow(2).sum(axis).view(len(x), - 1)
        self.loss = PodNetLoss(self.args.delta, self.num_classes, self.device)

        self.i = 0
        self.lambda_c = 1.
        self.lambda_f = 1.

    def get_name(self):
        return 'Podnet' + self.get_name_extension()

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        # feats = self.net.features(x).squeeze()
        try:
            feats = self.net.features(x).float().squeeze()
        except:
            _, feats = self.net(x, returnt='both')
            feats = feats.float().squeeze()

        feats = feats.reshape(feats.shape[0], -1)
        feats = F.normalize(feats, dim=1, p=2)
        feats = feats.unsqueeze(1)

        pred = (F.normalize(self.class_means).unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, ref_logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.task > 0:
            with torch.no_grad():
                ref_feature_set = self.old_net.forward_all(inputs)
                ref_feats, ref_hs, ref_rawfeat = ref_feature_set['features'], ref_feature_set['attention'][:-1], \
                                                 ref_feature_set['raw_features']
        self.opt.zero_grad()

        feature_set = self.net.forward_all(inputs)
        feats, hs, rawfeat = feature_set['features'], feature_set['attention'][:-1], feature_set['raw_features']
        # using raw features as the input to the classifier
        logits, eta = self.net.classifier(feats, with_eta=True)

        class_loss = self.loss(logits, labels, eta=eta)
        loss = class_loss
        if self.task > 0:
            l_pod_spatial = torch.mean(torch.stack([self.pod_spatial_loss(h, ref_h) for h, ref_h in zip(hs, ref_hs)]))
            # l_pod_flat = (F.normalize(rawfeat, dim=1, p=2) - F.normalize(ref_rawfeat, dim=1, p=2)).pow(2).sum()
            l_pod_flat = F.cosine_embedding_loss(
                rawfeat, ref_rawfeat,
                torch.ones(rawfeat.shape[0]).to(rawfeat.device)
            )
            loss += self.lambda_c * l_pod_spatial + self.lambda_f * l_pod_flat

            if self.args.rep_minibatch > 0 and self.args.rho > 0:
                replay_loss = self.get_replay_loss()
                loss += replay_loss * self.args.rho

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.net.parameters(), 5)
        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def pod_spatial_loss(self, h, ref_h):
        h_w = self.flatnorm(h, axis=2)
        h_h = self.flatnorm(h, axis=3)
        h_w_ref = self.flatnorm(ref_h, axis=2)
        h_h_ref = self.flatnorm(ref_h, axis=3)
        hidden = F.normalize(torch.cat([h_w, h_h], dim=-1), dim=1, p=2)
        ref = F.normalize(torch.cat([h_w_ref, h_h_ref], dim=-1), dim=1, p=2)
        loss = torch.mean(torch.frobenius_norm(hidden - ref, dim=-1))

        return loss

    def begin_task(self, dataset):
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.wd_reg, momentum=self.args.lr_momentum)
        for p in self.net.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))
        if self.args.scheduler == 'cosine':
            print('Warning: overwriting policy with cosine annealing')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)
        self.net.classifier.expand(dataset.N_CLASSES_PER_TASK)
        if self.task > 0:
            self.lambda_c = self.args.lambda_c * (
                        dataset.N_CLASSES_PER_TASK * (self.task + 1) / self.N_CLASSES_PER_TASK) ** 0.5
            self.lambda_f = self.args.lambda_f * (
                        dataset.N_CLASSES_PER_TASK * (self.task + 1) / self.N_CLASSES_PER_TASK) ** 0.5
            with torch.no_grad():
                with bn_track_stats(self.old_net, False):
                    self.net.classifier.imprint(self.old_net, dataset.train_loader)
            dataset.train_loader.dataset.targets = np.concatenate(
                [dataset.train_loader.dataset.targets,
                 self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [dataset.train_loader.dataset.data, torch.stack([(
                        self.buffer.examples[i].type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [dataset.train_loader.dataset.data, torch.stack([((self.buffer.examples[i] * 255).type(
                        torch.uint8).cpu()) for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(1, 3)])

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            icarl_fill_buffer(self, self.buffer, dataset, self.task)
        self.task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                try:
                    feat = self.net.features(x_buf)
                except:
                    _, feat = self.net(x_buf, returnt='both')
                class_means.append(feat.mean(0).flatten())
        self.class_means = torch.stack(class_means)
