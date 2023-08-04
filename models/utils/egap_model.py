from email.policy import default
import torch

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from utils.spectral_analysis import calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs
import os
import pickle


class CasperModel(ContinualModel):

    @staticmethod
    def add_replay_args(parser):
        parser.add_argument('--rep_minibatch', type=int, default=None,
                            help='Size of minibatch for casper.')
        parser.add_argument('--replay_mode', type=str, default='casper',
                            choices=['none', 'casper'])

        parser.add_argument('--rho', type=float, default=0.01, help='Weight of casper.')
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        parser.add_argument('--p', default=None, type=int, help='number of classes to be drawn from the buffer')
        return parser

    def __init__(self, backbone, loss, args, transform):
        if args.minibatch_size is None:
            args.minibatch_size = args.batch_size
        if args.rep_minibatch is None:
            args.rep_minibatch = args.batch_size
        if args.rep_minibatch < 0:
            args.rep_minibatch = args.buffer_size
        if args.replay_mode == 'none' or args.rho == 0:
            args.replay_mode = 'none'
            args.rho = 0
        super(CasperModel, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')

        self.nc = self.args.p if self.args.p is not None else self.N_CLASSES_PER_TASK

    def get_name(self):
        return self.NAME.capitalize() + self.get_name_extension()

    def get_name_extension(self):
        name = '' if self.args.replay_mode == 'none' else 'Casper'
        if self.args.rho == 0:
            return name
        # if len(self.args.replay_mode) > 4 and self.args.replay_mode[4] == 'B':
        #     name += f'P{self.args.p if self.args.p is not None else self.N_CLASSES_PER_TASK}'
        # name += f'K{self.args.knn_laplace}'
        return name

    def get_replay_loss(self):
        if self.args.replay_mode == 'none':
            return torch.tensor(0., dtype=torch.float, device=self.device)
        if self.args.rep_minibatch == self.args.buffer_size:
            buffer_data = self.buffer.get_all_data(self.transform)
        else:
            buffer_data = self.buffer.get_balanced_data(self.args.rep_minibatch, transform=self.transform,
                                                        n_classes=self.nc)
        inputs, labels = buffer_data[0], buffer_data[1]
        features = self.net.features(inputs)

        dists = calc_euclid_dist(features)
        A, D, L = calc_ADL_knn(dists, k=self.args.knn_laplace, symmetric=True)

        L = torch.eye(A.shape[0], device=A.device) - normalize_A(A, D)

        n = self.nc
        # evals = torch.linalg.eigvalsh(L)
        evals, _ = find_eigs(L, n_pairs=min(2*n, len(L)))

        #gaps = evals[1:] - evals[:-1]

        if self.args.replay_mode == 'casper':
            return evals[:n + 1].sum() - evals[n + 1]

    def save_checkpoint(self):
        log_dir = super().save_checkpoint()
        ## pickle the future_buffer
        with open(os.path.join(log_dir, f'task_{self.task}_buffer.pkl'), 'wb') as f:
            self.buffer.to('cpu')
            pickle.dump(self.buffer, f)
            self.buffer.to(self.device)
