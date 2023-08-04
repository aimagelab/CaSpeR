

from enum import unique
import torch
import numpy as np
from typing import Tuple
from torchvision import transforms

def reservoir(num_seen_examples: int, buffer_size: int, **kwargs) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :param labels: the set of buffer labels
    :param proposed_class: the class of the current example
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def balancoir(num_seen_examples: int, buffer_size: int, labels: np.array, proposed_class: int, unique_map: np.array) -> int:
    """
    balancoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :param labels: the set of buffer labels
    :param proposed_class: the class of the current example
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples
    
    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size or len(unique_map) <= proposed_class or unique_map[proposed_class] < np.median(unique_map[unique_map > 0]):
        target_class = np.argmax(unique_map)
        e = rand % unique_map.max()
        idx = np.arange(buffer_size)[labels.cpu() == target_class][rand % unique_map.max()]
        return idx
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir', 'balancoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.sampling_policy = {'reservoir': reservoir, 'ring': reservoir, 'balancoir': balancoir}[mode]
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

        self.unique_map = np.empty((0,), dtype=np.int32)

    def update_unique_map(self, label_in, label_out=None):
        while len(self.unique_map) <= label_in:
            self.unique_map = np.concatenate((self.unique_map, np.zeros((len(self.unique_map) * 2 + 1), dtype=np.int32)), axis=0)
        self.unique_map[label_in] += 1
        if label_out is not None:
            self.unique_map[label_out] -= 1

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = self.sampling_policy(self.num_seen_examples, self.buffer_size, unique_map=self.unique_map,
                        labels=self.labels if hasattr(self, 'labels') else None, proposed_class=labels[i])
            
            if index >= 0:
                self.update_unique_map(labels[i], self.labels[index] if index < self.num_seen_examples else None)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
            self.num_seen_examples += 1

    def get_data(self, size: int, transform: transforms=None, return_index=False)-> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_balanced_data(self, size: int, transform: transforms=None, n_classes=-1) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        tot_classes, class_counts = torch.unique(self.labels[:self.num_seen_examples], return_counts=True)
        if n_classes == -1:
            n_classes = len(tot_classes)

        
        finished = False
        selected = tot_classes
        while not finished:
            n_classes = min(n_classes, len(selected))
            size_per_class = torch.full([n_classes], size // n_classes)
            size_per_class[:size % n_classes] += 1
            selected = tot_classes[class_counts >= size_per_class[0]]
            if n_classes <= len(selected):
                finished = True
            if len(selected) == 0:
                print('WARNING: no class has enough examples')
                return self.get_data(0, transform)

        selected = selected[torch.randperm(len(selected))[:n_classes]]

        choice = []
        for i, id_class in enumerate(selected):
            choice += np.random.choice(torch.where(self.labels[:self.num_seen_examples] == id_class)[0].cpu(),
                                       size=size_per_class[i].item(),
                                       replace=False).tolist()
        choice = np.array(choice)

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def is_full(self) -> bool:
        """
        Returns true if the buffer is full, false otherwise.
        """
        if self.num_seen_examples >= self.buffer_size:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self
