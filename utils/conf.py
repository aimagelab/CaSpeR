
import socket
import os
import random
import torch
import numpy as np
import os
from utils import create_if_not_exists

def get_device(jobs_per_gpu=10) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies.
    """
    return './data/'


def base_path_dataset() -> str:
    """
    Returns the base bath where to store datasets.
    """
    return './data/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
