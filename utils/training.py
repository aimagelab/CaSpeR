

import torch
from utils.status import progress_bar, create_stash
import numpy as np
from utils.loggers import *
from utils.loggers import CsvLogger, DictxtLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from utils.metrics import forgetting
from datasets import get_dataset
import sys


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        #csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
        csv_logger = DictxtLogger(dataset.SETTING, dataset.NAME, model.NAME)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.reset_scheduler()
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        for epoch in range(args.n_epochs):
            if model.NAME == 'joint':
                break
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)
                model.wblogger({'training': {'task': t, 'loss': loss, **model.wb_log}})
                model.wb_log = {}

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
            model.scheduler_step()
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        mean_acc = np.mean(accs, axis=1)
        if hasattr(model, 'log_accs'):
            model.log_accs(accs)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        cil_acc, til_acc = np.mean(accs, axis=1).tolist()
        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': t,
            **model.wb_log,
        }
        model.log_results.append(log_obj)
        model.wblogger({'testing': log_obj})
        if args.save_checks:
            model.save_checkpoint()
        if t == model.N_TASKS - 1 and args.custom_log:
            model.save_logs()
        model.wb_log = {}

        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
            csv_logger.log_fullacc(accs)

    if args.csv_log:
        csv_logger.write(vars(args))
