from typing import Any, Union
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import (
    Engine,
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

logging.basicConfig(
    format="%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s|%(message)s"
)


def setup_single_modal_trainer(
    text_path: bool,
    model: nn.Module,
    model_checkpont_path: str,
    load_epoch: int,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: Union[str, torch.device],
    num_epochs: int,
) -> Engine:

    if load_epoch != 0:
        model.load_state_dict(torch.load(model_checkpont_path) + f"_epoch_{load_epoch}")
        logging.debug("model paramenters has been loaded")

    def prepare_batch(batch: Any) -> Any:
        w, t, y = batch
        return w.to(device), y.to(device) if not text_path else t, y.to(device)

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device, prepare_batch=prepare_batch
    )

    return trainer


def setup_single_modal_evaluator(
    model: nn.Module,
    text_path: bool,
    criterion: nn.Module,
    device: Union[str, torch.device],
):

    precision = Precision(device=device, average=False)
    recall = Recall(device=device, average=False)
    f1score = (precision * recall * 2 / (precision + recall)).mean()
    val_metrics = {
        "accuracy": Accuracy(device=device),
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "loss": Loss(criterion),
    }

    def prepare_batch(batch: Any) -> Any:
        w, t, y = batch
        return w.to(device), y.to(device) if not text_path else t, y.to(device)

    train_evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device, prepare_batch=prepare_batch
    )
    val_evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device, prepare_batch=prepare_batch
    )

    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)
        metric.attach(val_evaluator, name)

    return train_evaluator, val_evaluator
