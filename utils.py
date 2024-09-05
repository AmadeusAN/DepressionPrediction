import logging
from argparse import ArgumentParser
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    global_step_from_engine,
    ModelCheckpoint,
)
from ignite.handlers.early_stopping import EarlyStopping
from ignite.utils import setup_logger
from omegaconf import OmegaConf


def get_default_parser():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Config file path")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["nccl", "gloo"],
        type=str,
        help="DDP backend",
    )
    return parser


def setup_config(parser=None, config_path: str = None):
    if parser is None:
        parser = get_default_parser()

    args = parser.parse_args()
    if args.config != None:
        config_path = args.config
    config = OmegaConf.load(config_path)
    config.backend = args.backend

    return config


def setup_logging(config: Any) -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.
    engine 中自带了 logger 属性，需要传递过去。

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`
    """
    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset}",
        level=logging.DEBUG if config.debug else logging.INFO,
    )
    return logger


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: Any,
    to_save_train: Optional[dict] = None,
    to_save_eval: Optional[dict] = None,
):
    """Setup Ignite handlers."""

    ckpt_handler_train = ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config.output_dir / "checkpoints", require_empty=False)
    ckpt_handler_train = ModelCheckpoint(
        to_save_train,
        saver,
        filename_prefix=config.filename_prefix,
        n_saved=config.n_saved,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.save_every_iters),
        ckpt_handler_train,
    )
    global_step_transform = None
    if to_save_train.get("trainer", None) is not None:
        global_step_transform = global_step_from_engine(to_save_train["trainer"])
    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix="best",
        n_saved=config.n_saved,
        global_step_transform=global_step_transform,
        score_name="eval_accuracy",
        score_function=Checkpoint.get_default_score_fn("eval_accuracy"),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    # early stopping
    def score_fn(engine: Engine):
        return -engine.state.metrics["eval_loss"]

    es = EarlyStopping(config.patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)
    return ckpt_handler_train, ckpt_handler_eval
