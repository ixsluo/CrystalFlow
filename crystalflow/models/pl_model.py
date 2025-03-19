import logging
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

import hydra
import torch
import lightning as pl
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer, AdamW

metriclogger = logging.getLogger("metrics")


class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""
    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    """Callable to instantiate a learning rate scheduler."""
    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def get_default_optimizer(params):
    return AdamW(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


class FlowLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_partial,
        scheduler_partials,
        *args,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self._optimizer_partial = optimizer_partial or get_default_optimizer
        self._scheduler_partials = scheduler_partials or []
        self.save_hyperparameters(ignore=("model", "optimizer_partial", "scheduler_partials"))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ):
        ckpt = torch.load(checkpoint_path)
        config = DictConfig(ckpt["config"])
        model=instantiate(config.pl_model.model)
        optimizer_partial=instantiate(config.pl_model.optimizer_partial)
        scheduler_partials=instantiate(config.pl_model.scheduler_partials)
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            model=model,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
            **kwargs,
        )

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer_partial(params=self.model.parameters())
        if self._scheduler_partials:
            lr_schedulers = [
                {
                    **scheduler_dict,
                    "scheduler": scheduler_dict["scheduler"](optimizer=optimizer),  # lazy here to init scheduler
                }
                for scheduler_dict in self._scheduler_partials
            ]

            return [optimizer], lr_schedulers
        else:
            return optimizer

    def forward(self, *args, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            return self.model(*args, **kwargs)

    def log_stats(self, loss_dict, prefix, **log_kwargs):
        log_dict = {f"{prefix}_{k}": v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, sync_dist=(self.trainer.num_devices > 1), **log_kwargs)

    def training_step(self, batch, batch_idx: int, dataloader_idx=0):
        output = self.model(batch)
        loss_dict = {k: v for k, v in output.items() if k.startswith("loss")}
        self.log_stats(loss_dict, prefix="train", on_step=None, batch_size=batch.batch_size)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0):
        output = self.model(batch)
        loss_dict = {k: v for k, v in output.items() if k.startswith("loss")}
        self.log_stats(loss_dict, prefix="val", on_step=None, batch_size=batch.batch_size)
        return loss_dict["loss"]

    def test_step(self, batch, batch_idx: int, dataloader_idx=0):
        output = self.model(batch)
        loss_dict = {k: v for k, v in output.items() if k.startswith("loss")}
        self.log_stats(loss_dict, prefix="test", on_step=None, batch_size=batch.batch_size)
        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_validation_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_test_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")
