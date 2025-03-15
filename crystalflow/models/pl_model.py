import logging
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar, Union

import hydra
import lightning as pl

import torch.nn as nn
from torch.optim import Optimizer, AdamW

metriclogger = logging.getLogger("metrics")


# protocal
class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


# protocal
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

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer_partial(params=self.model.parameters())
        if self._scheduler_partials:
            lr_schedulers = [
                {
                    "scheduler": scheduler_dict["scheduler"](optimizer=optimizer),
                    **scheduler_dict,
                }
                for scheduler_dict in self._scheduler_partials
            ]

            return [optimizer], lr_schedulers
        else:
            return optimizer


    def compute_stats(self, output_dict, prefix, **log_kwargs):
        loss = output_dict["loss"]
        log_dict = {f"{prefix}_{k}": v for k, v in output_dict.items()}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, **log_kwargs)
        return loss

    def training_step(self, batch, batch_idx: int, dataloader_idx=0):
        output_dict = self.model(batch)
        return self.compute_stats(output_dict, prefix="train", on_step=True, batch_size=batch.batch_size)

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0):
        output_dict = self.model(batch)
        return self.compute_stats(output_dict, prefix="val", on_step=True, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx: int, dataloader_idx=0):
        output_dict = self.model(batch)
        return self.compute_stats(output_dict, prefix="test", on_step=None, batch_size=batch.batch_size)

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
