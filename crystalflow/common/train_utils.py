from typing import Any

import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class AddConfigCallback(Callback):
    """Adds a copy of the config to the checkpoint, so that `load_from_checkpoint` can use it to instantiate everything."""

    def __init__(self, config: dict[str, Any]):
        self._config_dict = config

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self._config_dict


class WandbWatcher:
    def __init__(self, trainer: pl.Trainer, pl_module: pl.LightningModule, config: dict|None = None):
        self.trainer = trainer
        self.pl_module = pl_module
        self.config = config if config is not None else {}

    def __enter__(self):
        if rank_zero_only.rank == 0 and isinstance(self.trainer.logger, pl.pytorch.loggers.WandbLogger):
            # Log the config to wandb so that it shows up in the portal.
            self.trainer.logger.experiment.config.update(self.config)
            self.trainer.logger.experiment.watch(self.pl_module, log="all", log_freq=100)

    def __exit__(self, exc_type, exc_value, traceback):
        if rank_zero_only.rank == 0 and isinstance(self.trainer.logger, pl.pytorch.loggers.WandbLogger):
            self.trainer.logger.experiment.unwatch(self.pl_module)
            self.trainer.logger.experiment.finish()