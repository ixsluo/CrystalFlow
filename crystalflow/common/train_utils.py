from typing import Any

import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from crystalflow.common.data_utils import StandardScalerTorch
from crystalflow.models.property_embeddings import PropertyEmbedding


class AddConfigCallback(Callback):
    """Adds a copy of the config to the checkpoint, so that `load_from_checkpoint` can use it to instantiate everything."""
    def __init__(self, config: dict[str, Any]):
        self._config_dict = config

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self._config_dict


class SetPropertyScalers(Callback):
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage):  # Called when fit, validate, test, predict, or tune begins.
        if stage != 'fit':
            return
        model = pl_module.model  # real inner model
        property_embeddings: nn.ModuleDict = model.property_embeddings
        for _, property_embedding in property_embeddings.items():
            property_name: str = property_embedding.name
            scaler: StandardScalerTorch = property_embedding.scaler
            if scaler.initialized:
                continue
            print(f"Fitting scaler: {property_name} (device {pl_module.device}) ...")
            data = [batch[property_name] for batch in trainer.datamodule.train_dataloader()]
            scaler.fit(data)


class CheckUnusedParameters(Callback):
    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        datamodule = trainer.datamodule
        model = pl_module.model
        param_usage = {name: False for name, _ in model.named_parameters()}
        for name, param in model.named_parameters():
            param.register_hook(lambda grad, name=name: param_usage.update({name: True}))
        batch = next(iter(datamodule.train_dataloader())).to(pl_module.device)
        output = model(batch)
        loss = output['loss']
        loss.backward()
        unused_params = [name for name, used in param_usage.items() if not used]
        if unused_params:
            print(f"Unused parameters (device {pl_module.device}):")
            for name in unused_params:
                print(f"  {name}")
        else:
            print(f"No unused parameters found (device {pl_module.device}).")


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