from pprint import pprint

import hydra
import lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from crystalflow.common.globals import PACKAGE_ROOT


@hydra.main(str(PACKAGE_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: DictConfig):
    pl_model: pl.LightningModule = instantiate(config.pl_model)
    pl_data: pl.LightningDataModule = instantiate(config.data)
    trainer: pl.Trainer = instantiate(config.trainer)
    if config.data.get('debug', False) or True:
        pprint(OmegaConf.to_container(config, resolve=True))
        print(pl_model)
        #print(pl_model.optimizers)
        print("For debug, exit")
        return
    trainer.strategy.connect(pl_model)
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.pytorch.loggers.WandbLogger):
        # Log the config to wandb so that it shows up in the portal.
        trainer.logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
        trainer.logger.experiment.watch(pl_model, log="all", log_freq=100)


    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.pytorch.loggers.WandbLogger):
        trainer.logger.experiment.unwatch(pl_model)
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    main()