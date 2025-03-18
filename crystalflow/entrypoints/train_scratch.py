import yaml
from pprint import pprint

import hydra
import lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.cli import SaveConfigCallback
from omegaconf import DictConfig, OmegaConf

from crystalflow.common.globals import PACKAGE_ROOT
from crystalflow.common.train_utils import AddConfigCallback, WandbWatcher


class SimpleParser:
    def save(self, config, path, **_):
        with open(path, "w") as f:
            yaml.dump(config, f)


@hydra.main(str(PACKAGE_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: DictConfig):
    config_as_dict = OmegaConf.to_container(config, resolve=True)

    pl_model: pl.LightningModule = instantiate(config.pl_model)
    pl_data: pl.LightningDataModule = instantiate(config.data)
    trainer: pl.Trainer = instantiate(config.trainer)
    trainer.callbacks.append(SaveConfigCallback(parser=SimpleParser(), config=config_as_dict, overwrite=True))
    trainer.callbacks.append(AddConfigCallback(config_as_dict))
    if config.data.get('debug', False):
        pprint(config_as_dict)
        print(pl_model)
        print(pl_model.model.type_model)
        print(pl_model.model.vfield)
        #trainer.strategy.connect(pl_model)
        #print(pl_model.optimizers)
        print("For debug, exit")
        return

    with WandbWatcher(trainer, pl_model, config_as_dict):
        trainer.fit(pl_model, datamodule=pl_data)


if __name__ == "__main__":
    main()