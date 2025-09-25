import sys
sys.path.append('.')
import textwrap
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import lightning as pl
import wandb
import swanlab
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything, Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.profilers import SimpleProfiler as Profiler
from lightning.pytorch.loggers import WandbLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT



def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if HydraConfig.initialized():
        ckpt_dir = str(Path(HydraConfig.get().run.dir).absolute())
    else:
        ckpt_dir = None
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    callbacks.append(
        TQDMProgressBar(
            refresh_rate=cfg.logging.progress_bar_refresh_rate,
        )
    )

    return callbacks


def get_wandb_logger(cfg, save_dir):
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            save_dir=save_dir,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
    else:
        hydra.utils.log.info("Not using <WandbLogger>")
    return wandb_logger


def get_swanlab_logger(cfg, save_dir):
    swanlab_logger = None
    if "swanlab" in cfg.logging:
        hydra.utils.log.info("Instantiating <SwanLabLogger>")
        swanlab_config = cfg.logging.swanlab
        swanlab_logger = SwanLabLogger(
            **swanlab_config,
            logdir=save_dir / "swanlab",
            settings=swanlab.Settings(),
            tags=cfg.core.tags,
        )
    else:
        hydra.utils.log.info("Not using <SwanLabLogger>")
    return swanlab_logger


def ammend_scalers_file(config, origin_path=None, save_path=None):
    """copy scaler files and ammend prop_scalers.pt from list-type to dict-type"""
    if origin_path is None:
        return

    lattice_scaler = torch.load(Path(origin_path) / 'lattice_scaler.pt')
    torch.save(lattice_scaler, Path(save_path) / 'lattice_scaler.pt')

    scaler = torch.load(Path(origin_path) / 'prop_scaler.pt')
    torch.save(scaler, Path(save_path) / 'prop_scaler.pt')

    scalers = torch.load(Path(origin_path) / 'prop_scalers.pt')
    if isinstance(scalers, list):
        hydra.utils.log.info("ammending prop_scalers.pt from list-type to dict-type")
        scalers = {key: scaler for key, scaler in zip(config.data.properties, scalers, strict=True)}
    elif isinstance(scalers, dict):
        pass
    else:
        raise ValueError("Unknown stored scalers type")
    torch.save(scalers, Path(save_path) / 'prop_scalers.pt')



def get_datamodule(cfg, scaler_path=None):
    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, scaler_path=scaler_path, _recursive_=False
    )
    return datamodule


def get_model(cfg_model, cfg_optim, cfg_data, cfg_logging, ckpt=None):
    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg_model._target_}>")
    if ckpt is None:
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg_model,
            optim=cfg_optim,
            data=cfg_data,
            logging=cfg_logging,
            _recursive_=False,
        )
    else:
        model_class = hydra.utils.get_class(cfg_model._target_)
        model = model_class.load_from_checkpoint(
            ckpt,
            optim=cfg_optim,
            data=cfg_data,
            logging=cfg_logging,
        )
    return model


def pass_scaler_to_model(model, datamodule):
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
        model.scalers = {key: scaler.copy() for key, scaler in datamodule.scalers.items()}


def save_scaler(datamodule, save_dir):
    torch.save(datamodule.lattice_scaler, save_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, save_dir / 'prop_scaler.pt')
    torch.save(datamodule.scalers, save_dir / 'prop_scalers.pt')



def find_ckpt(ckpt_dir) -> str | None:
    ckpts = list(ckpt_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
        # ckpt = Path(ckpt)
    else:
        ckpt = None
    return ckpt


def find_cfg(ckpt_dir) -> DictConfig:
    hparams_file = Path(ckpt_dir).joinpath("hparams.yaml")
    if not hparams_file.exists():
        raise FileNotFoundError(f"{hparams_file}")
    return OmegaConf.load(hparams_file)


def save_cfg(cfg, save_dir):
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (save_dir / "hparams.yaml").write_text(yaml_conf)


def find_finetune_schedule(cfg, finetune_dir, model=None):
    ft_schedule = finetune_dir / "ft_schedule.yaml"
    gen_ft_sched = cfg.train.get("gen_ft_sched", False)
    if gen_ft_sched:
        hydra.utils.log.info("Generating finetune schedure.")
        if (gen_ft_sched != "overwrite") and ft_schedule.exists():
            raise FileExistsError(
                f"Finetune schedule file {ft_schedule} is already exists. "
                "Refuse to continue.\n"
                "Or use `+train.gen_ft_sched=overwrite` to force overwrite."
            )
        with open(ft_schedule, "w") as f:
            f.write(textwrap.dedent(f"""\
                0:  # finetune phase index
                  # max_transition_epoch: 3
                  params:
                  # - model.albert.pooler.*  # regex is allowed"""))
            for name, param in model.named_parameters():
                f.write(f"  - {name}\n")
        raise SystemExit(f"Finetune scheduler template written in {ft_schedule}. You should edit it and then run again to start finetune.")
    else:
        if not Path(ft_schedule).exists():
            raise FileNotFoundError(
                f"Finetune schedule file {ft_schedule} is not found. "
                "You may use `+train.gen_ft_sched=true` to generate a schedule with all layer names."
            )
    return ft_schedule


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    # Hydra run directory
    run_dir = Path(HydraConfig.get().run.dir)

    torch.set_float32_matmul_precision(cfg.train.float32_matmul_precision)
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"
        cfg.logging.swanlab.mode = "offline"
    save_cfg(cfg, run_dir)

    datamodule = get_datamodule(cfg, scaler_path=None)
    model = get_model(cfg.model, cfg.optim, cfg.data, cfg.logging)
    pass_scaler_to_model(model, datamodule)
    save_scaler(datamodule, run_dir)

    # Logger instantiation/configuration
    wandb_logger = get_wandb_logger(cfg, run_dir)
    hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
    wandb_logger.watch(
        model,
        log=cfg.logging.wandb_watch.log,
        log_freq=cfg.logging.wandb_watch.log_freq,
    )
    swanlab_logger = get_swanlab_logger(cfg, run_dir)
    loggers = [logger for logger in [wandb_logger, swanlab_logger] if logger is not None]

    hydra.utils.log.info("Instantiating the Trainer")
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=loggers,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        profiler=Profiler(dirpath=run_dir, filename="time_report"),
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    ckpt = find_ckpt(run_dir)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    if not cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info("Starting testing!")
        trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
    if swanlab_logger is not None:
        swanlab_logger.experiment.finish()


def finetune(cfg):
    from finetuning_scheduler import FinetuningScheduler

    if "finetune_from" not in cfg.train:
        raise ValueError("`train.finetune_from` must be specified in finetune mode")
    finetune_from_dir = Path(cfg.train.finetune_from).absolute()
    # Hydra run directory
    run_dir = Path(HydraConfig.get().run.dir)

    torch.set_float32_matmul_precision(cfg.train.float32_matmul_precision)
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"
        cfg.logging.swanlab.mode = "offline"


    # cfg.model = ori_cfg.model  # overwrite model config
    save_cfg(cfg, run_dir)

    ori_cfg = find_cfg(finetune_from_dir)
    ammend_scalers_file(ori_cfg, finetune_from_dir, run_dir)
    datamodule = get_datamodule(cfg, scaler_path=run_dir)
    ckpt = find_ckpt(finetune_from_dir)
    model = get_model(cfg.model, cfg.optim, cfg.data, cfg.logging, ckpt=ckpt)
    pass_scaler_to_model(model, datamodule)
    save_scaler(datamodule, run_dir)

    ft_schedule = find_finetune_schedule(cfg, run_dir, model)

    # Logger instantiation/configuration
    wandb_logger = get_wandb_logger(cfg, run_dir)
    hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
    wandb_logger.watch(
        model,
        log=cfg.logging.wandb_watch.log,
        log_freq=cfg.logging.wandb_watch.log_freq,
    )
    swanlab_logger = get_swanlab_logger(cfg, run_dir)
    loggers = [logger for logger in [wandb_logger, swanlab_logger] if logger is not None]

    callbacks = build_callbacks(cfg)
    callbacks.append(FinetuningScheduler(ft_schedule=ft_schedule))
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=loggers,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        profiler=Profiler(dirpath=run_dir, filename="time_report"),
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    print(trainer.early_stopping_callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    #if not cfg.train.pl_trainer.fast_dev_run:
    #    hydra.utils.log.info("Starting testing!")
    #    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
    if swanlab_logger is not None:
        swanlab_logger.experiment.finish()

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    if cfg.train.get("finetune_from", False) is False:
        run(cfg)
    elif cfg.train.get("finetune_from", False):
        finetune(cfg)
    else:
        raise ValueError("Unknown start setting")


if __name__ == "__main__":
    main()
