import pprint
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="cbis_ddsm_concept_bottleneck_joint", version_base=None)
def main(cfg: omegaconf.DictConfig):
    print(cfg, cfg.__class__.__name__)

    if cfg.train.deterministic:
        pl.seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(omegaconf.OmegaConf.to_container(cfg))

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra.utils.log.info(f"Experiment directory {hydra_dir}")
    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <" f"{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)

    hydra.utils.log.info(f"Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        deterministic=cfg.train.deterministic,
        **cfg.train.pl_trainer,
    )

    hydra.utils.log.info(f"Starting training!")
    hydra.utils.log.info(
        f"It will run {cfg.train.pl_trainer.max_epochs} " f"epochs for training!"
    )
    trainer.fit(model=model, datamodule=datamodule)

    # hydra.utils.log.info(f"Starting testing!")
    # trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
