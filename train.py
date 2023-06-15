import hydra
import lightning as L
import pyrootutils
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="fish_vocoder/configs", version_base="1.3")
def main(cfg: DictConfig):
    logger.add(f"{cfg.paths.run_dir}/train.log")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set seed to {cfg.seed}")

    data = instantiate(cfg.data)
    logger.info(f"Data: {data}")


if __name__ == "__main__":
    main()
