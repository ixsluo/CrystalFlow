import hydra
import omegaconf

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.run import finetune


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    finetune(cfg)


if __name__ == "__main__":
    main()
