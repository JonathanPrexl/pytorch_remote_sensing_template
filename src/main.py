import os
import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base = None,
            config_path="./configs/",
            config_name="default")
def main(config : DictConfig) -> None:

    # we first have to set visible
    # devices before importing any torch libs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)

    # do fancy AI stuff
    trainer = hydra.utils.instantiate(config.trainroutine)
    trainer = trainer(config)
    trainer.fit()
    trainer.finalize()

if __name__  == "__main__":
    main()