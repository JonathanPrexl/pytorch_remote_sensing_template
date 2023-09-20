import os
import hydra
import glob
import sys
from omegaconf import DictConfig, OmegaConf
from train import S2_MAE
from os.path import dirname as up
import json

@hydra.main(version_base = None, config_path="./configs/", config_name="s2_mae_channel_responseAware")
def main(config : DictConfig) -> None:

    # we first have to set visible
    # devices before importing any torch libs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)

    # check that the major configs are the same
    # for the current config file and the one
    # of the loaded checkpoint
    if config.resume: 
        raise_da_error = False
        checkpoint_config_loc = os.path.join(up(up(config.resumeCheckpoint)),"used_parameters.json")
        with open(checkpoint_config_loc, "r") as file:
            checkpoint_config = json.load(file)
        
        skipkeys = ["gpu_idx","seed","nEpochs",
                    "outputpath","experimentname",
                    "resume","resumeCheckpoint",
                    "special_save_nEpoch"]

        for key in config.keys():
            if not key in skipkeys:
                if not config[key] == checkpoint_config[key]:
                    raise_da_error = True
                    raise Warning(f"Configs of checkpoint does not match with current config for key {key}: \n \n {checkpoint_config[key]} \n \n {config[key]}")
        if raise_da_error:
            raise ValueError("configs dont match")

    # do fancy AI stuff
    trainer = hydra.utils.instantiate(config.trainroutine)
    trainer = trainer(config)

    # import torch
    # # torch._dynamo.config.verbose=True
    # compiled_fit = torch.compile(trainer.fit, mode="reduce-overhead")
    # compiled_fit()

    trainer.fit()
    trainer.finalize()

if __name__  == "__main__":
    main()