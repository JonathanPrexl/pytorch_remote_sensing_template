import os
import glob
from abc import abstractmethod
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset

from utils import load_s2_image
from utils import preprocess_s2

class EuroSat_BaseDataloader(Dataset):

    def __init__(self,
                 topdir_dataset: str,
                 split_file: str,
                 train_val_key: str) -> None:

        assert os.path.isdir(topdir_dataset), ("'topdir_dataset' not a directory!", topdir_dataset)

        self.topdir_dataset = topdir_dataset

        # Get all the locations of the data and
        # the corresponding label into
        # a pandas dataframe
        self.classnames = ["AnnualCrop","HerbaceousVegetation","Industrial",
                           "PermanentCrop","River","Forest","Highway",
                           "Pasture","Residential","SeaLake"]
        
        # check that we specified the correct topdir_dataset
        assert set(os.listdir(self.topdir_dataset)) == set(self.classnames), \
            "Something wrong with the topdir_dataset. This folder should contain subfolder for all classes."

        # load the pre-determinded splits
        assert os.path.isfile(split_file), ("'split_file' not a file!", split_file)
        with open(split_file, 'r') as f:
            splits = json.load(f)
        

        # collect all items for a class
        print(f"Number of samples for {train_val_key}")
        self.locations = []
        for label_idx, classname in enumerate(self.classnames):
            current_samples = splits[train_val_key][classname]
            self.locations += [(loc, label_idx) for loc in current_samples]
            print(classname,"-->",len(current_samples))

        # shuffle the data
        np.random.shuffle(self.locations)

        return None

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.locations)
        
class EuroSat_AllBands(EuroSat_BaseDataloader):

    def __getitem__(self, i):

        assert isinstance(i, int), ("Index has to be int!", i, type(i))
        assert i >= 0, ("Index has to be positive!", i)
        s2_loc = self.locations[i][0]
        label = self.locations[i][1]

        # data
        data_s2 = load_s2_image(s2_loc).astype("float32")
        data_s2 = preprocess_s2(data_s2)        
        data_s2 = torch.Tensor(data_s2)

        # label
        label = torch.Tensor([label]).long()

        return {"s2":data_s2,"label":label}


if __name__ == "__main__":
    
    import hydra
    import omegaconf

    cfg = omegaconf.OmegaConf.load("./configs/default.yaml")

    dataset = hydra.utils.instantiate(cfg.dataset, train_val_key="train")
    dataset = hydra.utils.instantiate(cfg.dataset, train_val_key="val")

    print("data shape", dataset.__getitem__(0)["s2"].shape)
