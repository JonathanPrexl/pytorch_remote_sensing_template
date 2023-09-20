import os
import glob
import numpy as np
import pandas as pd

from abc import abstractmethod

import torch
from torch.utils.data import Dataset

from utils import load_s2_image
from utils import preprocess_s2

class EuroSat_BaseDataloader(Dataset):

    def __init__(self,
                 topdir_dataset,
                 validation_ratio,
                 train_val_key):

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

        # collect all items for a class
        self.all_data = {"path":[],"label":[]}
        for label_idx, classname in enumerate(self.classnames):
            locs = glob.glob(os.path.join(topdir_dataset,classname,"*.tif"))
            self.all_data["path"] += locs
            self.all_data["label"] += [label_idx for _ in range(len(locs))]

        # shuffle data
        # (make sure pathes and labels are shuffeld with same permuation)
        # Be aware: Random split is not allways the best for EO datasets!
        # Regional split is usally better!
        random_idx = np.arange(len(self.all_data["path"]))
        np.random.shuffle(random_idx)
        self.all_data["path"] = np.array(self.all_data["path"])[random_idx].tolist()
        self.all_data["label"] = np.array(self.all_data["label"])[random_idx].tolist()

        # split according to validation_ratio
        total_lenght = len(self.all_data["path"])
        assert (validation_ratio > 0.05 ) & (validation_ratio < 0.4), \
            f"validation_ratio should be in [0.05, 0.4] (currently {validation_ratio})"
        
        split_idx = int(validation_ratio * total_lenght)

        if train_val_key == "train":
           self.all_data["path"] = self.all_data["path"][split_idx:]
           self.all_data["label"] = self.all_data["label"][split_idx:]            
        elif train_val_key == "val":
           self.all_data["path"] = self.all_data["path"][:split_idx]
           self.all_data["label"] = self.all_data["label"][:split_idx]
        else:
            raise ValueError(f"train_val_key must be train or val (currently {train_val_key})")


        df_locs = pd.DataFrame(self.all_data)
        print("++++++++++++++++++++++++")
        print(f"Number of sampels for {train_val_key}")
        for cidx,cn in enumerate(self.classnames):
            print(cn,"-->",len(df_locs["label"][df_locs["label"]==cidx]))

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.all_data["path"])
        

class EuroSat_AllBands(EuroSat_BaseDataloader):

    """ 
    returns emtpy tensor (there is no s1) S2 and Annotation Tensors in this order
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) # Frag Thomas!


    def __getitem__(self, i):

        s2_loc = self.all_data["path"][i]
        label = self.all_data["label"][i]

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


    dataloader = hydra.utils.instantiate(cfg.dataset,
                                         train_val_key="train")

    dataloader = hydra.utils.instantiate(cfg.dataset,
                                         train_val_key="val")

    print("data shape", dataloader.__getitem__(0)["s2"].shape)
