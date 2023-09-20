import os
import glob
import numpy as np
import json
from skimage.transform import resize
import rasterio as rio
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import zoom
from rasterio.windows import Window

import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from enum import Enum, unique
from abc import ABC, abstractmethod
import hydra

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

from augmentations import TransformsPostivePair

from utils import load_s1_image, load_s2_image, load_lc
from utils import preprocess_s1, preprocess_s2, preprocess_lc

class BaseDataloader(Dataset):

    def __init__(self, config, trainvaltestkey):

        self.config = config
        self.topdir_dataset = config.dataloader.topdir_dataset
        self.trainvaltestkey = trainvaltestkey
        
        # load the file with the realive locations of the patches
        relative_locations_file = config.dataloader.relative_locations
        with open(relative_locations_file, "r") as f:
            self.relative_locations = json.load(f)[trainvaltestkey]

        # restrict the number of sampels if so specified
        # in the config file
        if trainvaltestkey == "train":
            if config.restrict_train_data != -1:
                self.relative_locations = self.relative_locations[:config.restrict_train_data]
        elif trainvaltestkey == "val":
            if config.restrict_val_data != -1:
                self.relative_locations = self.relative_locations[:config.restrict_val_data]
        else:
            raise ValueError("Incorect trainvalkey")


        print(f"In total {len(self.relative_locations)} for {trainvaltestkey}")

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.relative_locations)

class Sen12MS(BaseDataloader):

    """ 
    returns S1, S2 and Annotation Tensors in this order
    """

    def __init__(self, config, trainvaltestkey, setOffsetToZero=False, **kwargs):
        super().__init__(config, trainvaltestkey)   

        self.setOffsetToZero = setOffsetToZero
        self.resample = config.dataloader.resample_img
        if self.resample:
            self.outsize=config.dataloader.resample_img_size
            assert self.outsize==128, "for now hardcoded"

    def __getitem__(self, i):
    
        # s1_loc = os.path.join(self.topdir_dataset,
        #                       self.relative_locations[i]["s1"])

        s2_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["s2"])

        lc_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["landcover"])

        # # read s1 image from disk
        # data_s1 = load_s1_image(s1_loc).astype("float32")
        # data_s1 = preprocess_s1(data_s1)
        # data_s1 = torch.Tensor(data_s1.to_numpy())

        # read s2 image from disk
        # only take the 10 bands with 10 or 20m GSD
        bands_to_load = ["B02","B03","B04","B05","B06","B07","B08","B08A","B11","B12"]
        data_s2 = load_s2_image(s2_loc).loc[bands_to_load].astype("float32")
        data_s2 = preprocess_s2(data_s2)
        data_s2 = torch.Tensor(data_s2.to_numpy())

        # # read annotation from disk
        # data_lc = load_lc(lc_loc).astype("long")
        # data_lc = preprocess_lc(data_lc)
        # data_lc = torch.Tensor(data_lc)

        if self.resample:
            if self.setOffsetToZero:
                x0 = 0
                y0 = 0
            else:
                x0,y0 = np.random.randint(0,128,2)
            data_s2 = data_s2[:,x0:x0+128,y0:y0+128]


        # data_s2 = data_s2[:,:self.config.dataloader.random_crop_size,:self.config.dataloader.random_crop_size]

        # return {"s1":None,
        #         "s2":data_s2,
        #         "lc":None,
        #         "id":self.relative_locations[i]["s2"]}
        return {"s2":data_s2}
    

class Sen12MSRepsoneFunction(BaseDataloader):

    """ 
    returns S1, S2 and Annotation Tensors in this order
    """

    def __init__(self, config, trainvaltestkey, num_bands_output, **kwargs):
        super().__init__(config, trainvaltestkey)   
        
        self.num_bands_output = num_bands_output

        self.s2rf_10m = np.load("rs_s2_10m.npy")
        self.s2rf_20m = np.load("rs_s2_20m.npy")
        self.s2rf_30m = np.load("rs_s2_30m.npy")

        self.responsefunctiondict = {10:self.s2rf_10m,
                                     20:self.s2rf_20m,
                                     30:self.s2rf_30m}
        
        self.resample = config.dataloader.resample_img
        if not self.resample:
            assert self.outsize==128, "for now hardcoded"
            assert False, "hardcoded bellow"



    def preprocess_s2(self,s2):
        return s2/10000
    
    # def load_raster(self,path,upscale_factor):

    #     # load the raw data
    #     with rio.open(path,"r") as src:
    #         d = src.read(out_shape=(src.count,
    #                                 int(src.height * upscale_factor),
    #                                 int(src.width * upscale_factor)
    #                     ),
    #                     resampling=Resampling.cubic)

    #     if not d.shape[0] == 13:
    #         raise ValueError("Nope")

    #     bandnames = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
    #     xarr = xr.DataArray(d, dims=("channel","lat", "long"), coords={"channel": bandnames})

    #     return xarr

    #@profile
    def __getitem__(self, i):
    
        s2_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["s2"])

        with rio.open(s2_loc,"r") as src:
            x0,y0 = np.random.randint(0,128,2)
            data_s2_30m = src.read((2,3,4,5,6,7,8,9,12,13),
                                    window=Window.from_slices((x0, x0+128),
                                                              (y0, y0+128))
                                  ).astype("float32")



        data_s2_10m = data_s2_30m[[0,1,2,6]]
        data_s2_20m = data_s2_30m[[3,4,5,7,8,9]]

        # # TODO: blurring!! make a proper implementaion
        ds = zoom(data_s2_30m,(1,0.333,0.333), order=0)
        data_s2_30m[:] = zoom(ds,(1,3,3), order=0)[:,:-1,:-1]


        datadict = {10:data_s2_10m,
                    20:data_s2_20m,
                    30:data_s2_30m}

        o_data = []
        o_responsfunctions = []
        o_gsd = []

        for i in range(self.num_bands_output):

            
            # todo rethink values
            gsd = np.random.choice([10,20,30],p=[0.7,0.2,0.1])
            num_bands = np.random.choice([1,2,3],p=[0.15,0.7,0.15])

            bandindices = np.random.choice(datadict[gsd].shape[0],num_bands, replace=False)
            factors = np.random.uniform(0.1,0.9,num_bands) 
            inversemax = 1/factors.max()
            # factors /= factors.sum() # todo: is it good that it sums up to one?

            signal = datadict[gsd][bandindices]
            signal = signal*factors[:,None,None]
            # signal /= len(bandindices)
            signal = signal.sum(axis=0)
            signal *= inversemax
            
            rf = self.responsefunctiondict[gsd][bandindices]
            rf = rf*factors[:,None]
            rf = rf.sum(axis=0)
            # rf /= len(bandindices)
            rf *= inversemax

            o_gsd.append(gsd)
            o_data.append(signal)
            o_responsfunctions.append(rf)

        data_s2 = np.array(o_data)
        rf = np.array(o_responsfunctions).squeeze()
        gsd = np.array(o_gsd)

        data_s2 = self.preprocess_s2(data_s2)

        return {"s2":torch.Tensor(data_s2),"rf":torch.Tensor(rf),"gsd":torch.Tensor(gsd)}

# copy 
# class Sen12MSRepsoneFunction(BaseDataloader):

#     """ 
#     returns S1, S2 and Annotation Tensors in this order
#     """

#     def __init__(self, config, trainvaltestkey, num_bands_output, **kwargs):
#         super().__init__(config, trainvaltestkey)   
        
#         self.num_bands_output = num_bands_output

#         self.s2rf_10m = np.load("rs_s2_10m.npy")
#         self.s2rf_20m = np.load("rs_s2_20m.npy")
#         self.s2rf_30m = np.load("rs_s2_30m.npy")

#         self.responsefunctiondict = {10:self.s2rf_10m,
#                                      20:self.s2rf_20m,
#                                      30:self.s2rf_30m}
        
#         self.resample = config.dataloader.resample_img
#         if self.resample:
#             self.outsize=config.dataloader.resample_img_size
#             assert self.outsize==128, "for now hardcoded"

#     def preprocess_s2(self,s2):
#         return s2/10000
    
#     def load_raster(self,path,upscale_factor):

#         # load the raw data
#         with rio.open(path,"r") as src:
#             d = src.read(out_shape=(src.count,
#                                     int(src.height * upscale_factor),
#                                     int(src.width * upscale_factor)
#                         ),
#                         resampling=Resampling.cubic)

#         if not d.shape[0] == 13:
#             raise ValueError("Nope")

#         bandnames = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
#         xarr = xr.DataArray(d, dims=("channel","lat", "long"), coords={"channel": bandnames})

#         return xarr

#     #@profile
#     def __getitem__(self, i):
    
#         s2_loc = os.path.join(self.topdir_dataset,
#                               self.relative_locations[i]["s2"])

#         bands_to_load_10m = ["B02","B03","B04","B08"]
#         bands_to_load_20m = ["B05","B06","B07","B08A","B11","B12"]
#         bands_to_load_30m = ["B02","B03","B04","B05","B06","B07","B08","B08A","B11","B12"]

#         data_s2_10m = self.load_raster(s2_loc,1).loc[bands_to_load_10m].astype("float32")
#         data_s2_20m = self.load_raster(s2_loc,1).loc[bands_to_load_20m].astype("float32")
        
#         # TODO: blurring
#         data_s2_30m = self.load_raster(s2_loc,0.3).loc[bands_to_load_30m].astype("float32")
#         data_s2_30m = resize(data_s2_30m,(data_s2_30m.shape[0],
#                                           data_s2_10m.shape[1],
#                                           data_s2_10m.shape[2]))                         
        
#         datadict = {10:data_s2_10m,
#                     20:data_s2_20m,
#                     30:data_s2_30m}

#         o_data = []
#         o_responsfunctions = []
#         o_gsd = []

#         for i in range(self.num_bands_output):

            
#             # todo rethink values
#             gsd = np.random.choice([10,20,30],p=[0.7,0.2,0.1])
#             num_bands = np.random.choice([1,2,3],p=[0.15,0.7,0.15])

#             bandindices = np.random.choice(datadict[gsd].shape[0],num_bands, replace=False)
#             factors = np.random.uniform(0.1,0.9,num_bands) 
#             inversemax = 1/factors.max()
#             # factors /= factors.sum() # todo: is it good that it sums up to one?

#             signal = datadict[gsd][bandindices]
#             signal = signal*factors[:,None,None]
#             # signal /= len(bandindices)
#             signal = signal.sum(axis=0)
#             signal *= inversemax
            
#             rf = self.responsefunctiondict[gsd][bandindices]
#             rf = rf*factors[:,None]
#             rf = rf.sum(axis=0)
#             # rf /= len(bandindices)
#             rf *= inversemax

#             o_gsd.append(gsd)
#             o_data.append(signal)
#             o_responsfunctions.append(rf)

#         data_s2 = np.array(o_data)
#         rf = np.array(o_responsfunctions).squeeze()
#         gsd = np.array(o_gsd)

#         if self.resample:
#             x0,y0 = np.random.randint(0,128,2)
#             data_s2 = data_s2[:,x0:x0+128,y0:y0+128]

#         data_s2 = self.preprocess_s2(data_s2)

#         return {"s2":torch.Tensor(data_s2),"rf":torch.Tensor(rf),"gsd":torch.Tensor(gsd)}


class CIFAR100(DataLoader):

    """ 
    returns S1, S2 and Annotation Tensors in this order
    """

    def __init__(self, config, trainvaltestkey, **kwargs):

        # https://github.com/pytorch/vision/issues/528
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        assert config.dataloader.resample_img
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std),
             torchvision.transforms.Resize(config.dataloader.resample_img_size)])

        self.unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
        )

        if trainvaltestkey == "train":
            self.ds = torchvision.datasets.CIFAR100(root = config.dataloader.topdir_dataset,
                                                    transform=transform,
                                                    train = True,
                                                    download = False)
        elif trainvaltestkey == "val":
            self.ds = torchvision.datasets.CIFAR100(root = config.dataloader.topdir_dataset,
                                                    transform=transform,
                                                    train = False,
                                                    download = False)
        else:
            raise ValueError("key not known")
        
    def __getitem__(self, i):

        return self.ds[i]
    
    def __len__(self):
        return len(self.ds)


if __name__ == "__main__":

    # from omegaconf import OmegaConf
    # cfg = OmegaConf.load('./configs/cifar_mae.yaml')

    # dl = CIFAR100(cfg,"train")
    # batch = dl.__getitem__(0)

    from omegaconf import OmegaConf
    import hydra
    cfg = OmegaConf.load('./configs/s2_mae_channel_responseAware.yaml')
    dlc = hydra.utils.instantiate(cfg.dataloader)
    dl = dlc(cfg,"train")
    batch = dl.__getitem__(0)
    print(batch["s2"].shape)
    pass