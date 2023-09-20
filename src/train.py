import torch

# basic libarys
import os
import glob
import numpy as np
from datetime import datetime
import json
import csv
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
import hydra
import math

# geo stuff
import rasterio

# importing learning stuff
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchmetrics import MetricTracker, MetricCollection
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

# importing own modules
from dataloader import Sen12MS
# from losses import NT_Xent_SingGPU as NTXent
from model import MAE_ViT
from utils import s2toRGB

# =============================== Abstract Baseclass of Trainer ==================================

class BaseTrainer(ABC):
    
    def __init__(self,config):
        
        # some variables we need
        self.config = config
        self.globalstep = 0
        self.loss = 0

        if config.gpu_idx != "cpu":
            self.cuda = True
        else:
            self.cuda = False

        # seeding
        np.random.seed(config.seed)
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
        else:
            torch.manual_seed(config.seed)

        # make folders 
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.savepath = os.path.join(config.outputpath, config.experimentname, date_time)
        self.checkpoint_dir = os.path.join(self.savepath,"model_checkpoints")
        os.makedirs(self.savepath, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        # store the config file that has been used
        # to make it more traceable
        with open(os.path.join(self.savepath,"used_parameters.json"), 'w') as f:
            json.dump(OmegaConf.to_container(config), f) 

        # sice we have constant imput size in each
        # iteration we can set up benchmark mode
        # (its just a bit faster thats it...)
        torch.backends.cudnn.benchmark = True

        # loading datasets 
        dataset_class = hydra.utils.instantiate(config.dataloader)
        self.train_set = dataset_class(config, trainvaltestkey="train")
        self.val_set = dataset_class(config, trainvaltestkey="val", setOffsetToZero=True)


        # batchSampler = hydra.utils.instantiate(config.batchsampler)

        # self.batchSampler_train = batchSampler(neighbourSampels = self.train_set.neighbourSampels,
        #                                        lenDS = len(self.train_set),
        #                                        batch_size=config.dataloader.batch_size)
        
        # self.batchSampler_val = batchSampler(neighbourSampels = self.val_set.neighbourSampels,
        #                                      lenDS = len(self.val_set),
        #                                      batch_size=config.dataloader.batch_size)


        self.training_data_loader = DataLoader(dataset=self.train_set,
                                               num_workers=config.dataloader.threads,
                                               shuffle=True,
                                               pin_memory=True,
                                               batch_size=config.dataloader.batch_size)
                                               #batch_sampler=self.batchSampler_train)

        self.val_data_loader = DataLoader(dataset=self.val_set,
                                          num_workers=config.dataloader.threads,
                                          #shuffle=True,
                                          pin_memory=True,
                                          batch_size=config.dataloader.batch_size)
                                          #batch_sampler=self.batchSampler_val)

        self.model: torch.nn.Module = hydra.utils.instantiate(config.model)

        if config.compile:
            self.model = torch.compile(self.model)
        
        summary(self.model)

        # if you have a gpu we
        # shift all on the GPU
        if self.cuda:
            self.model = self.model.cuda()
            # self.lossfunction = self.lossfunction.cuda()


        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.optimizer.base_learning_rate * config.dataloader.batch_size / 256,
                                           betas=(0.9, 0.95),
                                           weight_decay=config.optimizer.weight_decay)
        
        lr_func = lambda epoch: min((epoch + 1) / (config.scheduler.warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / config.nEpochs * math.pi) + 1))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)
        
        # # if we start from scratch
        # if config.resume == False:

        # setup the model loss and optimizer
        # this is chaning for every experiment
        # self._setup_model_loss_optimizer(config)

        # # setup lr schedulert
        # if not self.config.lrscheduler._target_ == None: 
        #     self.scheduler = hydra.utils.instantiate(config.lrscheduler, optimizer=self.optimizer)
        # else:
        #     self.scheduler = None

        # elif config.resume == True:
            
        #     if not os.path.isfile(config.resumeCheckpoint):
        #         raise ValueError(f"checkpointfile {config.resumeCheckpoint} does not exist")

        #     # setup the model loss and optimizer
        #     # this is chaning for every experiment
        #     self._setup_model_loss_optimizer(config, loadCPT=True)

        #     # setup lr schedulert
        #     if not self.config.lrscheduler._target_ == None: 
        #         self.scheduler = hydra.utils.instantiate(config.lrscheduler, optimizer=self.optimizer)
        #     else:
        #         self.scheduler = None

        #     if not self.scheduler == None:
        #         raise ValueError("Checkpoint loading with lr scheduler not supported yet^^")


        # else:
        #     raise ValueError("")

        # gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # setup tensorboard
        self.TB_writer = SummaryWriter(log_dir=os.path.join(self.savepath,"logs"))
        self.TB_writer.add_text("Parameters",str(config))

        f, ax1 = plt.subplots(1,1,figsize=(8,4))
        x = np.arange(0,config.nEpochs+1)
        ax1.plot(x,[lr_func(xx)*config.optimizer.base_learning_rate for xx in x])
        
        self.TB_writer.add_figure(f"LR", f, global_step=0)
        plt.close()

        # save the config if we need it somehwere else as well
        self.config = config
        
        # when to validate:
        # to make it comparable we dont do it after x batches rather after x sampels
        self.nextValidationstep = self.config.validation_every_N_sampels
        self.nextPlottingstep = self.config.plotting_every_N_sampels

        self.best_metric_saveCrit = 10000
            

    # @torch.compile()
    def fit(self):
        
        self.current_epoch = 1

        if self.config.resume:
            self.current_epoch = self.init_epoch
            self.globalstep = self.init_global_step
            self.loss = self.init_loss
            if not self.config.nEpochs > self.current_epoch:
                raise ValueError(f"Model checkpoint has been trained for {self.current_epoch} and final number of epochs is {self.config.nEpochs}")
        
        self.model.train()

        for epoch in range(self.current_epoch, self.config.nEpochs + 1):

            self._train_one_epoch()

            if not self.config.validate_after_every_n_epoch == -1:
                if epoch % self.config.validate_after_every_n_epoch == 0:
                    self.model.eval() # todo: be carefull projection head
                    self._validate()
                    self.model.train()

            # if lr sceduler is active
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    # otherwise we only update if we have not hit T_max
                    # otherwise we leave constant... if we would not do that it would rise again
                    # -2 so we hit appox eta_min...
                    if self.current_epoch - 2 < self.config.lrscheduler.T_max:
                         self.scheduler.step()
                else:
                    self.scheduler.step()

            if self.current_epoch in self.config.special_save_nEpoch:
                self._save_checkpoint(nameoverwrite = f"special_save_E{self.current_epoch}")

            self.TB_writer.add_scalar(f"lr", self.optimizer.param_groups[0]["lr"], global_step=self.globalstep)
            self.TB_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            
            self.current_epoch += 1

        self.TB_writer.flush()

        return None
   
    def _train_one_epoch(self):
         

        pbar_train = tqdm(total=len(self.training_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
        
        for batch_idx, batch in enumerate(self.training_data_loader, start=1): 
            
            self._train_one_batch(batch)
            
            # if batch_idx == 1:
                
            #     # once a epoch show some example imgs in the
            #     # tensorboard

            #     for dashes in ["","d","dd"]:
            #         firstS1 = batch["s1"+dashes][:10]
            #         firstS2 = batch["s2"+dashes][:10]
            #         figure, loax = plt.subplots(2,10,figsize=(20,4))
            #         for i in range(10):
            #             loax[0,i].imshow( s2toRGB(firstS2[i]) ) 
            #             loax[1,i].imshow( firstS1[i,0] ) 
            #             loax[0,i].axis("off")
            #             loax[1,i].axis("off")
            #         self.TB_writer.add_figure(f"exampleInput/dashes_{dashes}", figure, global_step=self.globalstep)
            #         plt.close()

            # if batch_idx % 5 == 0:
            #     pbar_train.set_description("Epoch: {}".format(self.current_epoch))
            
            # if not self.config.validation_every_N_sampels == -1:
            #     if (self.globalstep * self.config.dataloader.batch_size) >= self.nextValidationstep:
                    
            #         self.model.eval()
            #         self._validate()                
            #         # set when to validate next time
            #         self.nextValidationstep += self.config.validation_every_N_sampels
            #         # set back to train mode
            #         self.model.train()
            
            if not self.config.plotting_every_N_sampels == -1:
                if (self.globalstep * self.config.dataloader.batch_size) >= self.nextPlottingstep:
                    print("Plotting Exampels",self.globalstep)
                    self.model.eval()
                    self._plot()                
                    # set when to validate next time
                    self.nextPlottingstep += self.config.plotting_every_N_sampels
                    # set back to train mode
                    self.model.train()

            pbar_train.update()
        pbar_train.close()        
        
        return None

    # def _setup_model_loss_optimizer(self, config, loadCPT=False):

    #     # loading model
    #     self.model: torch.nn.Module = hydra.utils.instantiate(config.model)

    #     if loadCPT:
    #         cpt = torch.load(config.resumeCheckpoint)
    #         self.model.load_state_dict(cpt["model_state_dict"])

    #     summary(self.model)

    #     # setup lossfunction
    #     #self.lossfunction = hydra.utils.instantiate(config.loss)

    #     # if you have a gpu we
    #     # shift all on the GPU
    #     if self.cuda:
    #         self.model = self.model.cuda()
    #         self.lossfunction = self.lossfunction.cuda()

    #     # set up the optimizer
    #     self.optimizer: torch.optim.optimizer.Optimizer = hydra.utils.instantiate(config.optimizer,
    #                                                                               params=self.model.parameters())
                                        
    #     if loadCPT:
    #         self.optimizer.load_state_dict(cpt["optimizer_state_dict"])

    #     # remember last epoch step and loss
    #     if loadCPT:
    #         self.init_epoch = cpt["epoch"]
    #         self.init_global_step = cpt["global_step"]
    #         self.init_loss = cpt["loss"]


    @abstractmethod
    def _train_one_batch(self, batch):

        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass

    @abstractmethod
    def _validate(self, batch):

        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass
        
    def _plot(self):

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(self.val_data_loader):
 
                s2 = batch["s2"][:1]
                if self.cuda:
                    s2 = s2.cuda()
                predicted_img, mask, lenFeat = self.model(s2)

                f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,3))


                # just firts img in batch
                s2 = s2[0]
                predicted_img = predicted_img[0]
                mask = mask[0]

                # turn everything into numpy
                s2 = s2.cpu().detach().numpy()
                predicted_img = predicted_img.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy().astype("uint8")

                reconstructed_img = predicted_img.copy()
                reconstructed_img[~mask.astype(bool)] = s2[~mask.astype(bool)]


                ax1.imshow( mask[0], vmin=0, vmax=1)
                ax2.imshow( s2toRGB(predicted_img) )
                ax3.imshow( s2toRGB(reconstructed_img) )
                ax4.imshow( s2toRGB(s2) )

                self.TB_writer.add_figure(f"example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()

                num_channels = predicted_img.shape[0]
                f, loax = plt.subplots(num_channels,4,figsize=(10,22))
                plt.subplots_adjust(hspace=.1,wspace=.1)
                for ijk in range(num_channels):

                    vmin = s2[ijk,:,:].min()
                    vmax = s2[ijk,:,:].max()

                    loax[ijk,0].imshow( mask[ijk,:,:], vmin=0, vmax=1)
                    loax[ijk,1].imshow( predicted_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,2].imshow( reconstructed_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,3].imshow( s2[ijk,:,:], vmin=vmin, vmax=vmax)
                    for ii in range(4):
                        loax[ijk,ii].axis("off")
                plt.tight_layout()
                self.TB_writer.add_figure(f"zz_channel_wise_example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()
                
                if batch_idx == 10: break




    def _save_checkpoint(self,nameoverwrite = ""):
        
        # only delete if nameoverwrite is "" so that
        # at last epoch we dont delte the inbetween checkpoint
        if nameoverwrite == "":
            all_models_there = glob.glob(os.path.join(self.checkpoint_dir,"checkpoint*.pt"))
            # there should be none or one model
            if not len(all_models_there) in [0,1]:
                raise ValueError(f"There is more then one model in the checkpoint dir ({len(self.checkpoint_dir)})... seems wrong")
            else:
                for model in all_models_there:
                    os.remove(model)
        
        if nameoverwrite == "":
            outputloc =  os.path.join(self.checkpoint_dir,f"checkpoint_{self.current_epoch}_{self.globalstep}_{self.best_metric_saveCrit}.pt")
        else:
            outputloc =  os.path.join(self.checkpoint_dir,f"{nameoverwrite}.pt")

        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.globalstep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            },
           outputloc)
          
        return None
              
    def finalize(self):

        # save hyperparameters
        self.TB_writer.add_hparams(
                    {"lr": self.config.optimizer.lr,
                     "bsize": self.config.dataloader.batch_size,
                    },
                    {
                    "hparam/IoU": self.best_metric_saveCrit,
                    },
                    run_name="hparams"
                )

        self.TB_writer.close()
        
        self._save_checkpoint("state_at_finalize")
        
        return None

class S2_MAE(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.mask_ratio = config.model.encoder.mask_ratio

    # @torch.compile()
    def _train_one_batch(self, batch):
        
        s2 = batch["s2"]
        
        if self.cuda:
            s2 = s2.cuda()

        with torch.cuda.amp.autocast():
            predicted_img, mask, lenFeat = self.model(s2)
            self.loss = torch.mean(torch.abs((predicted_img - s2)) * mask) #/ self.mask_ratio
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/lenFeat", lenFeat, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        # self.TB_writer.add_scalar("train/correct_order_inter", self.correct_order, global_step=self.globalstep)
        
        return None

    
    def _validate(self):
    
        pass

        return None

class S2_MAE_RFA(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.mask_ratio = config.model.encoder.mask_ratio

    # @torch.compile()
    def _train_one_batch(self, batch):
        
        s2 = batch["s2"]
        rf = batch["rf"]
        gsd = batch["gsd"]

        if self.cuda:
            s2 = s2.cuda()
            rf = rf.cuda()
            gsd = gsd.cuda()

        with torch.cuda.amp.autocast():
            predicted_img, mask, lenFeat = self.model(s2,rf,gsd)
            self.loss = torch.mean(torch.abs((predicted_img - s2)) * mask) #/ self.mask_ratio
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/lenFeat", lenFeat, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        # self.TB_writer.add_scalar("train/correct_order_inter", self.correct_order, global_step=self.globalstep)
        
        return None
    
    def _plot(self):

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(self.val_data_loader):
 
                s2 = batch["s2"]
                rf = batch["rf"]
                gsd = batch["gsd"]

                if self.cuda:
                    s2 = s2.cuda()
                    rf = rf.cuda()
                    gsd = gsd.cuda()

                predicted_img, mask, lenFeat = self.model(s2,rf,gsd)

                f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,3))


                # just firts img in batch
                s2 = s2[0]
                predicted_img = predicted_img[0]
                mask = mask[0]

                # turn everything into numpy
                s2 = s2.cpu().detach().numpy()
                predicted_img = predicted_img.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy().astype("uint8")

                reconstructed_img = predicted_img.copy()
                reconstructed_img[~mask.astype(bool)] = s2[~mask.astype(bool)]


                ax1.imshow( mask[0] , vmin=0, vmax=1)
                ax2.imshow( s2toRGB(predicted_img) )
                ax3.imshow( s2toRGB(reconstructed_img) )
                ax4.imshow( s2toRGB(s2) )

                self.TB_writer.add_figure(f"example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()

                num_channels = predicted_img.shape[0]
                f, loax = plt.subplots(num_channels,4,figsize=(10,22))
                plt.subplots_adjust(hspace=.1,wspace=.1)
                for ijk in range(num_channels):

                    vmin = s2[ijk,:,:].min()
                    vmax = s2[ijk,:,:].max()

                    loax[ijk,0].imshow( mask[ijk,:,:] , vmin=0, vmax=1)
                    loax[ijk,1].imshow( predicted_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,2].imshow( reconstructed_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,3].imshow( s2[ijk,:,:], vmin=vmin, vmax=vmax)
                    for ii in range(4):
                        loax[ijk,ii].axis("off")
                plt.tight_layout()
                self.TB_writer.add_figure(f"zz_channel_wise_example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()
                
                if batch_idx == 10: break

    
    def _validate(self):
    
        pass

        return None
    
class CIFAR_MAE(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.mask_ratio = config.model.encoder.mask_ratio
                       
    def _train_one_batch(self, batch):
        
        data = batch[0]
        
        if self.cuda:
            data = data.cuda()

        with torch.cuda.amp.autocast():
            predicted_img, mask, lenFeat = self.model(data)
            self.loss = torch.mean((predicted_img - data) ** 2 * mask) / self.mask_ratio
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/lenFeat", lenFeat, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        # self.TB_writer.add_scalar("train/correct_order_inter", self.correct_order, global_step=self.globalstep)
        
        return None
    
    def _plot(self):

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(self.val_data_loader):
 
                data = batch[0][:1]
                if self.cuda:
                    data = data.cuda()
                predicted_img, mask, lenFeat = self.model(data)

                f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,3))

                data = self.train_set.unnormalize( data[0] )
                data = np.moveaxis(data.cpu().detach().numpy(), 0, -1)
                data = np.clip(data,0,1)
                
                predicted_img = self.train_set.unnormalize( predicted_img[0] )
                predicted_img = np.moveaxis(predicted_img.cpu().detach().numpy(), 0, -1)
                predicted_img = np.clip(predicted_img,0,1)
                
                mask = mask[0]
                mask = mask.cpu().detach().numpy().astype("uint8")
                mask = np.moveaxis(mask,0,-1)

                reconstructed_img = predicted_img.copy()
                reconstructed_img[~mask.astype(bool)] = data[~mask.astype(bool)]
                assert np.all(reconstructed_img[~mask.astype(bool)] == data[~mask.astype(bool)])

                ax1.imshow( mask[:,:,0], vmin=0, vmax=1)
                ax2.imshow( predicted_img )
                ax3.imshow( reconstructed_img )
                ax4.imshow( data )

                self.TB_writer.add_figure(f"example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()

                if batch_idx == 10: break
    
    def _validate(self):

        pass

        return None
