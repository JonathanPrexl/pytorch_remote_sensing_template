import os
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
import hydra
import matplotlib.pyplot as plt

# importing learning stuff
import torch
from torchmetrics import MetricTracker, MetricCollection
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from utils import plot_confusion_matrix

# =============================== Abstract Baseclass of Trainer ==================================

class BaseTrainer(ABC):

    """ ABC --> its not allowed to instantiate this class directly"""
    
    def __init__(self,config):
        
        # some variables we need
        self.config = config
        self.globalstep = 0
        self.loss = 0

        # cpu or gpu
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
        output_config_json_path = os.path.join(self.savepath, "used_parameters.json")
        with open(output_config_json_path, 'w') as f:
            json.dump(OmegaConf.to_container(config), f) 

        # sice we have constant imput size in each
        # iteration we can set up benchmark mode
        # (its just a bit faster thats it...)
        torch.backends.cudnn.benchmark = True

        # loading datasets and loader
        self.train_data_set = hydra.utils.instantiate(config.dataset,
                                                 train_val_key="train")
        
        self.val_data_set = hydra.utils.instantiate(config.dataset,
                                               train_val_key="val")

        self.training_data_loader = hydra.utils.instantiate(config.dataloader,
                                                            dataset=self.train_data_set)
        
        self.val_data_loader = hydra.utils.instantiate(config.dataloader,
                                                       dataset=self.val_data_set)

        # loss function
        self.lossfunction: torch.nn.Module = hydra.utils.instantiate(config.lossfunction)

        # load the model
        self.model: torch.nn.Module = hydra.utils.instantiate(config.model)
        
        summary(self.model)

        self.optimizer = hydra.utils.instantiate(
            config.optimizer,
            params=self.model.parameters(),
        )

        # if you have a gpu we
        # shift all on the GPU
        if self.cuda:
            self.model = self.model.cuda()
            self.lossfunction = self.lossfunction.cuda()

        # setup tensorboard
        self.TB_writer = SummaryWriter(log_dir=os.path.join(self.savepath,"logs"))
        self.TB_writer.add_text("Parameters",str(config))

        # save the config if we need it somehwere else as well
        self.config = config
        
        # when to validate:
        # to make it comparable we dont do it after x batches rather after x sampels
        self.nextValidationstep = self.config.validation_every_N_samples
        
        list_of_metrics = [
                        Accuracy(task="multiclass", num_classes=config.model.num_classes,average="micro"),
                        F1Score(task="multiclass", num_classes=config.model.num_classes,average="macro"),
                        ConfusionMatrix(task="multiclass",num_classes=config.model.num_classes)
                        ]
        
        maximize_list=[True,True,True]

        if self.cuda:
            metric_coll = MetricCollection([m.cuda() for m in list_of_metrics])
        else:
            metric_coll = MetricCollection(list_of_metrics)

        self.tracker = MetricTracker(metric_coll, maximize=maximize_list)

    def fit(self):
        
        self.current_epoch = 1

        self.model.train()

        for epoch in range(self.current_epoch, self.config.nEpochs + 1):

            self._train_one_epoch()

            if self.current_epoch in self.config.special_save_nEpoch:
                self._save_checkpoint()

            self.TB_writer.add_scalar(f"lr", self.optimizer.param_groups[0]["lr"], global_step=self.globalstep)
            self.TB_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            
            self.current_epoch += 1

        self.TB_writer.flush()

        return None
   
    def _train_one_epoch(self):
         
        pbar_train = tqdm(total=len(self.training_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
        
        for batch_idx, batch in enumerate(self.training_data_loader, start=1): 
            
            self._train_one_batch(batch)
            
            if not self.config.validation_every_N_samples == -1:
                if (self.globalstep * self.config.dataloader.batch_size) >= self.nextValidationstep:
                    self.model.eval()
                    self._validate()                
                    self.nextValidationstep += self.config.validation_every_N_samples # set when to validate next time
                    self.model.train() # set back to train mode

            pbar_train.update()
        pbar_train.close()        
        
        return None

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
    
    def _save_checkpoint(self):
        
        outputloc =  os.path.join(self.checkpoint_dir,f"checkpoint_{self.current_epoch}_{self.globalstep}.pt")

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

        self.TB_writer.close()
        
        self._save_checkpoint()
        
        return None


# Custom Experiments

class EuroSat(BaseTrainer):

    def _train_one_batch(self, batch):
        
        s2 = batch["s2"]
        label = batch["label"]
        
        if self.cuda:
            s2 = s2.cuda()
            label = label.cuda()
        
        prediction = self.model(s2)

        self.loss = self.lossfunction(prediction, label.squeeze())

        self.optimizer.zero_grad()

        self.loss.backward()

        self.optimizer.step()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        
        return None

    def _validate(self):

        # start evaluation increment
        self.tracker.increment()
        current_eval_cycle = self.tracker.n_steps - 1 # starts with one so set it to zero
            
        with torch.no_grad():
            
            pbar_val = tqdm(total=len(self.val_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
            pbar_val.set_description("validation")

            for batch_idx, batch in enumerate(self.val_data_loader):
                s2 = batch["s2"]
                label = batch["label"]
                if self.cuda:
                    s2 = s2.cuda()
                    label = label.cuda()
                prediction = self.model(s2)
                self.tracker.update(prediction, label.squeeze())
                pbar_val.update()
                
            pbar_val.close()   


            # write to tensorboard
            # here use current_eval_cycle instead of
            # tracker.n_steps to index the array
            for key, val in self.tracker.compute_all().items():
                if not key == "MulticlassConfusionMatrix":
                    self.TB_writer.add_scalar(f"val/{key}", val[current_eval_cycle], global_step=self.globalstep)
                else:
                    # plot conf matrix
                    cm = self.tracker.compute()["MulticlassConfusionMatrix"].cpu().numpy()
                    figure = plot_confusion_matrix(cm, class_names=self.train_data_set.classnames)
                    self.TB_writer.add_figure(f"val/{key}", figure, global_step=self.globalstep)
                    plt.close()

        return None
    

class EuroSat_ViT(BaseTrainer):

    def _train_one_batch(self, batch):
        
        s2 = batch["s2"]
        label = batch["label"]
        
        if self.cuda:
            s2 = s2.cuda()
            label = label.cuda()
        
        prediction, _ = self.model(s2)

        self.loss = self.lossfunction(prediction, label.squeeze())

        self.optimizer.zero_grad()

        self.loss.backward()

        self.optimizer.step()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        
        return None

    def _validate(self):

        # start evaluation increment
        self.tracker.increment()
        current_eval_cycle = self.tracker.n_steps - 1 # starts with one so set it to zero
            
        with torch.no_grad():
            
            pbar_val = tqdm(total=len(self.val_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
            pbar_val.set_description("validation")

            for batch_idx, batch in enumerate(self.val_data_loader):
                s2 = batch["s2"]
                label = batch["label"]
                if self.cuda:
                    s2 = s2.cuda()
                    label = label.cuda()
                prediction, _ = self.model(s2)
                self.tracker.update(prediction, label.squeeze())
                pbar_val.update()
                
            pbar_val.close()   


            # write to tensorboard
            # here use current_eval_cycle instead of
            # tracker.n_steps to index the array
            for key, val in self.tracker.compute_all().items():
                if not key == "MulticlassConfusionMatrix":
                    self.TB_writer.add_scalar(f"val/{key}", val[current_eval_cycle], global_step=self.globalstep)
                else:
                    # plot conf matrix
                    cm = self.tracker.compute()["MulticlassConfusionMatrix"].cpu().numpy()
                    figure = plot_confusion_matrix(cm, class_names=self.train_data_set.classnames)
                    self.TB_writer.add_figure(f"val/{key}", figure, global_step=self.globalstep)
                    plt.close()

        return None
        