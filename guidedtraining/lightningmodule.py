from typing import Any
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils import predict, batch_loc_loss

from captum.attr import Saliency
from captum.attr import visualization as viz

class LightningNet(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = nn.BCELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)    
        
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        metric = BinaryAccuracy(threshold = 0.65).to('cuda')
        acc = metric(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        metric = BinaryAccuracy(threshold = 0.65).to('cuda')
        acc = metric(y_hat, y)
        self.log("val_loss", val_loss)
        self.log("val_acc", acc)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        metric = BinaryAccuracy(threshold = 0.65).to('cuda')
        acc = metric(y_hat, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.hparams.lr, momentum=0.9)
        return {"optimizer": optimizer}
    

class GuidedLightningNet(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = nn.BCELoss()
        self.save_hyperparameters()

    def model_wrapper(self, inputs, targets):
        output = self.model(inputs)
        # element-wise multiply outputs with one-hot encoded targets 
        # and compute sum of each row
        # This sums the prediction for all markers which exist in the cell
        return torch.sum(output * targets, dim=0)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        classloss = self.loss(y_hat, y)
        saliency = Saliency(self.model_wrapper)
        attr = saliency.attribute(x, additional_forward_args=y)
        attr = attr.cpu().detach().numpy()
        locloss = batch_loc_loss(attr, batch[2])
        aveloss = (classloss + locloss) / 2
        self.log("train_class_loss", classloss)
        self.log("train_loc_loss", locloss)
        self.log("train_loss", aveloss)
        return aveloss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        classloss = self.loss(y_hat, y)
        saliency = Saliency(self.model_wrapper)
        attr = saliency.attribute(x, additional_forward_args=y)
        attr = attr.cpu().detach().numpy()
        locloss = batch_loc_loss(attr, batch[2])
        aveloss = (classloss + locloss) / 2
        metric = BinaryAccuracy(threshold = 0.65).to('cuda')
        acc = metric(y_hat, y)
        self.log("val_class_loss", classloss)
        self.log("val_loc_loss", locloss)
        self.log("val_loss", aveloss)
        self.log("val_acc", acc)
        return {'val_loss': aveloss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        classloss = self.loss(y_hat, y)
        saliency = Saliency(self.model_wrapper)
        attr = saliency.attribute(x, additional_forward_args=y)
        attr = attr.cpu().detach().numpy()
        locloss = batch_loc_loss(attr, batch[2])
        aveloss = (classloss + locloss) / 2
        metric = BinaryAccuracy(threshold = 0.65).to('cuda')
        acc = metric(y_hat, y)
        self.log("test_class_loss", classloss)
        self.log("test_loc_loss", locloss)
        self.log("test_loss", aveloss)
        self.log("test_acc", acc)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.hparams.lr, momentum=0.9)
        return {"optimizer": optimizer}