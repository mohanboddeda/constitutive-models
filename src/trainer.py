import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanAbsoluteError, R2Score


class SequenceModule(pl.LightningModule):
    def __init__(self, model, lr=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_mae', self.train_mae(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        self.log('train_r2', self.train_r2(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mae', self.val_mae(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        self.log('val_r2', self.val_r2(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "epoch"}
    
class PhysicsModule(pl.LightningModule):
    def __init__(self, model, lr=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        self.train_physics = physicsLoss()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_physics = physicsLoss()
        self.test_physics = physicsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        physics_loss = self.train_physics(x, pred)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        loss = loss + physics_loss
        self.log('train_both_losses', loss, prog_bar=True, on_epoch=True) 
        #self.log('train_mae', self.train_mae(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        #self.log('train_r2', self.train_r2(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        physicsLoss = self.val_physics(x, pred)
        self.log('val_physics_loss', physicsLoss, prog_bar=True, on_epoch=True)
        self.log('val_mae', self.val_mae(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)
        self.log('val_r2', self.val_r2(pred.view(-1), y.view(-1)), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        physicsLoss = self.test_physics(x, pred)
        self.log('test_loss', loss)
        self.log('test_physics_loss', physicsLoss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "epoch"}
    

class physicsLoss(nn.Module):
    def __init__(self):
        super(physicsLoss, self).__init__()
        self.nu_0 = 5.28e-5
        self.nu_inf = 3.30e-6
        self.lambda_val = 1.902
        self.n = 0.22
        self.a = 1.25

    def forward(self, x, preds):
        # Compute invariants
        D = 0.5 * (x + x.mT) 
        D_squared = D @ D
        diagonals = torch.diagonal(D_squared, dim1=-2, dim2=-1)
        second_invariant_D = diagonals.sum(-1)
        epsilon = 1e-12
        shear_rate = torch.sqrt(2 * second_invariant_D + epsilon)
        
        # Compute the physics loss
        term1 = (self.lambda_val * shear_rate)**self.a
        term2 = (1 + term1)**((self.n - 1) / self.a)
        nu = self.nu_inf + (self.nu_0 - self.nu_inf) * term2
        
        return torch.mean((preds - nu)**2)
    