import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MeanAbsoluteError, R2Score
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.models import SimpleTransformer, MLP
from src.trainer import SequenceModule, PhysicsModule
from src.data.prepro import normalize_data, TensorSequenceDataset

import onnx
 
@hydra.main(config_path="config", config_name="main_conf", version_base=None)
def main(cfg: DictConfig) -> None:
    X = torch.load(f"datafiles/X_{cfg.data.dim}D_{cfg.data.constitutive_eq}.pt", weights_only=True)
    Y = torch.load(f"datafiles/Y_{cfg.data.dim}D_{cfg.data.constitutive_eq}.pt", weights_only=True)  

    #X, Y, stats_X, stats_Y = normalize_data(X, Y)
    #if not os.path.exists(f'data/stats_X_{cfg.data.constitutive_eq}.pt'):
    #    torch.save(stats_X, f'data/stats_X_{cfg.data.constitutive_eq}.pt')
    #    torch.save(stats_Y, f'data/stats_Y_{cfg.data.constitutive_eq}.pt')
    
    dataset = TensorSequenceDataset(X, Y)
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=1)

    #model initialisieren
    model = instantiate(cfg.model)
    module = instantiate(cfg.trainer)

    
    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=cfg.patience, mode='min')

    logger = TensorBoardLogger("lightning_logs", name="pinn_carreau")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, 
        callbacks=[checkpoint_cb, earlystop_cb],
        accelerator=cfg.device if torch.cuda.is_available() else 'cpu', 
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        #ckpt_path="lightning_logs/version_28/checkpoints/epoch=350-step=7020.ckpt",
        devices=1,
        logger=logger) # <<< ADD THIS
    
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader)

    if cfg.export_onnx:
        dummy_input = torch.randn(cfg.batch_size, X.shape[1], X.shape[2])
        onnx_path = f"model_{cfg.data.constitutive_eq}.onnx"
        torch.onnx.export(module.model, dummy_input, onnx_path, export_params=True, opset_version=11, do_constant_folding=True)
        print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    main()
