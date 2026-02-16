import tomllib
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, EarlyStopping
import wandb

from .dataset import BeatDataset
from .model import BeatModel

import os
os.environ["WANDB_MODE"] = "disabled"

class YourModelWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc = (logits == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc = (logits == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
    
def train(csv_path: str, config_path: str):
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    params = config['training']
    print(config)

    model_name = config["model_name"]
    wandb_logger = WandbLogger(project="alpacapella", name=model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_progress_bar=True
    )


    model = BeatModel(config)
    your_model = YourModelWrapper(model, learning_rate=params['lr'])
    train_set = BeatDataset(csv_path, config, split='train')
    val_set = BeatDataset(csv_path, config, split='val')
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

    trainer.fit(your_model, train_loader, val_loader)

    trainer.save_checkpoint("final_model.ckpt")
    wandb.finish()