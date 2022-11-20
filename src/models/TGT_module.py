from typing import Any, List
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from dgl import DGLGraph
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.utils.loss_func import RateLoss
from src.models.components.utils.metrics import SumRate
from src.models.components.TGT import TGT
import numpy as np

class TGTLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Any,#torch.optim.Optimizer,
        scheduler: Any,#torch.optim.lr_scheduler,
        noise
    ):
        super().__init__()
        self.net = net
        self.loss_func = RateLoss(noise)

        self.train_rate = SumRate()
        self.val_rate = SumRate()
        self.test_rate = SumRate()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_rate_best = MaxMetric()
        self.noise = noise
        self.save_hyperparameters(logger=True, ignore=["net"]) # Saves input arguments to self.hparams for quick storage good for quick iterations, but hard to understand code
    def forward(self, g: DGLGraph):
        return self.net(g)
    
    def on_train_start(self):
        pass

    def step(self, batch: Any):
        g = batch
        allocs = self.net(g)
        loss = self.loss_func(g, allocs)
        return loss, allocs
    
    def training_step(self, batch: Any, batch_idx: int):
        g = batch
        
        loss, allocs = self.step(batch) 
        self.train_loss(loss)
        self.train_rate(g, allocs, self.noise)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rate", self.train_rate, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "allocs": allocs}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, allocs= self.step(batch)

        self.val_loss(loss)
        g = batch
        self.val_rate(g, allocs, self.noise)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rate", self.val_rate, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "allocs": allocs}
    
    def validation_epoch_end(self, outputs: List[Any]):
        rate = self.val_rate.compute()
        self.val_rate_best(rate)
        self.log("val/rate_best", self.val_rate_best.compute(), prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, allocs= self.step(batch)

        self.test_loss(loss)
        g = batch
        self.test_rate(g, allocs, self.noise)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rate", self.test_rate, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "allocs": allocs}
    
    def test_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        g = batch
        allocs = self.net(g)
        return allocs


if __name__ == "__main__":
    from functools import partial
    from pytorch_lightning import Trainer
    from src.datamodules.d2d_datamodule import D2DDataModule
    model = TGT(64,32)
    datamodule = D2DDataModule(data_name='Node_',train_val_test_split=(25000, 2500, 200),n_list = [20],batch_size=64)
    optimizer = partial(torch.optim.AdamW, lr=5e-4)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, mode="min", patience=10, verbose=True)
    lit_module = TGTLitModule(model, optimizer, scheduler, 2.6e-5)
    trainer = Trainer(max_epochs=2,
            accelerator='gpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
    trainer.fit(lit_module, datamodule=datamodule)