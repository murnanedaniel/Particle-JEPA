# 3rd party imports
from lightning.pytorch.core import LightningModule
import torch
from torch.utils.data import DataLoader
from jepa.utils import TracksDatasetFixed, collate_fn_fixed
from typing import Dict, Any, Optional
from abc import ABC
from abc import abstractmethod

class BaseModule(ABC, LightningModule):
    def __init__(
            self, 
            batch_size: int,
            warmup: Optional[int] = 0,
            lr: Optional[float] = 1e-3,
            patience: Optional[int] = 10,
            factor: Optional[float] = 1,
            dataset_args: Optional[Dict[str, Any]] = {},
        ):
        super().__init__()
        """
        Initialise the Lightning Module
        """
    
    def _get_dataloader(self):        
        return DataLoader(
            TracksDatasetFixed(**self.hparams["dataset_args"]),
            batch_size=self.hparams["batch_size"],
            collate_fn=collate_fn_fixed,
            num_workers=self.hparams["workers"],
            persistent_workers=True
        )
    
    def train_dataloader(self):
        return self._get_dataloader()

    def val_dataloader(self):
        return self._get_dataloader()

    def test_dataloader(self):
        return self._get_dataloader()
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"]
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    
    @abstractmethod
    def embed(self, x, mask):
        raise NotImplementedError("implement embed mothod!")
    
    @abstractmethod   
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("implement training mothod!")
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        """
        implement the evaluation of pre-training models using the 
        embed method that all models should implement.
        Could be some regression task or even just some visualization
        of how different events are seperated
        """
        raise NotImplementedError("implement evaluation!")

    def validation_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become 
        built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()