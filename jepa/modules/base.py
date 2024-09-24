# 3rd party imports
from lightning.pytorch.core import LightningModule
import torch
from typing import Dict, Any, Optional
from abc import ABC
from abc import abstractmethod
from itertools import chain
from torch.optim.lr_scheduler import LambdaLR, StepLR, CyclicLR, SequentialLR

class BaseModule(ABC, LightningModule):
    def __init__(
            self, 
            batch_size: int,
            warmup: Optional[int] = 0,
            lr: Optional[float] = 1e-3,
            patience: Optional[int] = 10,
            factor: Optional[float] = 1,
            scheduler_type: Optional[str] = "step",
            dataset_args: Optional[Dict[str, Any]] = {},
        ):
        super().__init__()
        self.save_hyperparameters()
        """
        Initialise the Lightning Module
        """
    
    def _get_dataloader(self):        
        raise NotImplementedError("implement dataloader!")
    
    def train_dataloader(self):
        return self._get_dataloader()

    def val_dataloader(self):
        return self._get_dataloader()

    def test_dataloader(self):
        return self._get_dataloader()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
        )

        scheduler_type = self.hparams.get("scheduler_type", "step")
        warmup_epochs = self.hparams.get("warmup", 0)

        # Define the main scheduler based on scheduler_type
        scheduler_dict = {
            "step": lambda: StepLR(optimizer, step_size=self.hparams["patience"], gamma=self.hparams["factor"]),
            "cyclic": lambda: CyclicLR(optimizer, base_lr=self.hparams["lr"] * self.hparams["factor"], max_lr=self.hparams["lr"],
                                       step_size_up=self.hparams["patience"], mode="triangular2", cycle_momentum=False)
        }
        
        main_scheduler = scheduler_dict.get(scheduler_type)
        if not main_scheduler:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        main_scheduler = main_scheduler()

        # Create warmup scheduler if specified
        if warmup_epochs > 0:
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_epochs, 1.0))
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
            scheduler_name = "Warmup + Main Scheduler"
        else:
            scheduler = main_scheduler
            scheduler_name = f"{scheduler_type.capitalize()}LR"

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": scheduler_name
        }

        return [optimizer], [scheduler_config]
    
    @abstractmethod
    def embed(self, x, mask):
        raise NotImplementedError("implement embed method!")
    
    @abstractmethod   
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("implement training method!")
    
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