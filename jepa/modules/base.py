# 3rd party imports
from lightning.pytorch.core import LightningModule
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from jepa.utils import TracksDataset, collate_fn
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
            curriculum: Optional[int] = 0,
            min_scale: Optional[float] = 0.,
            dataset_args: Optional[Dict[str, Any]] = {},
        ):
        super().__init__()
        """
        Initialise the Lightning Module
        """
        self.save_hyperparameters()
    
    def _get_dataloader(self, is_training = False):
        dataset_args = self.hparams["dataset_args"].copy()
        if is_training and (self.trainer.current_epoch < self.hparams.get("curriculum", 0)):
            for name in ["minbias_lambda", "pileup_lambda", "hard_proc_lambda"]:
                if name in dataset_args:
                    dataset_args[name] *= (
                        (1 - self.hparams.get("min_scale", 0)) * self.trainer.current_epoch / self.hparams.get("curriculum", 0)
                        + self.hparams.get("min_scale", 0)
                    )
            
        dataset = TracksDataset(
            **dataset_args
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            collate_fn=collate_fn,
            num_workers=32,
            persistent_workers=True
        )
    
    def train_dataloader(self):
        return self._get_dataloader(is_training = True)

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
    def predict(self, x, mask):
        raise NotImplementedError("implement anomaly detection method!")
    
    def training_step(self, batch, batch_idx):
        """
        Particle-JEPA training procedure:
        1. Sample the batch for a context set of points
        2. Sample the batch for a target set of points
        3. Encode the context set of points with the ContextEncoder
        4. Encode the target set of points with the TargetEncoder
        5. Provide the context encoding and the TaskCondition to the Predictor
        6. Apply loss function to the prediction and the target encoding
        """
        context_x, context_mask = self.sample_context(batch)
        target_x, target_mask = self.sample_target(batch, context_mask)
        context_encoding = self.context_encoder(context_x, context_mask)
        target_encoding = self.target_encoder(target_x, target_mask)
        prediction = self.predictor(context_encoding, self.task_condition)
        loss = self.loss(prediction, target_encoding)
        self.log("training_loss", loss, on_step=True)
        return loss
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        """
        Particle-JEPA evaluation procedure is essentially the same
        as the training procedure
        """
        context_x, context_mask = self.sample_context(batch)
        target_x, target_mask = self.sample_target(batch, context_mask)
        context_encoding = self.context_encoder(context_x, context_mask)
        target_encoding = self.target_encoder(target_x, target_mask)
        prediction = self.predictor(context_encoding, self.task_condition)
        loss = self.loss(prediction, target_encoding)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

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