from jepa.modules.models.jepa import JEPA
from jepa.utils.sampling_utils import WedgePatchify, QualityCut
from jepa.modules.networks.network_utils import make_mlp

from toytrack.dataloaders import TracksDataset
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np




class ClassifyPt(JEPA):
    """
    A downstream task that takes embeddings of wedges and predicts the transverse
    momentum of the highest energy particle as either "high" or "low" (above or below some
    threshold).
    """

    def __init__(
        self,
        d_input: int = 2,
        d_ff: int = 128,
        d_model: int = 32,
        d_embedding: int = 8,
        n_layers: int = 4,
        n_agg_layers: int = 2,
        heads: int = 4,
        batch_size: int = 128,
        warmup: Optional[int] = 0,
        lr: Optional[float] = 1e-3,
        patience: Optional[int] = 10,
        factor: Optional[float] = 1,
        random_context: Optional[bool] = True,
        dataset_args: Optional[Dict[str, Any]] = {},
        pt_threshold: Optional[float] = 5.0,
        min_hits: Optional[int] = 3,
        pretrained_path: Optional[str] = None,
        freeze_backbone: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            d_input=d_input,
            d_ff=d_ff,
            d_model=d_model,
            d_embedding=d_embedding,
            n_layers=n_layers,
            n_agg_layers=n_agg_layers,
            heads=heads,
            batch_size=batch_size,
            warmup=warmup,
            lr=lr,
            patience=patience,
            factor=factor,
            random_context=random_context,
            dataset_args=dataset_args,
        )

        # Freeze unused components from JEPA
        for param in self.predictor.parameters():
            param.requires_grad = False
        for param in self.ema_encoder.parameters():
            param.requires_grad = False

        # Load pretrained weights if provided
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # Load only the encoder part of the state dict
            encoder_state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                                if k.startswith('encoder')}
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                print("Encoder backbone frozen")

        self.prediction_head = make_mlp(
            d_input=d_embedding,
            d_hidden=d_ff,
            d_output=1,
            n_layer=3
        )

        # Initialize metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")

    def _get_dataloader(self) -> DataLoader:
        """
        Creates and returns a DataLoader for the TracksDataset with WedgePatchify transformation.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        patchify = WedgePatchify(
            phi_range=np.pi / 4, 
            radius_midpoint = (self.dataset_args["detector"]["max_radius"] + self.dataset_args["detector"]["min_radius"]) / 2
        )

        quality_cut = QualityCut(
            min_hits=self.hparams["min_hits"],
            pt_threshold=self.hparams["pt_threshold"]
        )
        self.dataset = TracksDataset(
            self.dataset_args,
            transforms=[patchify, quality_cut]
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams.get("num_workers", 32),
            collate_fn=self.dataset.collate_fn,
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step.

        Args:
            batch (dict): Batch of data containing tracklets and related information.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the training step.
        """
        # Get inputs and forward pass
        x, mask, context_mask, y = self._extract_batch_data(batch)
        embedded_context_tracklets = self._embed_context_tracklets(x, context_mask, batch)
        logits = self.prediction_head(embedded_context_tracklets).squeeze(-1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate and log metrics
        metrics = self._calculate_metrics(logits, y, stage="train")
        self.log_dict({
            "train_loss": loss,
            **metrics
        })
        self._log_learning_rate()

        return loss

    def shared_evaluation(self, batch, batch_idx):
        """
        Executes a single evaluation step.

        Args:
            batch (dict): Batch of data containing tracklets and related information.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing loss, efficiency, purity, and mean distances.
        """
        # Get inputs and forward pass
        x, mask, context_mask, y = self._extract_batch_data(batch)
        embedded_context_tracklets = self._embed_context_tracklets(x, context_mask, batch)
        logits = self.prediction_head(embedded_context_tracklets).squeeze(-1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate and log metrics
        metrics = self._calculate_metrics(logits, y, stage="val")
        self.log_dict({
            "val_loss": loss,
            **metrics
        })

        return {
            "loss": loss,
            **metrics,
            "logits": logits,
            "labels": y
        }

    def _extract_batch_data(self, batch):
        """
        Extracts necessary data from the batch.

        Args:
            batch (dict): Batch of data.

        Returns:
            Tuple containing x, mask, context_mask and pids
        """
        x, y, mask, context_mask = (
            batch["x"],
            batch["y"].float(),
            batch["mask"],
            batch["context_mask"],
        )

        # Check for any batch entries that have at least config["min_hits"] in the masked context wedge
        min_hits = self.hparams.get("min_hits", 1)
        valid_rows = context_mask.sum(dim=1) >= min_hits
        if not valid_rows.all():
            # Remove those rows
            x = x[valid_rows]
            y = y[valid_rows]
            context_mask = context_mask[valid_rows]
            mask = mask[valid_rows]

        return x, mask, context_mask, y

    def _calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor, stage: str = "train"):
        """
        Calculate classification metrics.
        
        Args:
            logits (torch.Tensor): Raw model outputs before sigmoid
            labels (torch.Tensor): Ground truth labels
            stage (str): Either "train" or "val"
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        with torch.no_grad():
            # Get probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Calculate metrics based on stage
            metrics = {}
            if stage == "train":
                accuracy = self.train_accuracy(preds, labels)
                metrics.update({
                    "train_accuracy": accuracy,
                })
            else:
                accuracy = self.val_accuracy(preds, labels)
                auroc = self.val_auroc(probs, labels.long())
                metrics.update({
                    "val_accuracy": accuracy,
                    "val_auroc": auroc,
                })
            
        return metrics

    def on_train_epoch_end(self):
        """Reset metrics at the end of each training epoch"""
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        """Reset metrics at the end of each validation epoch"""
        self.val_accuracy.reset()
        self.val_auroc.reset()

    @torch.no_grad
    def update_ema_params(self):
        pass

    # ------------------- DEBUGGING + VISUALISATION ------------------- #

    def _log_learning_rate(self):
        """Logs the current learning rate."""
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", current_lr)
        if self.global_step == 0:
            print(f"Current learning rate: {current_lr}")
