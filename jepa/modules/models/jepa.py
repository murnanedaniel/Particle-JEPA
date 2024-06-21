import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from jepa.modules.base import BaseModule
from jepa.modules.networks.transformer import Transformer
from jepa.modules.networks.aggregator import Aggregator
from jepa.modules.networks.predictor import SplitTrackPredictor
from jepa.utils import random_rphi_sample, track_split_sample

class JEPAEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 1024,
        heads: int = 8,
        n_layers: int = 6,
        n_agg_layers: int = 2,
        dropout: float = 0,
    ):
        super().__init__()

        self.encoder = Transformer(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
            dropout=dropout
        )

        self.aggregator = Aggregator(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_agg_layers,
            dropout=dropout
        )
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        agg_mask: Optional[torch.Tensor] = None
    ):
        if agg_mask is None:
            agg_mask = mask
        else:
            agg_mask &= mask
        x = self.encoder(x, mask)
        x = self.aggregator(x, agg_mask)
        return x

class JEPA(BaseModule):
    def __init__(
            self, 
            model: str,
            d_model: int = 512,
            d_ff: int = 1024,
            heads: int = 8,
            n_layers: int = 6,
            n_pool_layer: int = 2,
            n_predictor_layers: int = 4,
            dropout: float = 0,
            batch_size: int = 128,
            warmup: Optional[int] = 0,
            lr: Optional[float] = 1e-3,
            patience: Optional[int] = 10,
            factor: Optional[float] = 1,
            num_gaussians: Optional[int] = 50,
            dataset_args: Optional[Dict[str, Any]] = {},
            ema_decay: Optional[float] = 0.99,
            *args,
            **kwargs,
        ):
        super().__init__(
            batch_size=batch_size,
            warmup=warmup,
            lr=lr,
            patience=patience,
            factor=factor,
            dataset_args=dataset_args
        )
        
        self.encoder = JEPAEncoder(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
            n_agg_layers=n_pool_layer,
            dropout=dropout
        )

        self.ema_encoder = JEPAEncoder(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
            n_agg_layers=n_pool_layer,
            dropout=dropout
        )

        self.max_radius = dataset_args.get("max_radius", 3.0)
        self.min_radius = dataset_args.get("min_radius", 0.5)

        self.predictor = SplitTrackPredictor(
            d_model=d_model,
            d_ff=d_ff,
            n_layer=n_predictor_layers,
            dropout=dropout,
        )

        self.ema_decay = ema_decay
        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        
    def sample_context(self, x: torch.Tensor, mask: torch.Tensor):

        r, phi = torch.linalg.norm(x, dim=-1), torch.atan2(x[..., 1], x[..., 0])

        # target_mask, rlim, philim = random_rphi_sample(r, phi, self.min_radius, self.max_radius)
        target_mask, inner_target_flag = track_split_sample(r, self.min_radius, self.max_radius)

        context_mask = ~target_mask & mask
        target_mask = target_mask & mask

        return context_mask, target_mask, inner_target_flag #, rlim.squeeze(-1), philim.squeeze(-1)
    
    @torch.no_grad
    def embed_target(self, x: torch.Tensor, mask: torch.Tensor, target_mask: torch.Tensor):
        target = self.ema_encoder(x, mask, agg_mask=target_mask)
        return target

    def embed(self, x, mask):
        raise NotImplementedError("implement anomaly detection method!")
    
    def training_step(self, batch, batch_idx):
        x, mask, *_ = batch
        context_mask, target_mask, inner_target_flag = self.sample_context(x, mask)
        x = x.permute(1, 0, 2)
        target = self.embed_target(x, mask, target_mask)
        context = self.encoder(x, context_mask, context_mask)
        prediction = self.predictor(context, inner_target_flag)

        loss = F.mse_loss(prediction, target)

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx):
        x, mask, *_ = batch
        context_mask, target_mask, inner_target_flag = self.sample_context(x, mask)
        x = x.permute(1, 0, 2)
        target = self.embed_target(x, mask, target_mask)
        context = self.encoder(x, context_mask, context_mask)
        prediction = self.predictor(context, inner_target_flag)
        
        loss = F.mse_loss(prediction, target)
        source_target_difference = F.mse_loss(prediction, context)

        try:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict({
                "val_loss": loss,
                "source_target_difference": source_target_difference,
                "lr": lr
            })
        except:
            pass

        return {
            "loss": loss,
            "prediction": prediction,
            "target": target,
            "context": context
        }
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.update_ema_params()

    @torch.no_grad
    def update_ema_params(self):
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data.copy_(ema_param.data * self.ema_decay + (1 - self.ema_decay) * param.data)
        