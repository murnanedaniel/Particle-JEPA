import torch
from typing import Optional, Dict, Any
from jepa.modules.base import BaseModule
from jepa.modules.networks.transformer import Transformer

class ContrastiveLearningModule(BaseModule):
    def __init__(
            self, 
            model: str,
            d_model: int = 512,
            d_ff: int = 1024,
            heads: int = 8,
            n_layers: int = 6,
            n_pool_layer: int = 2,
            dropout: float = 0,
            batch_size: int = 128,
            warmup: Optional[int] = 0,
            lr: Optional[float] = 1e-3,
            patience: Optional[int] = 10,
            factor: Optional[float] = 1,
            curriculum: Optional[str] = "1",
            t0: Optional[int] = 0,
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3,
            num_sectors: Optional[int] = 1,
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
            curriculum=curriculum,
            t0=t0,
            dataset_args=dataset_args
        )
        
        self.encoder = Transformer(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
            n_pool_layer=n_pool_layer,
            dropout=dropout
        )

        self.ema_encoder = Transformer(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
            n_pool_layer=n_pool_layer,
            dropout=dropout
        )

        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        
    def sample_context(self, x: torch.Tensor, mask: torch.Tensor, num_sectors: int):

        batch_size = x.size(0)
        r, phi = torch.linalg.norm(x, dim=-1), torch.atan2(x[..., 1], x[..., 0])

        rlim = (self.hparams["max_radius"] - self.hparams["min_radius"]) * torch.rand((2, batch_size, num_sectors)) + self.hparams["min_radius"]
        rlim[:, rlim[0] > rlim[1]] = rlim[:, rlim[0] > rlim[1]].flip(0)
        rlim = rlim[:, :, None, :]

        philim = 2 * torch.rand((2, batch_size, num_sectors)) * torch.pi - torch.pi
        philim = philim[:, :, None, :]
        phiorder = philim[0] > philim[1]

        context_mask = (r < rlim[1]) & (r > rlim[0]) & (
            (phiorder & (phi > philim[0]) & (phi < philim[1]))
            | ((~phiorder) & (phi < philim[0]) & (phi > philim[1]))
        )
        context_mask = (~ context_mask.any(3)) & mask

        return context_mask, rlim, philim
    
    @torch.no_grad
    def sample_target(self, x: torch.Tensor, mask: torch.Tensor, context_mask: torch.Tensor):
        target = self.ema_encoder(x, mask)
        target.masked_fill_(~mask | context_mask, 0)
        return target

    def embed(self, x, mask):
        raise NotImplementedError("implement anomaly detection method!")
    
    def training_step(self, batch, batch_idx):
        x, mask = batch.x, batch.mask
        context_mask, rlim, philim = self.sample_context(x, mask, self.hparams["num_sectors"])
        target = self.sample_target(x, mask, context_mask)
        context = x.masked_fill(~context_mask, 0)
        prediction = self.predictor(context, context_mask, rlim, philim)

        loss = ((target - prediction)[mask & ~context_mask]).square().mean()

        self.update_ema_variables()

        return loss

    @torch.no_grad
    def update_ema_variables(self):
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data.copy_(ema_param.data * self.hparams["ema_decay"] + (1 - self.hparams["ema_decay"]) * param.data)
        
