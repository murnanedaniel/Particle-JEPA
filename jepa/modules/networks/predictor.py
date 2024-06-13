import torch
from torch import nn
from torch.nn.parameter import Parameter
from .network_utils import AttentionBlock, make_mlp

class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.,
        stop: float = 1.,
        num_gaussians: int = 50,
        periodic: bool = False,
    ):
        super().__init__()

        self.span = stop - start
        self.periodic = periodic
        self.register_buffer("offset", torch.linspace(start, stop, num_gaussians))
        self.register_buffer("inv_var", torch.tensor(1 / self.span**2))

    def forward(self, x: torch.Tensor):
        # compute rbfs and embeddings
        shape = x.shape
        x = x.view(-1, 1) - self.offset.view(1, -1)
        if self.periodic:
            x[x > self.span/2].abs_().neg_().add_(self.span)
        x = - 0.5 * x.square() * self.inv_var
        x = torch.exp(x)
        x = x.view(*shape, -1)
        return x

class Predictor(nn.Module):
    def __init__(
        self,
        num_gaussians: int = 50,
        rmin: float = 0.5,
        rmax: float = 3.,
        d_model: int = 512,
        d_ff: int = 1024,
        n_layer: int = 4,
        dropout: float = 0.,

    ):
        super().__init__()

        self.radial_smearing = GaussianSmearing(
            start=rmin, stop=rmax, num_gaussians=num_gaussians, periodic=False
        )

        self.angular_smearing = GaussianSmearing(
            start=-torch.pi, stop=torch.pi, num_gaussians=num_gaussians, periodic=True
        )

        self.predictor = make_mlp(
            d_input=4*num_gaussians + d_model,
            d_hidden=d_ff,
            d_output=d_model,
            n_layer=n_layer,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor, # [B, D] 
        rlim: torch.Tensor, # [2, B] 
        philim: torch.Tensor, # [2, B]
    ):
        return self.predictor(torch.cat([
            x,
            self.radial_smearing(rlim[0]),
            self.radial_smearing(rlim[1]),
            self.angular_smearing(philim[0]),
            self.angular_smearing(philim[1])
        ], dim=-1))