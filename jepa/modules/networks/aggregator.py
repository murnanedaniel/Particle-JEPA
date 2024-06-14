import torch
from torch import nn
from torch.nn.parameter import Parameter
from .network_utils import AttentionBlock, make_mlp

class Aggregator(nn.Module):
    def __init__(
            self, 
            d_model: int = 512,
            d_ff: int = 1024,
            heads: int = 8,
            n_layers: int = 2,
            dropout: float = 0,
        ):
        super().__init__()

        self.encoder_layers = [
            AttentionBlock(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_ff = d_ff,
                d_source = d_model,
                self_attn = False,
                cross_attn = True,
            ) for _ in range(n_layers)
        ]

        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.embeddings = Parameter(torch.randn((1, 1, d_model)))
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        z = self.embeddings.expand(-1, x.size(1), -1)
        for layer in self.encoder_layers:
            z = layer(z, src=x, src_padding_mask=~mask)
        return z[0]