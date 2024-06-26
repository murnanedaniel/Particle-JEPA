import torch
from torch import nn
from torch.nn.parameter import Parameter
from .network_utils import AttentionBlock, make_mlp

class Transformer(nn.Module):
    def __init__(
            self, 
            d_model: int = 512,
            d_ff: int = 1024,
            heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0,
        ):
        super().__init__()
        
        self.input_encoder = make_mlp(
            d_input=2, 
            d_hidden=d_ff, 
            d_output=d_model,
            n_layer=2
        )
        
        self.encoder_layers = [
            AttentionBlock(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_ff = d_ff
            ) for _ in range(n_layers)
        ]
        
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        
        x = self.input_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask = ~mask)

        return x