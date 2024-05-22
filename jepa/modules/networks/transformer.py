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
            n_pool_layer: int = 2,
            dropout: float = 0,
        ):
        super().__init__()
        
        self.ff_input = make_mlp(
            d_input=2, 
            d_hidden=d_ff, 
            d_output=d_model,
            n_layer=2
        )
        self.ff_output = make_mlp(
            d_input=d_model, 
            d_hidden=d_ff, 
            d_output=1,
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
        
        self.pooling_layers = [
            AttentionBlock(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_source = d_model,
                d_ff = d_ff,
                cross_attn = True,
                self_attn = False
            ) for _ in range(n_pool_layer)
        ]
        
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.embeddings = Parameter(data = torch.randn((1, 1, d_model)))
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.permute(1, 0, 2)
        x = self.ff_input(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask = ~mask)
        
        z = self.embeddings.expand(-1, x.shape[1], -1)
        
        for layer in self.pooling_layers:
            z = layer(z, src=x, src_padding_mask = ~mask)
            
        return self.ff_output(z).squeeze(0, 2)