import torch
from torch import nn
from torch.nn.parameter import Parameter
from .network_utils import AttentionBlock, make_mlp

class Transformer(nn.Module):
    def __init__(
            self, 
            d_input: int = 2,
            d_model: int = 32,
            d_ff: int = 128,
            heads: int = 4,
            n_layers: int = 6,
            dropout: float = 0,
        ):
        super().__init__()
        
        self.input_encoder = make_mlp(
            d_input=d_input, 
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
        """
        A transformer model that takes in a tensor of shape [N*P, H, C] and returns a tensor of shape [N, P, H, C],
        where N is the batch size, P is the number of particles, H is the number of hits, and C is the embedding dimension.
        """
        # Assume x is already reshaped to [N*P, H, C] and mask to [N*P, H]
        x = self.input_encoder(x)        
        for layer in self.encoder_layers:
            x = layer(x, padding_mask=~mask)

        return x