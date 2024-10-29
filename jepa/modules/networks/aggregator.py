import torch
from torch import nn
from torch.nn.parameter import Parameter
from .network_utils import AttentionBlock, make_mlp

class Aggregator(nn.Module):
    def __init__(
            self, 
            d_model: int = 32,
            d_ff: int = 128,
            heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0,
            d_embedding: int = 8,
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
        self.embeddings_encoder = make_mlp(
            d_input=d_model, 
            d_hidden=d_ff, 
            d_output=d_embedding,
            n_layer=2
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Aggregates the embedded tracklets.

        Args:
            x (torch.Tensor): Embedded tracklets with shape [S, B, C].
            mask (torch.Tensor): Mask with shape [B, S].

        Returns:
            torch.Tensor: Aggregated embeddings with shape [B, E].
        """
        z = self.embeddings.expand(-1, x.size(1), -1)  # [1, B, C]
        for layer in self.encoder_layers:
            z = layer(z, src=x, src_padding_mask=~mask)
        aggregated = z.squeeze(0)  # [B, E]
        aggregated = self.embeddings_encoder(aggregated)
        return aggregated 