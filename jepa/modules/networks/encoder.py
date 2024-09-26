import torch.nn as nn
from .transformer import Transformer
from .aggregator import Aggregator

class Encoder(nn.Module):
    """
    Encoder model that combines a Transformer and an Aggregator.
    """
    def __init__(
        self,
        d_input: int = 2,
        d_model: int = 32,
        d_ff: int = 128,
        d_embedding: int = 8,
        n_layers: int = 4,
        heads: int = 4,
        n_agg_layers: int = 2,
    ):
        super(Encoder, self).__init__()
        self.transformer = Transformer(
            d_input=d_input,
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
        )
        self.aggregator = Aggregator(
            d_model=d_model,
            d_ff=d_ff,
            d_embedding=d_embedding,
            heads=heads,
            n_layers=n_agg_layers,
        )

    def forward(self, x, mask):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor with shape [S, B, C]
            mask (torch.Tensor): Mask tensor with shape [B, S]

        where S is sequence length, B is batch size, and C is the input dimension

        Returns:
            torch.Tensor: Aggregated embeddings with shape [B, S, E]

        where E is the embedding dimension
        """
        embedded = self.transformer(x, mask)  # [S, B, D]
        aggregated = self.aggregator(embedded, mask)  # [B, S, E]
        return aggregated