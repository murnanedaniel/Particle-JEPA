import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import knn
from typing import Optional, Dict, Any
from jepa.modules import BaseModule
from jepa.utils import AttentionBlock, make_mlp

class Transformer(BaseModule):
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
            curriculum: Optional[int] = 0,
            min_scale: Optional[float] = 0.,
            dataset_args: Optional[Dict[str, Any]] = {},
        ):
        
        super().__init__(
            batch_size=batch_size,
            warmup=warmup,
            lr=lr,
            patience=patience,
            factor=factor,
            curriculum=curriculum,
            min_scale=min_scale,
            dataset_args=dataset_args
        )
        
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
        
    def sample_context(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Context sampling process:
        1. Pick a random index - this is the center point
        2. Pick a random length - this is the length of the context
        3. Run KNN on the center point
        4. Return the context
        """

        center = torch.randint(0, x.shape[1], (1,))
        length = torch.randint(1, x.shape[1] // 5, (1,))
        
        context = knn(x, x[center], length)
        context_mask = mask[context]

        return context, context_mask
    
    def sample_target(self, x: torch.Tensor, mask: torch.Tensor, context_mask: torch.Tensor):
        """
        Target sampling process:
        1. Overlay this batch with another batch
        2. Pick a random index and random length
        3. Run KNN on the index and length
        4. Return the target and the event label
        """

        random_event = self.trainloader.dataset[torch.randint(0, len(self.trainloader.dataset), (1,))]

        x = torch.cat([x, random_event[0]], dim=1)
        mask = torch.cat([mask, random_event[1]], dim=1)
        label = torch.cat([torch.ones(x.shape[1] - random_event[0].shape[1]), torch.zeros(random_event[0].shape[1])])

        random_index = torch.randint(0, x.shape[1], (1,))
        random_length = torch.randint(1, x.shape[1] // 5, (1,))

        target = knn(x, x[random_index], random_length)
        target_mask = mask[target]

        return target, target_mask, label


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.permute(1, 0, 2)
        x = self.ff_input(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask = ~mask)
        
        z = self.embeddings.expand(-1, x.shape[1], -1)
        
        for layer in self.pooling_layers:
            z = layer(z, src=x, src_padding_mask = ~mask)
            
        return self.ff_output(z).squeeze(0, 2)

    def predict(self, x, mask):
        return self(x, mask)