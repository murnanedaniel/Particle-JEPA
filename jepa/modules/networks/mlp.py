import torch.nn as nn
import torch.nn.functional as F

from jepa.modules.networks.network_utils import make_mlp

class MLP(nn.Module):
    def __init__(self, 
        d_input: int,
        d_output: int,
        d_hidden: int,
        n_layers: int,
        normalize_output: bool = False,
        ):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        self.network = make_mlp(
            d_input=d_input,
            d_hidden=d_hidden, 
            d_output=d_output,
            n_layer=n_layers
        )

        self.normalize_output = normalize_output

    def forward(self, x):
        x_out = self.network(x)
        if self.normalize_output:
            x_out = F.normalize(x_out)
        return x_out