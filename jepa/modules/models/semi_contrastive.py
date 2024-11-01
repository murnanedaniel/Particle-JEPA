import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import knn
from typing import Optional, Dict, Any
from jepa.modules import JEPA
from jepa.modules.networks.transformer import Transformer

class SemiContrastiveLearning(JEPA):
    """
    The only tweak required for this model in the case of single-particle events
    that are partitioned into innermost and outermost tracklets is to set the Predictor
    to be a passthrough function. Then the encoder is exactly trying to predict the outermost
    tracklet encoding.

    This is not quite "strong" contrastive learning, since we are just matching
    encodings, rather than looking at positive and negative distances.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the predictor to be a passthrough function
        self.predictor = nn.Identity()

    def training_step(self, batch, batch_idx):
        x, mask, *_ = batch
        context_mask, target_mask, inner_target_flag = self.sample_context(x, mask)
        x = x.permute(1, 0, 2)
        target = self.embed_target(x, mask, target_mask)
        context = self.encoder(x, context_mask, context_mask)
        prediction = self.predictor(context)

        loss = F.mse_loss(prediction, target)

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx):
        x, mask, *_ = batch
        context_mask, target_mask, inner_target_flag = self.sample_context(x, mask)
        x = x.permute(1, 0, 2)
        target = self.embed_target(x, mask, target_mask)
        context = self.encoder(x, context_mask, context_mask)
        prediction = self.predictor(context)
        
        loss = F.mse_loss(prediction, target)
        source_target_difference = F.mse_loss(prediction, context)

        try:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict({
                "val_loss": loss,
                "source_target_difference": source_target_difference,
                "lr": lr
            })
        except:
            pass

        return {
            "loss": loss,
            "prediction": prediction,
            "target": target,
            "context": context
        }