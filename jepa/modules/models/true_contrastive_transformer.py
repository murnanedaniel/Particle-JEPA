from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from torch_geometric.nn import knn

from jepa.modules.models.true_contrastive import TrueContrastiveLearning
from jepa.modules.networks.transformer import Transformer
from jepa.modules.networks.aggregator import Aggregator
from toytrack.dataloaders import TracksDataset
from toytrack.transforms import TrackletPatchify


class TransformerContrastiveLearning(TrueContrastiveLearning):
    """
    True Contrastive Learning model that embeds sources and targets using the same encoder.
    It minimizes distances between seeds from the same particle and maximizes distances
    between seeds from different particles using contrastive loss.
    """

    def __init__(
        self,
        d_ff: int = 128,
        d_model: int = 32,
        d_embedding: int = 8,
        n_layers: int = 4,
        heads: int = 4,
        batch_size: int = 128,
        warmup: Optional[int] = 0,
        lr: Optional[float] = 1e-3,
        patience: Optional[int] = 10,
        factor: Optional[float] = 1,
        margin: Optional[float] = 1,
        random_context: Optional[bool] = True,
        dataset_args: Optional[Dict[str, Any]] = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            warmup=warmup,
            lr=lr,
            patience=patience,
            factor=factor,
            dataset_args=dataset_args,
        )

        self.embedding = Transformer(
            d_model=d_model,  # Note: Adjusted based on original Transformer
            d_ff=d_ff,
            heads=heads,
            n_layers=n_layers,
        )

        self.aggregator = Aggregator(
            d_model=d_model,
            d_ff=d_ff,
            d_embedding=d_embedding,
            heads=heads,
            n_layers=n_layers,
        )

        self.dataset_args = dataset_args
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step.

        Args:
            batch (dict): Batch of data containing tracklets and related information.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the training step.
        """
        if self.global_step == 0:
            print("Starting first training step...")

        x, mask, pids, edge_index, edge_mask, y = self._extract_batch_data(batch)

        if self.global_step == 0:
            self._debug_batch_shapes(x, mask, pids, edge_index, edge_mask, y)

        embedded_tracklets = self._embed_tracklets(x, mask)

        if self.global_step == 0:
            print(f"Embedded tracklets shape: {embedded_tracklets.shape}")

        embeddings_0, embeddings_1 = self._get_edge_embeddings(
            embedded_tracklets, edge_index, edge_mask, x.shape[1]
        )

        if self.global_step == 0:
            self._debug_embeddings(embeddings_0, embeddings_1)

        distances = self._compute_distances(embeddings_0, embeddings_1)

        loss = self._compute_loss(distances, y, edge_mask)

        if self.global_step == 0:
            print(f"Loss: {loss.item()}")

        self.log("train_loss", loss)
        self._log_learning_rate()

        if self.global_step == 0:
            print("Finished first training step.")

        return loss

    def shared_evaluation(self, batch, batch_idx):
        """
        Executes a single evaluation step.

        Args:
            batch (dict): Batch of data containing tracklets and related information.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing loss, efficiency, purity, and mean distances.
        """
        x, mask, pids, edge_index, edge_mask, y = self._extract_batch_data(batch)

        if self.global_step == 0:
            self._debug_batch_shapes(x, mask, pids, edge_index, edge_mask, y)
            
        embedded_tracklets = self._embed_tracklets(x, mask)
        embeddings_0, embeddings_1 = self._get_edge_embeddings(
            embedded_tracklets, edge_index, edge_mask, x.shape[1]
        )
        distances = self._compute_distances(embeddings_0, embeddings_1)
        loss = self._compute_loss(distances, y, edge_mask)
        efficiency, purity = self._calculate_metrics(distances, y, edge_mask)
        
        # Calculate mean distances for true and fake pairs
        mean_true_distance, mean_fake_distance = self._calculate_mean_distances(distances, y, edge_mask)

        self.log_dict({
            "val_loss": loss,
            "val_efficiency": efficiency,
            "val_purity": purity,
            "val_mean_true_distance": mean_true_distance,
            "val_mean_fake_distance": mean_fake_distance,
        })

        if batch_idx == 0:
            self._plot_evaluation(
                x, pids, edge_index, edge_mask, y, distances, embedded_tracklets
            )
            self._print_metrics(efficiency, purity, mean_true_distance, mean_fake_distance)

        return {
            "loss": loss,
            "efficiency": efficiency,
            "purity": purity,
            "mean_true_distance": mean_true_distance,
            "mean_fake_distance": mean_fake_distance,
            "embeddings": embedded_tracklets,
        }

    def _extract_batch_data(self, batch):
        """
        Extracts necessary data from the batch.

        Args:
            batch (dict): Batch of data.

        Returns:
            Tuple containing x, mask, pids, edge_index, edge_mask, y.
        """
        return (
            batch["x"],
            batch["mask"],
            batch["pids"],
            batch["edge_index"],
            batch["edge_mask"],
            batch["y"],
        )

    def _debug_batch_shapes(self, x, mask, pids, edge_index, edge_mask, y):
        """
        Prints the shapes of the batch components for debugging.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            pids (torch.Tensor): Particle IDs.
            edge_index (torch.Tensor): Edge indices.
            edge_mask (torch.Tensor): Edge mask.
            y (torch.Tensor): Labels.
        """
        print(f"Batch shapes: x={x.shape}, mask={mask.shape}, pids={pids.shape}, "
              f"edge_index={edge_index.shape}, edge_mask={edge_mask.shape}, y={y.shape}")

    def _embed_tracklets(self, x, mask):
        """
        Embeds the tracklets by reshaping and passing through the embedding and aggregator networks.

        Args:
            x (torch.Tensor): Tracklets with shape [N, P, H, C], 
            where N is the batch size, P is the number of particles, H is the number of hits, and C is the input dimension.
            mask (torch.Tensor): Mask with shape [N, P, H]

        Returns:
            torch.Tensor: Aggregated embedded tracklets with shape [N, P, C]
        """

        # Reshape x from [N, P, H, C] to [H, N*P, C]
        batch_size, num_particles, num_hits, input_dim = x.shape
        x_reshaped = x.view(batch_size * num_particles, num_hits, input_dim)
        x_reshaped = x_reshaped.permute(1, 0, 2)

        # Reshape mask from [N, P, H] to [N*P, H]
        mask_reshaped = mask.view(batch_size * num_particles, num_hits)

        # Pass through the embedding network
        embedded = self.embedding(x_reshaped, mask_reshaped)  # [H, N*P, C']

        # Pass through the aggregator
        aggregated = self.aggregator(embedded, mask_reshaped)  # [N*P, C]

        # Reshape back to [N, P, C]
        aggregated = aggregated.view(batch_size, num_particles, -1)

        return aggregated

    def _get_edge_embeddings(self, embedded_tracklets, edge_index, edge_mask, n_particles):
        """
        Retrieves embeddings for the edges based on edge indices and mask.

        Args:
            embedded_tracklets (torch.Tensor): Embedded tracklets with shape [N, P, C]
            edge_index (torch.Tensor): Edge indices with shape [N, E, 2]
            edge_mask (torch.Tensor): Edge mask with shape [N, E]
            n_particles (int): Number of particles.

        Returns:
            Tuple of torch.Tensor: (embeddings_0, embeddings_1) each with shape [Total_Edges, C]
        """
        if self.global_step == 0:
            max_edge_index = edge_index.max()
            print(f"Max edge index: {max_edge_index}")
            assert max_edge_index < n_particles, "edge_index contains out-of-bounds indices."

        # Expand batch indices
        batch_size = embedded_tracklets.size(0)
        device = embedded_tracklets.device
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, edge_index.size(1))

        # Gather embeddings
        embeddings_0 = embedded_tracklets[batch_indices, edge_index[..., 0]]
        embeddings_1 = embedded_tracklets[batch_indices, edge_index[..., 1]]

        # Apply edge mask
        embeddings_0 = embeddings_0[edge_mask]
        embeddings_1 = embeddings_1[edge_mask]

        return embeddings_0, embeddings_1

    def _debug_embeddings(self, embeddings_0, embeddings_1):
        """
        Prints the shapes of the embeddings for debugging.

        Args:
            embeddings_0 (torch.Tensor): First set of embeddings.
            embeddings_1 (torch.Tensor): Second set of embeddings.
        """
        print(f"Embeddings 0 shape: {embeddings_0.shape}")
        print(f"Embeddings 1 shape: {embeddings_1.shape}")
