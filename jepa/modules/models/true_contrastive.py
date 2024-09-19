from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.nn import knn

from jepa.modules.base import BaseModule
from jepa.modules.networks.mlp import MLP
from toytrack.dataloaders import TracksDataset
from toytrack.transforms import TrackletPatchify


class TrueContrastiveLearning(BaseModule):
    """
    True Contrastive Learning model that embeds sources and targets using the same encoder.
    It minimizes distances between seeds from the same particle and maximizes distances
    between seeds from different particles using contrastive loss.
    """

    def __init__(
        self,
        d_ff: int = 1024,
        d_model: int = 6,
        n_layers: int = 6,
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

        self.embedding = MLP(
            d_input=8,
            d_hidden=d_ff,
            d_output=d_model,
            n_layers=n_layers,
        )

        self.dataset_args = dataset_args
        self.save_hyperparameters()

    def embed(self, x):
        """Placeholder for embedding function."""
        pass

    def _get_dataloader(self) -> DataLoader:
        """
        Creates and returns a DataLoader for the TracksDataset with TrackletPatchify transformation.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        patchify = TrackletPatchify(num_patches_per_track=2)
        self.dataset = TracksDataset(
            self.dataset_args,
            transform=patchify,
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams.get("num_workers", 16),
            collate_fn=self.dataset.collate_fn,
        )
        return dataloader

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

        x_flat = self._flatten_tracklets(x)

        if self.global_step == 0:
            print(f"Flattened x shape: {x_flat.shape}")

        embedded_tracklets = self._embed_tracklets(x_flat)

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
        print(f"x: {x} \n mask: {mask} \n pids: {pids} \n edge_index: {edge_index} \n edge_mask: {edge_mask} \n y: {y}")

        x_flat = self._flatten_tracklets(x)
        embedded_tracklets = self._embed_tracklets(x_flat)
        embeddings_0, embeddings_1 = self._get_edge_embeddings(
            embedded_tracklets, edge_index, edge_mask, x.shape[1]
        )
        distances = self._compute_distances(embeddings_0, embeddings_1)
        print(f"embedded_tracklets: {embedded_tracklets} \n distances: {distances}")
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

    def _flatten_tracklets(self, x):
        """
        Flattens the tracklets tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_particles, n_hits, 2).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, n_particles, n_hits * 2).
        """
        batch_size, n_particles, n_hits, _ = x.shape
        return x.view(batch_size, n_particles, -1)

    def _embed_tracklets(self, x_flat):
        """
        Embeds the flattened tracklets using the embedding network.

        Args:
            x_flat (torch.Tensor): Flattened tracklets.

        Returns:
            torch.Tensor: Embedded tracklets.
        """
        return self.embedding(x_flat)

    def _get_edge_embeddings(self, embedded_tracklets, edge_index, edge_mask, n_particles):
        """
        Retrieves embeddings for the edges based on edge indices and mask.

        Args:
            embedded_tracklets (torch.Tensor): Embedded tracklets.
            edge_index (torch.Tensor): Edge indices.
            edge_mask (torch.Tensor): Edge mask.
            n_particles (int): Number of particles.

        Returns:
            Tuple of torch.Tensor: (embeddings_0_masked, embeddings_1_masked)
        """
        if self.global_step == 0:
            max_edge_index = edge_index.max()
            print(f"Max edge index: {max_edge_index}")
            assert max_edge_index < n_particles, "edge_index contains out-of-bounds indices."

        batch_size = embedded_tracklets.size(0)
        batch_indices = torch.arange(batch_size, device=edge_index.device).unsqueeze(1).expand(-1, edge_index.size(1))

        embeddings_0 = embedded_tracklets[batch_indices, edge_index[..., 0]][edge_mask]
        embeddings_1 = embedded_tracklets[batch_indices, edge_index[..., 1]][edge_mask]

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

    def _compute_distances(self, embeddings_0, embeddings_1):
        """
        Computes Euclidean distances between pairs of embeddings.

        Args:
            embeddings_0 (torch.Tensor): First set of embeddings.
            embeddings_1 (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: Distances between embeddings.
        """
        distances = torch.norm(embeddings_0 - embeddings_1, dim=-1)
        if self.global_step == 0:
            print(f"Distances shape: {distances.shape}")
            print(f"Sample distances: {distances[:5]}")
        return distances

    def _compute_loss(self, distances, y, edge_mask):
        """
        Computes the hinge embedding loss.

        Args:
            distances (torch.Tensor): Distances between embeddings.
            y (torch.Tensor): Labels.
            edge_mask (torch.Tensor): Edge mask.

        Returns:
            torch.Tensor: Computed loss.
        """
        truth_for_loss = y[edge_mask].float() * 2 - 1  # Convert 1/0 to 1/-1 for hinge loss
        assert truth_for_loss.shape == distances.shape, f"Mismatch between truth_for_loss and distances shapes: truth_for_loss={truth_for_loss.shape} != distances={distances.shape}"
        loss = F.hinge_embedding_loss(distances, truth_for_loss, margin=self.hparams.margin)
        
        print(f"truth_for_loss: {truth_for_loss}")
        return loss

    def _log_learning_rate(self):
        """Logs the current learning rate."""
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", current_lr)
        if self.global_step == 0:
            print(f"Current learning rate: {current_lr}")

    def _calculate_metrics(self, distances, y, edge_mask):
        """
        Calculates efficiency and purity based on a distance threshold.

        Args:
            distances (torch.Tensor): Distances between embeddings.
            y (torch.Tensor): Labels.
            edge_mask (torch.Tensor): Edge mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Efficiency and Purity.
        """
        threshold = self.hparams.margin
        truth = y[edge_mask]
        true_positives = (distances < threshold) & truth
        false_positives = (distances < threshold) & ~truth
        false_negatives = (distances >= threshold) & truth

        efficiency = true_positives.sum().float() / (true_positives.sum() + false_negatives.sum()).float()
        purity = true_positives.sum().float() / (true_positives.sum() + false_positives.sum()).float()

        return efficiency, purity

    def _calculate_mean_distances(self, distances, y, edge_mask):
        """
        Calculates mean distances for true pairs and fake pairs.

        Args:
            distances (torch.Tensor): Distances between embeddings.
            y (torch.Tensor): Labels.
            edge_mask (torch.Tensor): Edge mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean true distance and mean fake distance.
        """
        true_pairs = y[edge_mask] == 1
        fake_pairs = y[edge_mask] == 0
        
        mean_true_distance = distances[true_pairs].mean() if true_pairs.any() else torch.tensor(0.0)
        mean_fake_distance = distances[fake_pairs].mean() if fake_pairs.any() else torch.tensor(0.0)
        
        return mean_true_distance, mean_fake_distance

    def _plot_evaluation(
        self, x, pids, edge_index, edge_mask, y, distances, embedded_tracklets
    ):
        """
        Generates and displays plots for evaluation.

        Args:
            x (torch.Tensor): Input tensor.
            pids (torch.Tensor): Particle IDs.
            edge_index (torch.Tensor): Edge indices.
            edge_mask (torch.Tensor): Edge mask.
            y (torch.Tensor): Labels.
            distances (torch.Tensor): Distances between embeddings.
            embedded_tracklets (torch.Tensor): Embedded tracklets.
        """
        # Move to CPU and select first event
        x_cpu = x[0].cpu()
        pids_cpu = pids[0].cpu()
        edge_index_cpu = edge_index[0].cpu()
        edge_mask_cpu = edge_mask.cpu()
        truth_cpu = y[edge_mask].cpu()
        distances_cpu = distances.cpu()
        embedded_tracklets_cpu = embedded_tracklets[0].detach().cpu()

        # Plot original tracklets
        self._plot_original_tracklets(x_cpu, pids_cpu)

        # Plot embedded tracklets with PCA
        embedded_tracklets_2d = PCA(n_components=2).fit_transform(embedded_tracklets_cpu.numpy())
        self._plot_embedded_tracklets_with_edges(
            pids_cpu, embedded_tracklets_2d, edge_index_cpu, edge_mask_cpu, truth_cpu, distances_cpu
        )

    def _plot_original_tracklets(self, x_cpu, pids_cpu):
        """
        Plots the original tracklets for the first event.

        Args:
            x_cpu (torch.Tensor): Input tensor for the first event.
            pids_cpu (torch.Tensor): Particle IDs for the first event.
        """
        fig, ax = plt.subplots()
        for pid in torch.unique(pids_cpu):
            pid_mask = pids_cpu == pid
            ax.scatter(x_cpu[pid_mask, :, 0], x_cpu[pid_mask, :, 1], label=f'PID {pid.item()}')
        ax.legend()
        ax.set_title('Original Tracklets (First Event)')
        ax.set_aspect('equal', 'box')
        plt.show()

    def _plot_embedded_tracklets_with_edges(
        self, pids_cpu, embedded_tracklets_2d, edge_index_cpu, edge_mask_cpu, truth_cpu, distances_cpu
    ):
        """
        Plots the embedded tracklets using 2D PCA along with evaluation edges and distances.

        Args:
            pids_cpu (torch.Tensor): Particle IDs for the first event.
            embedded_tracklets_2d (ndarray): 2D PCA transformed embeddings.
            edge_index_cpu (torch.Tensor): Edge indices for the first event.
            edge_mask_cpu (torch.Tensor): Edge mask for the first event.
            truth_cpu (torch.Tensor): Truth labels for the edges.
            distances_cpu (torch.Tensor): Distances between embeddings.
        """
        fig, ax = plt.subplots()
        for pid in torch.unique(pids_cpu):
            pid_mask = pids_cpu == pid
            ax.scatter(
                embedded_tracklets_2d[pid_mask, 0],
                embedded_tracklets_2d[pid_mask, 1],
                label=f'PID {pid.item()}',
                color=plt.cm.tab10(pid.item() % 10),
            )

        # Plot evaluation edges
        for i in range(edge_index_cpu.shape[0]):
            start = embedded_tracklets_2d[edge_index_cpu[i, 0]]
            end = embedded_tracklets_2d[edge_index_cpu[i, 1]]
            color = 'blue' if truth_cpu[i] else 'red'
            distance = distances_cpu[i].item()
            alpha = max(0, self.hparams.margin - distance) / self.hparams.margin
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=alpha)

            # Add distance label
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.annotate(f'{distance:.2f}', (mid_x, mid_y), alpha=alpha, fontsize=6)

        ax.legend()
        ax.set_title('Embedded Tracklets (2D PCA) with Evaluation Edges and Distances - First Event')
        plt.show()

    def _print_metrics(self, efficiency, purity, mean_true_distance, mean_fake_distance):
        """
        Prints the evaluation metrics for the first event.

        Args:
            efficiency (torch.Tensor): Computed efficiency.
            purity (torch.Tensor): Computed purity.
            mean_true_distance (torch.Tensor): Mean distance for true pairs.
            mean_fake_distance (torch.Tensor): Mean distance for fake pairs.
        """
        print("First Event Metrics:")
        print(f"  Efficiency: {efficiency:.4f}")
        print(f"  Purity: {purity:.4f}")
        print(f"  Mean True Distance: {mean_true_distance:.4f}")
        print(f"  Mean Fake Distance: {mean_fake_distance:.4f}")
