from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.nn import knn

from jepa.modules.base import BaseModule
from jepa.modules.networks.encoder import Encoder
from toytrack.dataloaders import TracksDataset
from toytrack.transforms import TrackletPatchify


class JEA(BaseModule):
    """
    This is a compromise between true supervised contrastive learning, and the full JEPA model.
    In this case, we take pairs of tracklets that are known to come from the same particle, and
    embed them with two models, a context encoder and a target encoder. The context encoder is
    updated with a Smooth L1 loss, while the target encoder is updated with an exponential moving
    average of the context encoder.
    """

    def __init__(
        self,
        d_input: int = 2,
        d_ff: int = 128,
        d_model: int = 32,
        d_embedding: int = 8,
        n_layers: int = 4,
        n_agg_layers: int = 2,
        heads: int = 4,
        batch_size: int = 128,
        warmup: Optional[int] = 0,
        lr: Optional[float] = 1e-3,
        patience: Optional[int] = 10,
        factor: Optional[float] = 1,
        margin: Optional[float] = 1,
        random_context: Optional[bool] = True,
        dataset_args: Optional[Dict[str, Any]] = {},
        ema_decay: Optional[float] = 0.99,
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

        self.encoder = Encoder(
            d_input=d_input,
            d_model=d_model,
            d_ff=d_ff,
            d_embedding=d_embedding,
            n_layers=n_layers,
            heads=heads,
            n_agg_layers=n_agg_layers,
        )
        self.ema_encoder = Encoder(
            d_input=d_input,
            d_model=d_model,
            d_ff=d_ff,
            d_embedding=d_embedding,
            n_layers=n_layers,
            heads=heads,
            n_agg_layers=n_agg_layers,
        )

        self.ema_decay = ema_decay
        self.dataset_args = dataset_args
        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)

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
            num_workers=self.hparams.get("num_workers", 32),
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

        # Embed context (updated) and target (fixed) tracklets
        embedded_context_tracklets = self._embed_context_tracklets(x, mask)
        embedded_target_tracklets = self._embed_target_tracklets(x, mask)

        if self.global_step == 0:
            print(f"Embedded context tracklets shape: {embedded_context_tracklets.shape}")
            print(f"Embedded target tracklets shape: {embedded_target_tracklets.shape}")

        # Get edge embeddings
        embeddings_0, embeddings_1, edge_index = self._get_edge_embeddings(
            embedded_context_tracklets, embedded_target_tracklets, edge_index, edge_mask, x.shape[1]
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
            
        embedded_context_tracklets = self._embed_context_tracklets(x, mask)
        embedded_target_tracklets = self._embed_target_tracklets(x, mask)

        embeddings_0, embeddings_1, edge_index = self._get_edge_embeddings(
            embedded_context_tracklets, embedded_target_tracklets, edge_index, edge_mask, x.shape[1]
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
                x, pids, edge_index, edge_mask, y, distances, embedded_context_tracklets, embedded_target_tracklets
            )
            self._print_metrics(efficiency, purity, mean_true_distance, mean_fake_distance)

        return {
            "loss": loss,
            "efficiency": efficiency,
            "purity": purity,
            "mean_true_distance": mean_true_distance,
            "mean_fake_distance": mean_fake_distance,
            "embeddings_0": embedded_context_tracklets,
            "embeddings_1": embedded_target_tracklets,
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

    def _reshape_tracklets(self, x, mask):

         # Reshape x from [N, P, H, C] to [H, N*P, C]
        batch_size, num_particles, num_hits, input_dim = x.shape
        x_reshaped = x.view(batch_size * num_particles, num_hits, input_dim)
        x_reshaped = x_reshaped.permute(1, 0, 2)

        # Reshape mask from [N, P, H] to [N*P, H]
        mask_reshaped = mask.view(batch_size * num_particles, num_hits)

        return x_reshaped, mask_reshaped, batch_size, num_particles
    
    @torch.no_grad
    def _embed_target_tracklets(self, x, mask):
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
        x_reshaped, mask_reshaped, batch_size, num_particles = self._reshape_tracklets(x, mask)

        encoded = self.ema_encoder(x_reshaped, mask_reshaped)

        # Reshape back
        embedded = encoded.view(batch_size, num_particles, -1)

        return embedded

    def _embed_context_tracklets(self, x, mask):
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
        x_reshaped, mask_reshaped, batch_size, num_particles = self._reshape_tracklets(x, mask)

        embedded = self.encoder(x_reshaped, mask_reshaped)

        # Reshape back
        embedded = embedded.view(batch_size, num_particles, -1)

        return embedded

    def _get_edge_embeddings(self, embedded_context_tracklets, embedded_target_tracklets, edge_index, edge_mask, n_particles):
        """
        Retrieves embeddings for the edges based on edge indices and mask.

        Args:
            embedded_context_tracklets (torch.Tensor): Embedded context tracklets with shape [N, P, C]
            embedded_target_tracklets (torch.Tensor): Embedded target tracklets with shape [N, P, C]
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
        batch_size = embedded_context_tracklets.size(0)
        device = embedded_context_tracklets.device
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, edge_index.size(1))

        # Gather embeddings
        edge_index = self.random_flip_edges(edge_index)
        embeddings_0 = embedded_context_tracklets[batch_indices, edge_index[..., 0]]
        embeddings_1 = embedded_target_tracklets[batch_indices, edge_index[..., 1]]

        # Apply edge mask
        embeddings_0 = embeddings_0[edge_mask]
        embeddings_1 = embeddings_1[edge_mask]

        return embeddings_0, embeddings_1, edge_index

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
        Computes the L2 loss (equal to squared Euclidean distance)

        Args:
            distances (torch.Tensor): Distances between embeddings.
            y (torch.Tensor): Labels.
            edge_mask (torch.Tensor): Edge mask.

        Returns:
            torch.Tensor: Computed loss.
        """
        assert y[edge_mask].shape == distances.shape, f"Mismatch between y[edge_mask] and distances shapes: y[edge_mask]={y[edge_mask].shape} != distances={distances.shape}"

        true_pairs = y[edge_mask] == 1

        loss = distances[true_pairs].pow(2).mean()

        return loss

    def random_flip_edges(self, edge_index, flip_prob=0.5):
        """
        Randomly flips the source and target of edge indices within each batch.

        Args:
            edge_index (torch.Tensor): Edge indices of shape [batch_size, num_edges, 2].
            flip_prob (float): Probability of flipping each edge. Defaults to 0.5.

        Returns:
            torch.Tensor: Flipped edge indices of shape [batch_size, num_edges, 2].
        """

        # Generate a random mask for flipping with shape [batch_size, num_edges]
        flip_mask = torch.rand(edge_index.size(0), edge_index.size(1), device=edge_index.device) < flip_prob

        # Expand flip_mask to match the shape [batch_size, num_edges, 2]
        flip_mask_expanded = flip_mask.unsqueeze(-1).expand_as(edge_index)

        # Identify positions to flip (only the second dimension)
        edge_index_flipped = torch.where(
            flip_mask_expanded,
            edge_index.flip(-1),
            edge_index
        )

        return edge_index_flipped
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.update_ema_params()

    @torch.no_grad
    def update_ema_params(self):
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data.copy_(ema_param.data * self.ema_decay + (1 - self.ema_decay) * param.data)
        

    # ------------------- DEBUGGING + VISUALISATION ------------------- #

    def _debug_embeddings(self, embeddings_0, embeddings_1):
        """
        Prints the shapes of the embeddings for debugging.

        Args:
            embeddings_0 (torch.Tensor): First set of embeddings.
            embeddings_1 (torch.Tensor): Second set of embeddings.
        """
        print(f"Embeddings 0 shape: {embeddings_0.shape}")
        print(f"Embeddings 1 shape: {embeddings_1.shape}")

    
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
        # Threshold is 10% of max distance
        threshold = self.hparams.get('threshold', 0.1)
        truth = y[edge_mask]
        true_positives = (distances < threshold) & truth
        false_positives = (distances < threshold) & ~truth
        false_negatives = (distances >= threshold) & truth

        tp_sum = true_positives.sum().float()
        fp_sum = false_positives.sum().float()
        fn_sum = false_negatives.sum().float()

        efficiency_denominator = tp_sum + fn_sum
        purity_denominator = tp_sum + fp_sum

        efficiency = tp_sum / efficiency_denominator if efficiency_denominator > 0 else torch.tensor(0.0)
        purity = tp_sum / purity_denominator if purity_denominator > 0 else torch.tensor(0.0)

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
        self, x, pids, edge_index, edge_mask, y, distances, embedded_context_tracklets, embedded_target_tracklets
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
            embedded_context_tracklets (torch.Tensor): Embedded context tracklets.
            embedded_target_tracklets (torch.Tensor): Embedded target tracklets.
        """
        # Move to CPU and select first event
        x_cpu = x[0].cpu()
        pids_cpu = pids[0].cpu()
        edge_mask_cpu = edge_mask[0].cpu()
        edge_index_cpu = edge_index[0].cpu()[edge_mask_cpu]
        truth_cpu = y[0][edge_mask[0]].cpu()
        distances_cpu = distances[:truth_cpu.shape[0]].cpu()
        embedded_context_tracklets_cpu = embedded_context_tracklets[0].detach().cpu()
        embedded_target_tracklets_cpu = embedded_target_tracklets[0].detach().cpu()

        # Plot original tracklets
        self._plot_original_tracklets(x_cpu, pids_cpu)

        # Combine context and target tracklets
        combined_tracklets = torch.cat([embedded_context_tracklets_cpu, embedded_target_tracklets_cpu], dim=0)

        # Plot embedded tracklets with PCA
        pca = PCA(n_components=2)
        combined_tracklets_2d = pca.fit_transform(combined_tracklets.numpy())
        embedded_context_tracklets_2d = combined_tracklets_2d[:len(embedded_context_tracklets_cpu)]
        embedded_target_tracklets_2d = combined_tracklets_2d[len(embedded_context_tracklets_cpu):]

        self._plot_embedded_tracklets_with_edges(
            pids_cpu, embedded_context_tracklets_2d, embedded_target_tracklets_2d, edge_index_cpu, edge_mask_cpu, truth_cpu, distances_cpu
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
        self, pids_cpu, embedded_context_tracklets_2d, embedded_target_tracklets_2d, edge_index_cpu, edge_mask_cpu, truth_cpu, distances_cpu
    ):
        """
        Plots the embedded tracklets using 2D PCA along with evaluation edges and distances.

        Args:
            pids_cpu (torch.Tensor): Particle IDs for the first event.
            embedded_context_tracklets_2d (ndarray): 2D PCA transformed context tracklets.
            embedded_target_tracklets_2d (ndarray): 2D PCA transformed target tracklets.
            edge_index_cpu (torch.Tensor): Edge indices for the first event. Shape [num_edges, 2]
            edge_mask_cpu (torch.Tensor): Edge mask for the first event.
            truth_cpu (torch.Tensor): Truth labels for the edges.
            distances_cpu (torch.Tensor): Distances between embeddings.
        """
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size to accommodate legend
        margin = distances_cpu.max().item() / 10

        # Extract unique context and target indices involved in edges
        context_indices = edge_index_cpu[truth_cpu, 0].unique()
        target_indices = edge_index_cpu[truth_cpu, 1].unique()
        
        # Get the unique PIDs involved in both context and target indices
        involved_pids_context = pids_cpu[context_indices].unique()
        involved_pids_target = pids_cpu[target_indices].unique()
        involved_pids = torch.cat([involved_pids_context, involved_pids_target]).unique()
    
        for pid in involved_pids:
            # Context Involved
            pid_mask_context = (pids_cpu[context_indices] == pid)
            context_involved = context_indices[pid_mask_context]
            
            # Target Involved
            pid_mask_target = (pids_cpu[target_indices] == pid)
            target_involved = target_indices[pid_mask_target]
            
            # Scatter plot for context tracklets
            ax.scatter(
                embedded_context_tracklets_2d[context_involved, 0],
                embedded_context_tracklets_2d[context_involved, 1],
                label=f'PID {pid.item()} Context',
                color=plt.cm.tab10(pid.item() % 10),
                marker='o',
            )

            # Scatter plot for target tracklets
            ax.scatter(
                embedded_target_tracklets_2d[target_involved, 0],
                embedded_target_tracklets_2d[target_involved, 1],
                label=f'PID {pid.item()} Target',
                color=plt.cm.tab10(pid.item() % 10),
                marker='x',
            )

        # Move legend outside the plot
        ax.set_title('Embedded Tracklets Context and Target (2D PCA) with Evaluation Edges and Distances - First Event')
        plt.tight_layout()  # Adjust the layout to prevent clipping of the legend
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
        print("First Batch Metrics:")
        print(f"  Efficiency: {efficiency:.4f}")
        print(f"  Purity: {purity:.4f}")
        print(f"  Mean True Distance: {mean_true_distance:.4f}")
        print(f"  Mean Fake Distance: {mean_fake_distance:.4f}")