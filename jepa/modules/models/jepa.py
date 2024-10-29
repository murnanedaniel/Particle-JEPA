from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.nn import knn

from jepa.modules.base import BaseModule
from jepa.modules.networks.encoder import Encoder
from jepa.modules.networks.network_utils import make_mlp
from jepa.utils.sampling_utils import WedgePatchify
from toytrack.dataloaders import TracksDataset


class JEPA(BaseModule):
    """
    This is the full JEPA model. We take slices of the detector (wedges or annuli) and embed
    them with two encoders, a context encoder and a target encoder. The context encoder is
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

        self.predictor = make_mlp(
            d_input=d_embedding,
            d_hidden=d_ff,
            d_output=d_embedding,
            n_layer=3
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
        Creates and returns a DataLoader for the TracksDataset with WedgePatchify transformation.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        patchify = WedgePatchify(
            phi_range=np.pi / 4, 
            radius_midpoint = (self.dataset_args["detector"]["max_radius"] + self.dataset_args["detector"]["min_radius"]) / 2
        )
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

        # Get inputs
        x, mask, context_mask, target_mask, pids = self._extract_batch_data(batch)

        if self.global_step == 0:
            self._debug_batch_shapes(x, mask, context_mask, target_mask, pids)

        # Embed context (updated) and target (fixed) tracklets
        embedded_context_tracklets = self._embed_context_tracklets(x, context_mask, batch)
        embedded_target_tracklets = self._embed_target_tracklets(x, target_mask, batch)
        predicted_embedded_target_tracklets = self.predictor(embedded_context_tracklets)

        # Add NaN checks
        self._check_nan(embedded_context_tracklets, "embedded_context_tracklets", batch)
        self._check_nan(embedded_target_tracklets, "embedded_target_tracklets", batch)
        self._check_nan(predicted_embedded_target_tracklets, "predicted_embedded_target_tracklets", batch)

        if self.global_step == 0:
            print(f"Embedded context tracklets shape: {embedded_context_tracklets.shape}")
            print(f"Embedded target tracklets shape: {embedded_target_tracklets.shape}")
            print(f"Predicted embedded target tracklets shape: {predicted_embedded_target_tracklets.shape}")

        if self.global_step == 0:
            self._debug_embeddings(embedded_target_tracklets, predicted_embedded_target_tracklets)

        distances = self._compute_distances(embedded_context_tracklets, embedded_target_tracklets)
        loss = self._compute_loss(distances)

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
        x, mask, context_mask, target_mask, pids = self._extract_batch_data(batch)

        if self.global_step == 0:
            self._debug_batch_shapes(x, mask, context_mask, target_mask, pids)

        embedded_context_tracklets = self._embed_context_tracklets(x, context_mask, batch)
        embedded_target_tracklets = self._embed_target_tracklets(x, target_mask, batch)
        predicted_embedded_target_tracklets = self.predictor(embedded_context_tracklets)

        distances = self._compute_distances(embedded_context_tracklets, embedded_target_tracklets)
        loss = self._compute_loss(distances)
        
        # Calculate mean distances for true and fake pairs
        mean_true_distance, mean_fake_distance = self._calculate_mean_distances(distances, embedded_target_tracklets, predicted_embedded_target_tracklets)

        self.log_dict({
            "val_loss": loss,
            "val_mean_true_distance": mean_true_distance,
            "val_mean_fake_distance": mean_fake_distance,
        })

        if batch_idx == 0:
            self._plot_evaluation(
                x, context_mask, target_mask, distances, embedded_context_tracklets, embedded_target_tracklets
            )
            self._print_metrics(mean_true_distance, mean_fake_distance)

        return {
            "loss": loss,
            "mean_true_distance": mean_true_distance,
            "mean_fake_distance": mean_fake_distance,
            "embedded_context_tracklets": embedded_context_tracklets,
            "embedded_target_tracklets": embedded_target_tracklets,
            "predicted_embedded_target_tracklets": predicted_embedded_target_tracklets,
        }

    def _enforce_no_nans(self, embedded_context_tracklets, embedded_target_tracklets, predicted_embedded_target_tracklets, batch):
        """
        Enforces no nans in the embeddings.
        """
        
        nan_indices = (
            torch.isnan(embedded_context_tracklets) |
            torch.isnan(embedded_target_tracklets) |
            torch.isnan(predicted_embedded_target_tracklets)
        )

        nan_row_indices = nan_indices.any(dim=1)

        # Remove those rows that are nans
        if nan_row_indices.any():
            embedded_context_tracklets = embedded_context_tracklets[~nan_row_indices]
            embedded_target_tracklets = embedded_target_tracklets[~nan_row_indices]
            predicted_embedded_target_tracklets = predicted_embedded_target_tracklets[~nan_row_indices]

        return embedded_context_tracklets, embedded_target_tracklets, predicted_embedded_target_tracklets

    def _extract_batch_data(self, batch):
        """
        Extracts necessary data from the batch.

        Args:
            batch (dict): Batch of data.

        Returns:
            Tuple containing x, mask, context_mask, target_mask and pids
        """
        x, mask, context_mask, target_mask, pids = (
            batch["x"],
            batch["mask"],
            batch["context_mask"],
            batch["target_mask"],
            batch["pids"],
        )

        # Check for any batch entries that have all false context or target masks
        valid_rows = context_mask.any(dim=1) & target_mask.any(dim=1)
        if not valid_rows.all():
            # Remove those rows
            x = x[valid_rows]
            context_mask = context_mask[valid_rows]
            target_mask = target_mask[valid_rows]
            mask = mask[valid_rows]
            pids = pids[valid_rows]

        return x, mask, context_mask, target_mask, pids

    def _debug_batch_shapes(self, x, mask, context_mask, target_mask, pids):
        """
        Prints the shapes of the batch components for debugging.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            pids (torch.Tensor): Particle IDs.
            context_mask (torch.Tensor): Context mask.
            target_mask (torch.Tensor): Target mask.
        """
        print(f"Batch shapes: x={x.shape}, mask={mask.shape}, context_mask={context_mask.shape}, target_mask={target_mask.shape}, pids={pids.shape}")
    
    @torch.no_grad
    def _embed_target_tracklets(self, x, mask, batch):
        """
        Embeds the tracklets by reshaping and passing through the embedding and aggregator networks.

        Args:
            x (torch.Tensor): Tracklets with shape [N, H, C], 
            where N is the batch size, H is the number of hits, and C is the input dimension.
            mask (torch.Tensor): Mask with shape [N, H]

        Returns:
            torch.Tensor: Aggregated embedded tracklets with shape [N, C]
        """

        x = x.transpose(0, 1) # Transformer expects [S, B, C]
        embedded = self.ema_encoder(x, mask)

        return embedded

    def _embed_context_tracklets(self, x, mask, batch):
        """
        Embeds the tracklets by reshaping and passing through the embedding and aggregator networks.

        Args:
            x (torch.Tensor): Tracklets with shape [N, H, C], 
            where N is the batch size, H is the number of hits, and C is the input dimension.
            mask (torch.Tensor): Mask with shape [N, H]

        Returns:
            torch.Tensor: Aggregated embedded tracklets with shape [N, C]
        """

        x = x.transpose(0, 1) # Transformer expects [S, B, C]
        embedded = self.encoder(x, mask)

        return embedded

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

    def _compute_loss(self, distances):
        """
        Computes the L1 loss (Mean Absolute Error)

        Args:
            distances (torch.Tensor): Distances between embeddings.

        Returns:
            torch.Tensor: Computed loss.
        """

        loss = distances.abs().mean()

        return loss
    
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

    def _calculate_mean_distances(self, distances, target_embeddings, pred_embeddings):
        """
        Calculates mean distances for true pairs and fake pairs.

        Args:
            distances (torch.Tensor): Distances between embeddings.
            target_embeddings (torch.Tensor): Target embeddings.
            pred_embeddings (torch.Tensor): Predicted embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean true distance and mean fake distance.
        """

        # True pair distances are given by distances
        mean_true_distance = distances.mean()
        
        # False pair distances are calculated from random combinations where i != j
        num_embeddings = target_embeddings.shape[0]
        i = torch.randint(0, num_embeddings, (num_embeddings,))
        j = torch.randint(0, num_embeddings, (num_embeddings,))
        false_mask = i != j
        i, j = i[false_mask], j[false_mask]

        mean_fake_distance = self._compute_distances(target_embeddings[i], pred_embeddings[j]).mean()

        return mean_true_distance, mean_fake_distance

    def _plot_evaluation(
        self, x, context_mask, target_mask, distances, embedded_context_tracklets, embedded_target_tracklets
    ):
        """
        Generates and displays plots for evaluation.

        Args:
            x (torch.Tensor): Input tensor.
            context_mask (torch.Tensor): Context mask.
            target_mask (torch.Tensor): Target mask.
            distances (torch.Tensor): Distances between embeddings.
            embedded_context_tracklets (torch.Tensor): Embedded context tracklets.
            embedded_target_tracklets (torch.Tensor): Embedded target tracklets.
        """
        # Move to CPU and select first event
        x_cpu = x[0].cpu()
        context_mask_cpu = context_mask[0].cpu()
        target_mask_cpu = target_mask[0].cpu()
        distances_cpu = distances.cpu()
        embedded_context_tracklets_cpu = embedded_context_tracklets.detach().cpu()
        embedded_target_tracklets_cpu = embedded_target_tracklets.detach().cpu()

        # Plot original tracklets
        self._plot_original_tracklets(x_cpu, context_mask_cpu, target_mask_cpu)

        # Combine context and target tracklets
        combined_tracklets = torch.cat([embedded_context_tracklets_cpu, embedded_target_tracklets_cpu], dim=0)

        # Plot embedded tracklets with PCA
        pca = PCA(n_components=2)
        combined_tracklets_2d = pca.fit_transform(combined_tracklets.numpy())
        embedded_context_tracklets_2d = combined_tracklets_2d[:len(embedded_context_tracklets_cpu)]
        embedded_target_tracklets_2d = combined_tracklets_2d[len(embedded_context_tracklets_cpu):]

        self._plot_embedded_tracklets(
            embedded_context_tracklets_2d, embedded_target_tracklets_2d, distances_cpu
        )

    def _plot_original_tracklets(self, x_cpu, context_mask_cpu, target_mask_cpu):
        """
        Plots the original tracklets for the first event. All hits are scattered, along
        with the colored context and target tracklets.

        Args:
            x_cpu (torch.Tensor): Input tensor for the first event.
            context_mask_cpu (torch.Tensor): Context mask for the first event.
            target_mask_cpu (torch.Tensor): Target mask for the first event.
        """
        fig, ax = plt.subplots()

        # Scatter all hits
        ax.scatter(x_cpu[:, 0], x_cpu[:, 1], label='All Hits', color='gray', alpha=0.5)

        # Scatter context tracklets
        ax.scatter(x_cpu[context_mask_cpu, 0], x_cpu[context_mask_cpu, 1], label='Context Tracklets', color='blue')
        
        # Scatter target tracklets
        ax.scatter(x_cpu[target_mask_cpu, 0], x_cpu[target_mask_cpu, 1], label='Target Tracklets', color='red')
        
        ax.legend()
        ax.set_title('Original Tracklets (First Event)')
        ax.set_aspect('equal', 'box')
        plt.show()

    def _plot_embedded_tracklets(
        self, embedded_context_tracklets_2d, embedded_target_tracklets_2d, distances_cpu
    ):
        """
        Plots the embedded tracklets using 2D PCA along with evaluation edges and distances.

        Args:
            embedded_context_tracklets_2d (ndarray): 2D PCA transformed context tracklets.
            embedded_target_tracklets_2d (ndarray): 2D PCA transformed target tracklets.
            distances_cpu (torch.Tensor): Distances between embeddings.
        """
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size to accommodate legend
        
        # Create a rainbow color map with discrete entries
        num_colors = len(distances_cpu)
        cmap = plt.cm.rainbow
        colors = cmap(np.linspace(0, 1, num_colors))

        # Plot context tracklets as circles
        ax.scatter(embedded_context_tracklets_2d[:, 0], embedded_context_tracklets_2d[:, 1], 
                   label='Context Tracklets', color=colors, marker='o')
        
        # Plot target tracklets as stars
        ax.scatter(embedded_target_tracklets_2d[:, 0], embedded_target_tracklets_2d[:, 1], 
                   label='Target Tracklets', color=colors, marker='*')  # Increased size for better visibility

        # Move legend outside the plot
        ax.set_title('Embedded Tracklets Context and Target (2D PCA)')
        plt.tight_layout()  # Adjust the layout to prevent clipping of the legend
        plt.show()

    def _print_metrics(self, mean_true_distance, mean_fake_distance):
        """
        Prints the evaluation metrics for the first event.

        Args:
            efficiency (torch.Tensor): Computed efficiency.
            purity (torch.Tensor): Computed purity.
            mean_true_distance (torch.Tensor): Mean distance for true pairs.
            mean_fake_distance (torch.Tensor): Mean distance for fake pairs.
        """
        print(f"  Mean True Distance: {mean_true_distance:.4f}")
        print(f"  Mean Fake Distance: {mean_fake_distance:.4f}")

    def _check_nan(self, tensor, tensor_name, batch):
        """
        Check if a tensor contains NaN values and print debugging information if it does.

        Args:
            tensor (torch.Tensor): The tensor to check for NaN values.
            tensor_name (str): A name to identify the tensor in debug messages.
        """
        if torch.isnan(tensor).any():
            print(f"NaN detected in {tensor_name}")
            print(f"Tensor shape: {tensor.shape}")
            nan_indices = torch.isnan(tensor).nonzero()
            print(f"NaN locations: {nan_indices}")
            print(f"Tensor statistics:")
            print(f"  Min: {tensor.min()}")
            print(f"  Max: {tensor.max()}")
            print(f"  Mean: {tensor.mean()}")
            print(f"  Std: {tensor.std()}")
            
            # Print the batch entries containing NaN values
            if len(tensor.shape) > 1:  # If tensor has more than 1 dimension
                batch_indices = nan_indices[:, 0].unique()
                for idx in batch_indices:
                    print(f"\nBatch entry {idx} containing NaN:")
                    print(tensor[idx])
                    x, mask, context_mask, target_mask, pids = self._extract_batch_data(batch)
                    print(f"x: {x[idx]}")
                    print(f"mask: {mask[idx]}")
                    print(f"context_mask: {context_mask[idx]}")
                    print(f"target_mask: {target_mask[idx]}")
                    print(f"pids: {pids[idx]}")

            else:
                print("\nTensor containing NaN:")
                print(tensor)
            
            raise ValueError(f"NaN detected in {tensor_name}")
