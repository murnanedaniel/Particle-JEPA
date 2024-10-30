import torch
import numpy as np
from typing import Dict

class WedgePatchify:
    """
    A class to transform hitwise data into wedges of annuli.

    Give an event of shape [num_hits, 2] (x, y), return two mask tensors (context, target) of shape [num_hits]
    """

    def __init__(self, phi_range: float, radius_midpoint: float, random_context: bool = True):
        self.phi_range = phi_range
        self.radius_midpoint = radius_midpoint
        self.random_context = random_context

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply the WedgePatchify transform to the input sample.

        Args:
            sample (Dict): A dictionary containing hitwise data. Must include an 'x' key
                           with a tensor of shape (num_hits, 2).

        Returns:
            Dict: The transformed sample with context, target, and mask tensors.
        """
        x, y = self._extract_coordinates(sample)
        radius, phi = self._calculate_radius_and_phi(x, y)
        selected_phi = self._select_random_phi(phi)
        phi_mask = self._create_phi_mask(phi, selected_phi)
        inner_mask, outer_mask = self._create_radius_masks(radius)
        context_mask, target_mask = self._assign_masks(inner_mask, outer_mask, phi_mask)

        sample["context_mask"] = context_mask
        sample["target_mask"] = target_mask
        sample["pt"] = self._get_pt(sample)

        return sample

    def _extract_coordinates(self, sample: Dict) -> (torch.Tensor, torch.Tensor):
        x = sample["x"][:, 0]
        y = sample["x"][:, 1]
        return x, y

    def _calculate_radius_and_phi(self, x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        radius = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)  # Returns values between -pi and pi
        return radius, phi

    def _select_random_phi(self, phi: torch.Tensor) -> torch.Tensor:
        hit_idx = torch.randint(0, phi.shape[0], (1,))
        return phi[hit_idx]

    def _create_phi_mask(self, phi: torch.Tensor, selected_phi: torch.Tensor) -> torch.Tensor:
        phi_min = selected_phi - self.phi_range / 2
        phi_max = selected_phi + self.phi_range / 2
        return torch.logical_or(
            torch.logical_and(phi >= phi_min, phi <= phi_max),
            torch.logical_and(phi + 2*torch.pi >= phi_min, phi + 2*torch.pi <= phi_max)
        )

    def _create_radius_masks(self, radius: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        inner_mask = radius <= self.radius_midpoint
        outer_mask = radius > self.radius_midpoint
        return inner_mask, outer_mask

    def _assign_masks(self, inner_mask: torch.Tensor, outer_mask: torch.Tensor, phi_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if self.random_context and torch.rand(1).item() > 0.5:
            context_mask = inner_mask & phi_mask
            target_mask = outer_mask & phi_mask
        else:
            context_mask = outer_mask & phi_mask
            target_mask = inner_mask & phi_mask
        return context_mask, target_mask

    def _get_pt(self, sample: Dict) -> torch.Tensor:
        """
        Get the pT of the particle associated with the hit.
        """
        hits, particles = sample["event"].hits, sample["event"].particles
        hits = hits.merge(particles, on="particle_id")
        particle_pT = torch.tensor(hits["pt"], dtype=torch.float)
        return particle_pT


class QualityCut:
    """
    A transform class to cut on the quality of the event and define a 
    high/low pT truth.
    """

    def __init__(self, min_hits: int = 1, pt_threshold: float = 5.0):
        self.min_hits = min_hits
        self.pt_threshold = pt_threshold

    def __call__(self, sample: Dict) -> Dict:
        """
        Define a binary tensor y, which is true if:
            - The context masked wedge has at least one hit associated with a particle with pT > pt_threshold
            - There are greater than min_hits of that particle in the context masked wedge
        """
        # Extract the context mask and particle pT from the sample
        context_mask = sample["context_mask"]
        particle_pT = sample["pt"]

        # Apply the context mask to filter hits
        masked_pT = particle_pT[context_mask]
        particle_ids = torch.tensor(sample["event"].hits["particle_id"], dtype=torch.long)[context_mask]

        # Identify high pT particles
        high_pt_particle_ids = particle_ids[masked_pT > self.pt_threshold]

        # Count the number of hits for each high pT particle
        unique_high_pt_ids, counts = torch.unique(high_pt_particle_ids, return_counts=True)

        # Check if any high pT particle has more than min_hits
        sufficient_hits_mask = counts > self.min_hits

        # Determine if the event passes the quality cut
        y = torch.any(sufficient_hits_mask)

        # Add the result to the sample
        sample["y"] = y

        return sample


def fit_circle(x, y):
    # Assemble the A matrix
    A = np.vstack([x, y, np.ones(len(x))]).T
    # Assemble the f matrix
    f = x**2 + y**2
    # Solve the least squares problem
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    # Extract circle parameters
    cx, cy = C[0]/2, C[1]/2
    radius = np.sqrt(C[2] + cx**2 + cy**2)
    return cx, cy, radius

def random_rphi_sample(r, phi, min_radius, max_radius):

    rlim = (max_radius - min_radius) * torch.rand((2, r.shape[0])) + min_radius
    rlim[:, rlim[0] > rlim[1]] = rlim[:, rlim[0] > rlim[1]].flip(0)
    rlim = rlim[:, :, None]

    philim = 2 * torch.rand((2, r.shape[0])) * torch.pi - torch.pi
    philim = philim[:, :, None]
    phiorder = philim[0] < philim[1]

    target_mask = (r < rlim[1]) & (r > rlim[0]) & (
        (phiorder & (phi > philim[0]) & (phi < philim[1]))
        | ((~phiorder) & (phi < philim[0]) & (phi > philim[1]))
    )

    return target_mask, rlim, philim

def track_split_sample(r, min_radius, max_radius, random=True):
    """
    For each event in batch, flip a coin to see whether source or target context is innermost.
    For each pid, take the innermost r/2 hits and assign them to the context.
    """

    if random:
        batch_randoms = torch.rand((r.shape[0], ), device=r.device)
    else:
        batch_randoms = torch.ones((r.shape[0], ), device=r.device)

    mid_r_point = (min_radius + max_radius) / 2
    target_mask = r < mid_r_point

    # Where batch_randoms is less than 0.5, we want to flip the target mask
    inner_target = batch_randoms < 0.5
    target_mask = target_mask ^ inner_target.unsqueeze(-1).expand_as(target_mask)

    return target_mask, inner_target