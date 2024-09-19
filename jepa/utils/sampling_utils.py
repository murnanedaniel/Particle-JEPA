import torch
import numpy as np

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