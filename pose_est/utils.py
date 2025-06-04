from os.path import join, dirname
import numpy as np
from typing import Optional


def get_exp_config_from_checkpoint(checkpoint_path: str) -> str:
    """
    Get the experiment configuration file path from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint file.

    Returns
    -------
    str
        The path to the experiment configuration file.
    """
    return join(dirname(dirname(checkpoint_path)), "config.yaml")

def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for point cloud downsampling.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3).
    num_samples : int
        Number of points to sample.

    Returns
    -------
    np.ndarray
        Indices of the sampled points, shape (num_samples,).
    """
    n = points.shape[0]
    sampled_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf)
    
    # Start with a random point
    sampled_indices[0] = np.random.randint(n)
    
    for i in range(1, num_samples):
        # Compute distances to the last sampled point
        last_point = points[sampled_indices[i-1]]
        dist_to_last = np.sum((points - last_point) ** 2, axis=1)
        
        # Update distances (keep the minimum distance to any sampled point)
        distances = np.minimum(distances, dist_to_last)
        
        # Select the point with the maximum distance
        sampled_indices[i] = np.argmax(distances)
    
    return sampled_indices