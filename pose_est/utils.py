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

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T