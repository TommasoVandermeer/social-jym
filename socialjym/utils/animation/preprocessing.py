"""
Quantities needed by the animations, computed ONCE for all the frames of an episode (JAX vmap where useful) and moved to numpy.
The animation loop then only indexes numpy arrays: no JAX dispatch, no recompilation and no per-element Python math at each frame.
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import tree_map

from socialjym.envs.base_env import HUMAN_POLICIES
from jhsfm.hsfm import get_linear_velocity

def to_numpy(tree):
    """Move a pytree of (JAX) arrays to host memory as numpy arrays (None leaves are kept)."""
    return tree_map(np.asarray, tree)

def states_to_scene(states, humans_policy:int):
    """
    Split env states (n_steps, n_humans+1, 6) in robot poses (n_steps, 3), humans poses (n_steps, n_humans, 3)
    and humans velocities in the global frame (n_steps, n_humans, 2).
    """
    robot_positions = states[:,-1,:2]
    robot_orientations = states[:,-1,4]
    robot_poses = jnp.hstack((robot_positions, robot_orientations.reshape(-1,1)))
    humans_positions = states[:,:-1,:2]
    humans_orientations = states[:,:-1,4]
    humans_poses = jnp.dstack((humans_positions, humans_orientations))
    humans_body_velocities = states[:,:-1,2:4]
    if humans_policy == HUMAN_POLICIES.index('hsfm'):
        humans_velocities = vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(humans_orientations, humans_body_velocities)
    else:
        humans_velocities = humans_body_velocities
    return robot_poses, humans_poses, humans_velocities

def heading_points(poses:np.ndarray, radii) -> np.ndarray:
    """Center of the small black disk marking the heading of disks with poses (..., 3) [x, y, theta] and radii (...)."""
    radii = np.asarray(radii)[...,None]
    return poses[...,:2] + radii * np.stack((np.cos(poses[...,2]), np.sin(poses[...,2])), axis=-1)

def robot_to_world(points:np.ndarray, robot_poses:np.ndarray, translate:bool=True) -> np.ndarray:
    """Roto-translate points (n_steps, n_points, 2) from the robot frame to the world frame (only rotate vectors if translate=False)."""
    c = np.cos(robot_poses[:,2])[:,None]
    s = np.sin(robot_poses[:,2])[:,None]
    world_points = np.stack((
        points[...,0] * c - points[...,1] * s,
        points[...,0] * s + points[...,1] * c,
    ), axis=-1)
    if translate:
        world_points = world_points + robot_poses[:,None,:2]
    return world_points

def lidar_ray_segments(robot_poses:np.ndarray, lidar_scans:np.ndarray, lidar_angles_robot_frame:np.ndarray) -> np.ndarray:
    """Segments (n_steps, n_rays, 2, 2) going from the robot position to each LiDAR reading."""
    angles = robot_poses[:,2:3] + np.asarray(lidar_angles_robot_frame)[None,:]
    origins = np.broadcast_to(robot_poses[:,None,:2], angles.shape + (2,))
    ends = origins + lidar_scans[...,None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    return np.stack((origins, ends), axis=2)

def gaussian_ellipses(distributions:dict, biv_gaussian) -> tuple:
    """
    1-sigma ellipses of a batch of bivariate Gaussians (dict with 'means', 'logsigmas', 'correlation' with any leading shape).
    Returns centers (..., 2), widths (...), heights (...) and angles in degrees (...), as drawn by the old animate methods.
    """
    lead_shape = distributions['means'].shape[:-1]
    flat_distrs = {k: jnp.reshape(distributions[k], (-1,) + distributions[k].shape[len(lead_shape):]) for k in ['means', 'logsigmas', 'correlation']}
    covariances = vmap(biv_gaussian.covariance)(flat_distrs)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariances)
    widths = 2 * jnp.sqrt(eigenvalues[:,0])
    heights = 2 * jnp.sqrt(eigenvalues[:,1])
    angles = jnp.degrees(jnp.arctan2(eigenvectors[:,1,0], eigenvectors[:,0,0]))
    return (
        np.asarray(distributions['means']),
        np.asarray(widths).reshape(lead_shape),
        np.asarray(heights).reshape(lead_shape),
        np.asarray(angles).reshape(lead_shape),
    )

def roto_translate_gaussians(distributions:dict, reference_poses, biv_gaussian) -> dict:
    """Roto-translate bivariate Gaussians (n_steps, n_humans) from the frame of reference_poses (n_steps, 3) to the world frame."""
    distributions = {k: distributions[k] for k in ['means', 'logsigmas', 'correlation']}
    return vmap(vmap(biv_gaussian.roto_translate, in_axes=(0, None)), in_axes=(0, 0))(distributions, jnp.asarray(reference_poses))

def action_distribution_grid(distribution, actor_distrs:dict, samples, threshold:float) -> tuple:
    """
    Density of a fixed set of action samples (n_samples, 2) under the action distribution of each frame (one single vmap,
    instead of one recompilation per frame due to boolean indexing). Returns the densities (n_steps, n_samples) and the mask
    of the samples to show (n_steps, n_samples): inside the distribution support and with density above threshold.
    """
    @jit
    def _frame_density(actor_distr):
        return distribution.batch_is_in_support(actor_distr, samples), distribution.batch_p(actor_distr, samples)
    in_support, p = vmap(_frame_density)(actor_distrs)
    p = np.asarray(p)
    with np.errstate(invalid='ignore'):
        mask = np.asarray(in_support) & (p > threshold)
    return p, mask

def static_humans_mask(humans_poses:np.ndarray) -> np.ndarray:
    """Humans that never move during the episode (e.g. columns in CCSO scenarios)."""
    return np.all(humans_poses[:1,:,:2] == humans_poses[:,:,:2], axis=(0,2))

def safe_nanargmin(values:np.ndarray, axis:int=-1) -> np.ndarray:
    """np.nanargmin that returns 0 instead of raising for all-NaN slices."""
    all_nan = np.all(np.isnan(values), axis=axis, keepdims=True)
    return np.nanargmin(np.where(all_nan, 0., values), axis=axis)
