"""JAX-friendly per-episode robot and environment parameter contexts.

The legacy environments keep their constructor attributes and public return
signatures.  These helpers support the additive ``*_with_params`` API used by
policies that explicitly opt into robot/domain randomization.
"""

from collections.abc import Mapping

import jax.numpy as jnp
from jax import jit, random


ROBOT_PARAM_KEYS = (
    "radius",
    "v_max",
    "wheels_distance",
    "control_dt",
    "wheel_accel_max",
    "tau_linear",
    "tau_angular",
    "control_delay_mean",
    "control_delay_std",
    "actuation_gain",
    "slip_scale",
)

ENV_PARAM_KEYS = (
    "lidar_period",
    "odometry_period",
    "lidar_latency",
    "odometry_latency",
    "lidar_noise_fixed",
    "lidar_noise_proportional",
    "lidar_dropout_probability",
    "lidar_range_scale",
    "obstacle_noise",
    "robot_visibility_probability",
    "human_speed_scale",
    "human_radius_scale",
)


def _as_scalar_dict(values, required_keys, name):
    missing = set(required_keys) - set(values)
    extra = set(values) - set(required_keys)
    if missing or extra:
        raise ValueError(f"Invalid {name} keys; missing={sorted(missing)}, extra={sorted(extra)}")
    return {key: jnp.asarray(values[key], dtype=jnp.float32) for key in required_keys}


def validate_robot_params(values):
    values = _as_scalar_dict(values, ROBOT_PARAM_KEYS, "robot_params")
    positive = ("radius", "v_max", "control_dt", "actuation_gain", "slip_scale")
    for key in positive:
        if float(values[key]) <= 0.0:
            raise ValueError(f"robot_params['{key}'] must be positive")
    for key in ("wheels_distance", "wheel_accel_max", "tau_linear", "tau_angular", "control_delay_mean", "control_delay_std"):
        if float(values[key]) < 0.0:
            raise ValueError(f"robot_params['{key}'] must be non-negative")
    return values


def validate_env_params(values):
    values = _as_scalar_dict(values, ENV_PARAM_KEYS, "env_params")
    for key in ("lidar_period", "odometry_period", "lidar_range_scale", "human_speed_scale", "human_radius_scale"):
        if float(values[key]) <= 0.0:
            raise ValueError(f"env_params['{key}'] must be positive")
    for key in ("lidar_latency", "odometry_latency", "lidar_noise_fixed", "lidar_noise_proportional", "obstacle_noise"):
        if float(values[key]) < 0.0:
            raise ValueError(f"env_params['{key}'] must be non-negative")
    for key in ("lidar_dropout_probability", "robot_visibility_probability"):
        if not 0.0 <= float(values[key]) <= 1.0:
            raise ValueError(f"env_params['{key}'] must be in [0, 1]")
    return values


def bounds_from_nominal(nominal, bounds=None):
    """Return fixed-structure lower/upper dictionaries for JIT sampling.

    Bounds may contain only the fields that should be randomized.  Every value
    is a ``(low, high)`` pair; unspecified fields remain at their nominal value.
    """
    bounds = {} if bounds is None else bounds
    unknown = set(bounds) - set(nominal)
    if unknown:
        raise ValueError(f"Unknown parameter bounds: {sorted(unknown)}")
    lower, upper = {}, {}
    for key, nominal_value in nominal.items():
        pair = bounds.get(key, (nominal_value, nominal_value))
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"Bounds for '{key}' must be a (low, high) pair")
        low, high = pair
        if float(low) > float(high):
            raise ValueError(f"Bounds for '{key}' have low > high")
        lower[key] = jnp.asarray(low, dtype=jnp.float32)
        upper[key] = jnp.asarray(high, dtype=jnp.float32)
    return lower, upper


@jit
def sample_context(key, nominal, lower, upper):
    """Sample a fixed-key scalar context, safely retaining non-finite constants."""
    keys = random.split(key, len(nominal))
    sampled = {}
    for subkey, name in zip(keys, nominal):
        finite_bounds = jnp.isfinite(lower[name]) & jnp.isfinite(upper[name])
        safe_low = jnp.where(finite_bounds, lower[name], nominal[name])
        safe_high = jnp.where(finite_bounds, upper[name], nominal[name])
        draw = random.uniform(subkey, (), minval=safe_low, maxval=safe_high)
        sampled[name] = jnp.where(finite_bounds, draw, nominal[name])
    return sampled


def context_from_mapping(values, validator):
    if not isinstance(values, Mapping):
        raise TypeError("Parameter contexts must be mappings")
    return validator(values)
