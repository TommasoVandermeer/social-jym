"""ROS-independent helpers for deploying JESSI-S2R checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from jax import tree_util


def load_jessi_s2r_parameters(path: str | Path, selection: str = "best"):
    """Load deployable E2E parameters from rollout, snapshot, or checkpoint files."""
    if selection not in ("best", "final"):
        raise ValueError("selection must be 'best' or 'final'")
    path = Path(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)

    if isinstance(payload, tuple) and len(payload) >= 5:
        return payload[0 if selection == "best" else 1], f"rollout:{selection}"
    if isinstance(payload, dict) and "params" in payload:
        return payload["params"], "snapshot:params"
    if isinstance(payload, dict) and "state" in payload:
        state = payload["state"]
        key = "best_params" if selection == "best" else "params"
        if key not in state:
            raise ValueError(f"Checkpoint does not contain '{key}'")
        return state[key], f"checkpoint:{key}"
    raise ValueError(
        "Unsupported JESSI-S2R file. Expected a five-element rollout, "
        "a {'params': ...} snapshot, or a training checkpoint."
    )


def validate_parameter_shapes(parameters, expected_parameters) -> None:
    """Fail early when controller architecture and saved weights do not match."""
    actual_structure = tree_util.tree_structure(parameters)
    expected_structure = tree_util.tree_structure(expected_parameters)
    if actual_structure != expected_structure:
        raise ValueError("JESSI-S2R parameter tree does not match the controller policy")
    mismatches = []
    for index, (actual, expected) in enumerate(zip(
        tree_util.tree_leaves(parameters), tree_util.tree_leaves(expected_parameters)
    )):
        if np.shape(actual) != np.shape(expected):
            mismatches.append((index, np.shape(actual), np.shape(expected)))
    if mismatches:
        raise ValueError(f"JESSI-S2R parameter shape mismatch: {mismatches[:5]}")


def relative_sensor_timing(observation_stack, control_time: float) -> np.ndarray:
    """Preserve sub-second sensor timing before conversion to JAX float32."""
    observations = np.asarray(observation_stack, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] < 11:
        raise ValueError("LaserNav observation stack must have shape (n_stack, >=11)")
    timing = observations[:, 8:11] - float(control_time)
    if not np.all(np.isfinite(timing)):
        raise ValueError("Non-finite sensor timestamps in observation stack")
    return timing.astype(np.float32)
