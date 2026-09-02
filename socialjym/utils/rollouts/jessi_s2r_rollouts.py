import optax
import glob
import hashlib
import os
import pickle
import tempfile
from jax import jit, lax, random, vmap, device_put, device_get, device_count, eval_shape, ShapeDtypeStruct, debug
from jax.tree_util import tree_map, tree_leaves
import jax.numpy as jnp
from jax import nn
import numpy as np
from tqdm import tqdm
from functools import partial
from jax import value_and_grad
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

from jhsfm.hsfm import get_linear_velocity
from socialjym.envs.base_env import ROBOT_KINEMATICS, SCENARIOS
from socialjym.envs.lasernav import LaserNav
from socialjym.envs.parameter_context import (
    bounds_from_nominal,
    validate_env_params,
    validate_robot_params,
)
from socialjym.policies.jessi_s2r import JESSI_S2R


TRAINING_TYPES = ["multitask", "modular", "policy"]
CHECKPOINT_SCHEMA_VERSION = 2
LEGACY_CHECKPOINT_SCHEMA_VERSIONS = (1,)
SOCIAL_SCENARIOS = (0, 1, 2, 3, 4, 6, 9)
NAVIGATION_SCENARIOS = (10, 11, 12, 13, 14, 15, 16)

# Exactly one source of difficulty changes between adjacent stages.
CURRICULUM_STAGES = tuple(
    [(round(0.1 * level, 1), 1.0) for level in range(6)]
    + [
        (0.5, 0.9), (0.6, 0.9), (0.6, 0.8), (0.7, 0.8),
        (0.7, 0.7), (0.8, 0.7), (0.8, 0.6), (0.9, 0.6),
        (0.9, 0.5), (1.0, 0.5),
    ]
    + [(1.0, round(value, 1)) for value in (0.4, 0.3, 0.2, 0.1, 0.0)]
)


def _host_tree(tree):
    # ``device_get`` may return a zero-copy NumPy view backed by a JAX buffer.
    # PPO donates several of those buffers to compiled updates, so a checkpoint
    # assembled before a later donation can otherwise contain deleted arrays.
    # Materialise an owning host copy at the checkpoint boundary.
    def copy_leaf(leaf):
        if isinstance(leaf, (str, bytes)):
            return leaf
        return np.array(device_get(leaf), copy=True)
    return tree_map(copy_leaf, tree)


def _checkpoint_fingerprint(config):
    return hashlib.sha256(pickle.dumps(config, protocol=5)).hexdigest()


def _plain_value(value):
    if isinstance(value, dict):
        return {str(key): _plain_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    array = np.asarray(device_get(value))
    return array.item() if array.ndim == 0 else array.tolist()


def _atomic_pickle(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=".checkpoint-", delete=False
        ) as temporary:
            temporary_path = temporary.name
            pickle.dump(payload, temporary, protocol=5)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_training_checkpoint(checkpoint_dir, next_update, state, config, keep=3):
    """Atomically save a full update-boundary checkpoint."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config,
        "fingerprint": _checkpoint_fingerprint(config),
        "next_update": int(next_update),
        "state": _host_tree(state),
    }
    path = os.path.join(checkpoint_dir, f"update_{next_update:06d}.pkl")
    _atomic_pickle(path, payload)
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "update_*.pkl")))
    for stale_path in checkpoints[:-max(int(keep), 1)]:
        os.unlink(stale_path)
    return path


def load_training_checkpoint(path, expected_config):
    with open(path, "rb") as checkpoint_file:
        payload = pickle.load(checkpoint_file)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {payload.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    expected_fingerprint = _checkpoint_fingerprint(expected_config)
    if payload.get("fingerprint") != expected_fingerprint:
        actual = payload.get("config", {})
        differences = {
            key: (actual.get(key), expected_config.get(key))
            for key in sorted(set(actual) | set(expected_config))
            if actual.get(key) != expected_config.get(key)
        }
        raise ValueError(f"Checkpoint configuration mismatch: {differences}")
    return payload


def load_warm_start_candidates(path):
    """Load actor/critic candidates without restoring legacy optimizer state."""
    with open(path, "rb") as checkpoint_file:
        payload = pickle.load(checkpoint_file)
    if isinstance(payload, dict) and "schema_version" in payload:
        schema = payload.get("schema_version")
        if schema not in (*LEGACY_CHECKPOINT_SCHEMA_VERSIONS, CHECKPOINT_SCHEMA_VERSION):
            raise ValueError(f"Unsupported warm-start checkpoint schema {schema}.")
        state = payload["state"]
        return {
            "final": (state["params"], state["critic_params"]),
            "best": (state.get("best_params", state["params"]), state.get("best_critic_params", state["critic_params"])),
        }
    if isinstance(payload, tuple) and len(payload) >= 4:
        return {
            "best": (payload[0], payload[2]),
            "final": (payload[1], payload[3]),
        }
    raise ValueError("Warm-start file is neither a training checkpoint nor an RL output tuple.")


def atomic_pickle(path, payload):
    """Public wrapper used by training scripts for crash-safe result files."""
    _atomic_pickle(path, _host_tree(payload))


def prepare_numeric_metrics(metrics):
    """Convert plottable logs while preserving structured evaluation records."""
    processed = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            processed[key] = (
                value
                if value and isinstance(value[0], dict)
                else jnp.asarray(value)
            )
        elif isinstance(value, dict):
            def convert_leaf(leaf):
                if isinstance(leaf, (str, bytes)):
                    return leaf
                try:
                    return jnp.asarray(leaf)
                except (TypeError, ValueError):
                    return leaf
            processed[key] = tree_map(convert_leaf, value)
        else:
            processed[key] = value
    return processed


def tree_all_finite(tree):
    """Return a scalar JAX boolean indicating whether every leaf is finite."""
    leaves = tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def tree_select(predicate, true_tree, false_tree):
    """Select between matching pytrees using a scalar JAX predicate."""
    return tree_map(lambda new, old: jnp.where(predicate, new, old), true_tree, false_tree)


def social_mask_from_scenarios(scenarios):
    scenarios = jnp.asarray(scenarios)
    return jnp.any(
        scenarios[..., None] == jnp.asarray(SOCIAL_SCENARIOS), axis=-1
    )


def group_normalize_advantages(advantages, scenarios):
    """Normalize social/navigation advantages independently and safely."""
    social_mask = social_mask_from_scenarios(scenarios)

    def normalize(values, mask):
        mask_f = mask.astype(values.dtype)
        count = jnp.sum(mask_f)
        mean = jnp.sum(values * mask_f) / jnp.maximum(count, 1.0)
        variance = jnp.sum(jnp.square(values - mean) * mask_f) / jnp.maximum(count, 1.0)
        normalized = (values - mean) / jnp.sqrt(variance + 1e-8)
        return jnp.where(mask & (count >= 2), normalized, values)

    return jnp.where(
        social_mask,
        normalize(advantages, social_mask),
        normalize(advantages, ~social_mask),
    )


def group_weighted_mean(values, scenarios, social_weight):
    """Combine group means without NaNs when a minibatch lacks one group."""
    social_mask = social_mask_from_scenarios(scenarios)
    social_count = jnp.sum(social_mask)
    navigation_count = jnp.sum(~social_mask)
    social_mean = jnp.sum(jnp.where(social_mask, values, 0.0)) / jnp.maximum(social_count, 1)
    navigation_mean = jnp.sum(jnp.where(~social_mask, values, 0.0)) / jnp.maximum(navigation_count, 1)
    social_available = (social_count > 0).astype(values.dtype)
    navigation_available = (navigation_count > 0).astype(values.dtype)
    social_budget = social_weight * social_available
    navigation_budget = (1.0 - social_weight) * navigation_available
    total_budget = jnp.maximum(social_budget + navigation_budget, 1e-8)
    return (
        social_budget * social_mean + navigation_budget * navigation_mean
    ) / total_budget

@partial(jit, static_argnames=("policy", "env", "n_steps"))
def collect_rollout_step(
    network_params, 
    critic_network_params,
    env_state, 
    policy_keys, 
    reset_keys, 
    env_keys,
    template_outcomes,
    policy:JESSI_S2R, 
    env:LaserNav, 
    n_steps,
    scenarios_prob,
    visibility,
    robot_lower,
    robot_upper,
    env_lower,
    env_upper,
):
    def _scan_step(carry, _):
        (states, obses, infos, outcomes, returns, times, success_per_scenario, episodes_per_scenario, p_keys, r_keys, e_keys, outcomes_acc) = carry
        keys = vmap(random.split)(p_keys)
        p_keys, c_keys = keys[:,0], keys[:,1]
        # Actor
        runtime_robot_params = infos["_robot_params"]
        runtime_env_params = infos["_env_params"]
        actions, new_p_keys, inputs0, inputs1, _, sampled_actions, _, actor_distrs, _, _, _, _ = policy.batch_act_with_params(
            p_keys,
            obses,
            infos,
            runtime_robot_params,
            network_params,
            sample=True,
        )
        # Critic
        env_params = {
            "humans_goal": infos["humans_goal"],
            "humans_parameters": infos["humans_parameters"],
            "static_obstacles": infos["static_obstacles"],
        }
        robot_params = {
            "robot_goal": infos["robot_goal"],
            "robot_radius": runtime_robot_params["radius"],
            "v_max": runtime_robot_params["v_max"],
            "wheels_distance": runtime_robot_params["wheels_distance"],
            "wheels_max_linear_acceleration": runtime_robot_params["wheel_accel_max"],
            "robot_delay": infos["robot_delay"],
        }
        values = policy.batch_critic_forward(
            c_keys,
            critic_network_params,
            states,
            obses[:, :policy.n_actions_history,6:8],
            env_params, # Environment params
            robot_params, # Robot params
            inputs1[:,0,:3], # Action space parameters
        )
        # Environment
        (
            new_states,
            new_obses,
            new_infos,
            _,
            _,
            (rewards, _),
            new_outcomes,
            (new_r_keys, new_e_keys),
        ) = env.batch_step_with_param_bounds(
            states,
            infos,
            runtime_robot_params,
            runtime_env_params,
            actions,
            r_keys,
            e_keys,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            test=False,
            reset_if_done=True,
            scenarios_prob=scenarios_prob,
        )
        rc_humans_positions, _, rc_humans_velocities, rc_obstacles, _ = env.batch_robot_centric_transform(
            states[:,:-1,:2], 
            states[:,:-1,4], 
            vmap(vmap(get_linear_velocity))(states[:,:-1,4], states[:,:-1,2:4]),
            infos["static_obstacles"][:,-1], 
            states[:,-1,:2], 
            states[:,-1,4], 
            infos["robot_goal"],
        )
        human_radii = infos["humans_parameters"][:, :, 0]
        robot_radii = runtime_robot_params["radius"][:, None]
        human_clearances = jnp.linalg.norm(rc_humans_positions, axis=-1) - (
            human_radii + robot_radii
        )
        if env.kinematics == ROBOT_KINEMATICS.index("holonomic"):
            robot_heading = states[:, -1, 4]
            cos_heading = jnp.cos(robot_heading)
            sin_heading = jnp.sin(robot_heading)
            robot_velocity_rc = jnp.stack(
                (
                    cos_heading * states[:, -1, 2] + sin_heading * states[:, -1, 3],
                    -sin_heading * states[:, -1, 2] + cos_heading * states[:, -1, 3],
                ),
                axis=-1,
            )
        else:
            # In the unicycle body frame the robot velocity is (linear speed, 0).
            robot_velocity_rc = jnp.stack(
                (states[:, -1, 2], jnp.zeros_like(states[:, -1, 2])), axis=-1
            )
        relative_velocities = rc_humans_velocities - robot_velocity_rc[:, None, :]
        relative_speed_sq = jnp.sum(relative_velocities**2, axis=-1) + 1e-6
        raw_ttc = -jnp.sum(
            rc_humans_positions * relative_velocities, axis=-1
        ) / relative_speed_sq
        closest = (
            rc_humans_positions
            + jnp.clip(raw_ttc, 0.0, 2.5)[..., None] * relative_velocities
        )
        predicted_clearance = jnp.linalg.norm(closest, axis=-1) - (
            human_radii + robot_radii
        )
        ttc_violation = jnp.any(
            (raw_ttc > 0.0) & (raw_ttc < 2.5) & (predicted_clearance < 0.4),
            axis=-1,
        )
        front_human = jnp.any(
            (rc_humans_positions[..., 0] > 0.0)
            & (jnp.abs(jnp.arctan2(
                rc_humans_positions[..., 1], rc_humans_positions[..., 0]
            )) <= jnp.pi / 3.0)
            & (jnp.linalg.norm(rc_humans_positions, axis=-1) <= 2.5),
            axis=-1,
        )
        step_data = {
            # "obs": obses,
            # "robot_goal": infos["robot_goal"],
            "inputs0": inputs0,
            "inputs1": inputs1,
            "states": states,
            "actions_history": obses[:, :policy.n_actions_history,6:8],
            "env_params": env_params,
            "robot_params": robot_params,
            "gt_poses": rc_humans_positions,
            "gt_vels": rc_humans_velocities,
            "gt_mask": infos["humans_visibility_mask"],
            "values": values,
            "actions": sampled_actions,
            "rewards": rewards,
            "dones": ~(outcomes["nothing"]),
            "neglogpdfs": policy.action_distribution.batch_neglogp(actor_distrs, sampled_actions),
            "stds": policy.action_distribution.batch_std(actor_distrs),
            "scenario": infos["current_scenario"],
            "min_human_clearance": jnp.min(human_clearances, axis=-1),
            "ttc_violation": ttc_violation,
            "yielding_violation": front_human & (actions[:, 0] > 0.1),
            "completed_episode": ~new_outcomes["nothing"],
            "collision_with_human": new_outcomes["collision_with_human"],
            "collision_with_obstacle": new_outcomes["collision_with_obstacle"],
        }
        new_times = times + (new_outcomes["success"]) * (infos['time'] + policy.dt)
        new_returns = returns + (~new_outcomes["nothing"]) * (
            infos['return']
            + jnp.power(
                env.reward_function.gamma,
                (infos['step'] + 1) * policy.dt * runtime_robot_params["v_max"],
            ) * rewards
        )
        new_success_per_scenario = {k: success_per_scenario[k] + (new_outcomes["success"]) * (infos["current_scenario"] == k) for k in success_per_scenario}
        new_episodes_per_scenario = {k: episodes_per_scenario[k] + (~new_outcomes["nothing"]) * (infos["current_scenario"] == k) for k in episodes_per_scenario}
        new_outcomes_acc = {k: outcomes_acc[k] + new_outcomes[k] for k in new_outcomes}
        return (new_states, new_obses, new_infos, new_outcomes, new_returns, new_times, new_success_per_scenario, new_episodes_per_scenario, new_p_keys, new_r_keys, new_e_keys, new_outcomes_acc), step_data
    
    init_outcomes_acc = {
        k: jnp.zeros_like(template_outcomes[k], dtype=jnp.int32) 
        for k in template_outcomes
    }
    init_carry = (
        env_state[0], 
        env_state[1], 
        env_state[2], 
        env_state[3], 
        jnp.zeros_like(env_state[2]['return']), 
        jnp.zeros_like(env_state[2]['time']),
        {k: jnp.zeros_like(env_state[2]['return'], dtype=jnp.int32) for k in range(len(SCENARIOS[:-1]))},
        {k: jnp.zeros_like(env_state[2]['return'], dtype=jnp.int32) for k in range(len(SCENARIOS[:-1]))},
        policy_keys, 
        reset_keys, 
        env_keys, 
        init_outcomes_acc
    )
    final_carry, history = lax.scan(_scan_step, init_carry, None, length=n_steps)
    (final_states, final_obses, final_infos, final_outcomes, final_returns, final_times, final_success_per_scenario, final_episodes_per_scenario, final_p_keys, final_r_keys, final_e_keys, sum_outcomes) = final_carry
    next_env_state = (final_states, final_obses, final_infos, final_outcomes)
    return next_env_state, final_p_keys, final_r_keys, final_e_keys, history, sum_outcomes, final_returns, final_times, final_success_per_scenario, final_episodes_per_scenario

@partial(jit, static_argnames=("policy","env"))
def process_buffer_and_gae(
    critic_params, 
    critic_keys,
    last_states,
    last_obs, 
    last_info, 
    last_dones,
    history, 
    policy:JESSI_S2R, 
    env:LaserNav,
    gamma, 
    dt,
    vmax,
    lambda_gae
):
    """
    Calcola l'ultimo value, GAE, Returns e appiattisce il buffer.
    """
    _, robot_state_inputs = vmap(
        policy.compute_e2e_input_with_params,
        in_axes=(0, 0, 0, None),
    )(
        last_obs,
        last_info['robot_goal'],
        last_info['_robot_params'],
        None,
    )
    # _, _, _, _, _, last_values, _, _, _ = policy.e2e.apply(
    #     network_params, None, perception_inputs, robot_state_inputs
    # )
    ## Compute last values using the critic network
    last_values = policy.batch_critic_forward(
        critic_keys,
        critic_params,
        last_states,
        last_obs[:, :policy.n_actions_history,6:8],
        {
            "humans_goal": last_info["humans_goal"],
            "humans_parameters": last_info["humans_parameters"],
            "static_obstacles": last_info["static_obstacles"],
        }, # env_params
        {
            "robot_goal": last_info["robot_goal"],
            "robot_radius": last_info["_robot_params"]["radius"],
            "v_max": last_info["_robot_params"]["v_max"],
            "wheels_distance": last_info["_robot_params"]["wheels_distance"],
            "wheels_max_linear_acceleration": last_info["_robot_params"]["wheel_accel_max"],
            "robot_delay": last_info["robot_delay"],
        }, # robot_params
        robot_state_inputs[:,0,:3] # action_space_parameters
    )
    rewards = history["rewards"]
    values = history["values"]
    dones = history["dones"]
    values_ext = jnp.concatenate([values, last_values[None, :]], axis=0)
    dones_ext = jnp.concatenate([dones, last_dones[None, :]], axis=0)
    gamma_step = gamma ** (dt * history["robot_params"]["v_max"])
    def _gae_step(gae_carry, i):
        adv_next = gae_carry
        mask = 1.0 - dones_ext[i+1].astype(jnp.float32)
        delta = rewards[i] + gamma_step[i] * values_ext[i+1] * mask - values_ext[i]
        advantage = delta + gamma_step[i] * lambda_gae * adv_next * mask
        return advantage, advantage # Carry, Output
    n_steps = rewards.shape[0]
    _, advantages = lax.scan(_gae_step, jnp.zeros_like(values[0]), jnp.arange(n_steps)[::-1])
    advantages = advantages[::-1]
    critic_targets = advantages + values
    def flatten(tree):
        return tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), tree)
    flattened_buffer = {
        # "observations": flatten(history["obs"]),
        # "robot_goals": flatten(history["robot_goal"]),
        "inputs0": flatten(history["inputs0"]),
        "inputs1": flatten(history["inputs1"]),
        "states": flatten(history["states"]),
        "actions_history": flatten(history["actions_history"]),
        "env_params": flatten(history["env_params"]),
        "robot_params": flatten(history["robot_params"]),
        "gt_poses": flatten(history["gt_poses"]),
        "gt_vels": flatten(history["gt_vels"]),
        "gt_mask": flatten(history["gt_mask"]),
        "actions": flatten(history["actions"]),
        "values": flatten(history["values"]),
        "neglogpdfs": flatten(history["neglogpdfs"]),
        "critic_targets": flatten(critic_targets),
        "advantages": flatten(advantages),
        "scenario": flatten(history["scenario"]),
    }
    
    return flattened_buffer

@partial(
    jit,
    static_argnames=("policy", "optimizer", "critic_optimizer", "clip_range", "compute_safety_loss", "compute_risk_auxiliary_loss", "training_type"),
    donate_argnums=(1, 2, 3, 4)
)
def train_one_epoch(
    key,
    network_params,
    opt_state,
    critic_network_params,
    critic_opt_state,
    batched_buffer,
    policy:JESSI_S2R,
    optimizer,
    critic_optimizer,
    clip_range,
    beta_entropy,
    social_weight,
    compute_safety_loss,
    compute_risk_auxiliary_loss,
    training_type,
    debugging=False,
):
    multitask_training = (training_type == TRAINING_TYPES.index("multitask"))
    modular_training = (training_type == TRAINING_TYPES.index("modular"))
    policy_training = (training_type == TRAINING_TYPES.index("policy"))

    n_minibatches = batched_buffer["actions"].shape[0]
    n_micro_splits = batched_buffer["actions"].shape[1]

    def _batch_step(carry_inner, micro_batches): 
        params_inner, critic_params_inner, opt_st_inner, critic_opt_st_inner, batch_idx, batch_key = carry_inner 

        # Normalize advantages within task group so the easier navigation
        # distribution cannot set the scale for social updates.
        all_mb_advantages = micro_batches["advantages"]
        norm_advantages = group_normalize_advantages(
            all_mb_advantages, micro_batches["scenario"]
        )
        # We clip the normalized advantages to avoid too large policy updates
        micro_batches["advantages"] = micro_batches["advantages"].at[:].set(jnp.clip(norm_advantages, -5, 5))

        def micro_batch_loss_fn(p, u_mb, micro_batch_key):
            inputs0, inputs1 = u_mb["inputs0"], u_mb["inputs1"]
            # Lowe input precision to save memory
            if multitask_training or modular_training:
                inputs0_f16 = inputs0.astype(jnp.bfloat16)
                inputs1_f16 = inputs1.astype(jnp.bfloat16)
                # Actor forward pass 
                (safety_perc_dist, _, _, actor_dist, _, __build_class__, _, _, _) = policy.e2e.apply(
                    p, None, inputs0_f16, inputs1_f16, stop_perception_gradient=not multitask_training
                )
            else:
                # Actor forward pass
                (safety_perc_dist, _, _, actor_dist, _, _, _, _, _) = policy.e2e.apply(
                    p, None, inputs0, inputs1, stop_perception_gradient=not multitask_training
                )               
            # Cast back to higher precision for loss computation
            if multitask_training or modular_training:
                def dist_to_f32(dist):
                    return tree_map(lambda x: x.astype(jnp.float32), dist)
                actor_dist = dist_to_f32(actor_dist)  
                safety_perc_dist = dist_to_f32(safety_perc_dist)
            # Actor
            new_neglogp = policy.action_distribution.batch_neglogp(actor_dist, u_mb["actions"])
            log_ratio = jnp.clip(u_mb["neglogpdfs"] - new_neglogp, -20.0, 20.0)
            ratio = jnp.exp(log_ratio)
            lax.cond(
                debugging & (batch_idx == 0),
                lambda : debug.print("Mean Ratio is: {m} - Std Ratio is: {s}", m=jnp.mean(ratio), s=jnp.std(ratio)),
                lambda : None,
            )
            surr1 = ratio * u_mb["advantages"]
            surr2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * u_mb["advantages"]
            surrogate = jnp.minimum(surr1, surr2)
            actor_loss = -group_weighted_mean(
                surrogate, u_mb["scenario"], social_weight
            )
            social_actor_loss = -group_weighted_mean(
                surrogate, u_mb["scenario"], 1.0
            )
            navigation_actor_loss = -group_weighted_mean(
                surrogate, u_mb["scenario"], 0.0
            )
            approx_kl = group_weighted_mean(
                (ratio - 1) - log_ratio, u_mb["scenario"], social_weight
            )
            clip_frac = group_weighted_mean(
                (jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32),
                u_mb["scenario"], social_weight,
            )
            # Entropy
            locs_entropy = group_weighted_mean(
                policy.action_distribution.batch_entropy(actor_dist),
                u_mb["scenario"], social_weight,
            )
            weight_entropy = group_weighted_mean(
                policy.action_distribution.batch_weight_entropy(actor_dist),
                u_mb["scenario"], social_weight,
            )
            entropy_loss = -beta_entropy * (locs_entropy + 10 * weight_entropy)
            max_weight = group_weighted_mean(
                jnp.max(nn.softmax(actor_dist["locs"]), axis=-1),
                u_mb["scenario"], social_weight,
            )
            policy_loss = actor_loss + entropy_loss
            # Perception
            if multitask_training or modular_training:
                gt_dict = {"gt_mask": u_mb["gt_mask"], "gt_poses": u_mb["gt_poses"], "gt_vels": u_mb["gt_vels"]}
                augment_key, mask_drop_key = random.split(micro_batch_key)
                # Data augmentation (random rotations, angular mask dropout) for regularization
                def augment_data(inputs0, gt_dict, key):
                    # Input shape is (B, n_stack, num_beams, 7) and gt_poses/gt_vels shape is (B, n_stack, num_beams, 2)
                    # Random rotation
                    alpha = random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
                    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
                    rot_mat = jnp.array([[ca, -sa], [sa, ca]])
                    s_new = inputs0[..., 4] * ca + inputs0[..., 5] * sa
                    c_new = inputs0[..., 5] * ca - inputs0[..., 4] * sa
                    xy_rotated = inputs0[..., 2:4]  @ rot_mat.T
                    inputs0 = inputs0.at[..., 2:4].set(xy_rotated)
                    inputs0 = inputs0.at[..., 4].set(s_new) 
                    inputs0 = inputs0.at[..., 5].set(c_new) 
                    gt_dict['gt_poses'] = gt_dict['gt_poses'] @ rot_mat.T
                    gt_dict['gt_vels'] = gt_dict['gt_vels'] @ rot_mat.T
                    return inputs0, gt_dict
                inputs0_corrupt, gt_dict = augment_data(inputs0, gt_dict, augment_key)
                # Cast corrupted input to float16 to save memory during forward pass
                inputs0_corrupt_f16 = inputs0_corrupt.astype(jnp.bfloat16)
                # Forward pass through perception head
                (perc_dist, _, _, _, _, _, _, _, _) = policy.e2e.apply(
                    p, None, inputs0_corrupt_f16, inputs1, stop_perception_gradient=False, only_perception=True, perception_key=mask_drop_key
                )
                # Compute perception loss
                perc_dist = dist_to_f32(perc_dist)
                batch_perc_loss, _ = policy._perception_loss(perc_dist, gt_dict)
                perception_loss = jnp.mean(batch_perc_loss)
            else:
                perception_loss = 0.0
            # Safety loss (optional)
            if compute_risk_auxiliary_loss:
                safety_loss = 0.25 * policy._risk_auxiliary_loss(
                    actor_dist,
                    safety_perc_dist,
                )
            elif compute_safety_loss:
                safety_loss = policy._safety_loss(
                    actor_dist,
                    safety_perc_dist,
                )
            else:
                safety_loss = 0.0
            # Total loss
            total_loss = policy_loss + .05 * perception_loss + safety_loss
            return total_loss, (actor_loss, perception_loss, safety_loss, entropy_loss, weight_entropy, max_weight, approx_kl, clip_frac, social_actor_loss, navigation_actor_loss)

        def micro_batch_critic_loss_fn(critic_p, u_mb, micro_batch_key):
            states, actions_history, env_params, robot_params, inputs1 = u_mb["states"], u_mb["actions_history"], u_mb["env_params"], u_mb["robot_params"], u_mb["inputs1"]
            action_space_parameters = inputs1[:,0,:3]
            # Critic forward pass
            critic_keys = random.split(micro_batch_key, states.shape[0])
            pred_val = policy.batch_critic_forward(
                critic_keys,
                critic_p,
                states,
                actions_history,
                env_params, # Environment params
                robot_params, # Robot params
                action_space_parameters, # Action space parameters
            )
            # Critic loss
            v_loss = jnp.square(pred_val - u_mb["critic_targets"])
            v_clipped = u_mb["values"] + jnp.clip(pred_val - u_mb["values"], -clip_range, clip_range)
            v_loss_clipped = jnp.square(v_clipped - u_mb["critic_targets"])
            critic_loss = 0.5 * group_weighted_mean(
                jnp.maximum(v_loss, v_loss_clipped),
                u_mb["scenario"],
                social_weight,
            )
            y_true = u_mb["critic_targets"].flatten()
            y_pred = pred_val.flatten()
            var_y = jnp.var(y_true)
            explained_var = 1 - jnp.var(y_true - y_pred) / (var_y + 1e-8)
            return critic_loss, explained_var
        
        def _micro_step_scan(carry, u_mb):
            current_grads_acc, current_critic_grads_acc, current_metrics_acc, batch_key = carry
            batch_key, sub_key1, subkey2 = random.split(batch_key, 3)
            # Actor
            (policy_loss, aux), grads = value_and_grad(micro_batch_loss_fn, has_aux=True)(params_inner, u_mb, sub_key1)
            actor_new_grads_acc = tree_map(lambda acc, g: acc + g, current_grads_acc, grads)
            l_act, l_perc, l_safety, l_ent, w_ent, max_w, approx_kl, clip_frac, social_actor, navigation_actor = aux
            # Critic
            (critic_loss, explained_var), grads = value_and_grad(micro_batch_critic_loss_fn, has_aux=True)(critic_params_inner, u_mb, subkey2)
            critic_new_grads_acc = tree_map(lambda acc, g: acc + g, current_critic_grads_acc, grads)
            # Accumulation
            acc_pol, acc_act, acc_crit, acc_perc, acc_safety, acc_ent, acc_w_ent, acc_max_w, acc_kl, acc_clip, acc_explained_var, acc_social_actor, acc_navigation_actor = current_metrics_acc
            new_metrics_acc = (
                acc_pol + policy_loss,
                acc_act + l_act, 
                acc_crit + critic_loss, 
                acc_perc + l_perc, 
                acc_safety + l_safety, 
                acc_ent + l_ent, 
                acc_w_ent + w_ent,
                acc_max_w + max_w,
                acc_kl + approx_kl, 
                acc_clip + clip_frac, 
                acc_explained_var + explained_var,
                acc_social_actor + social_actor,
                acc_navigation_actor + navigation_actor,
            )
            return (actor_new_grads_acc, critic_new_grads_acc, new_metrics_acc, batch_key), None

        grads_acc_init = tree_map(jnp.zeros_like, params_inner)
        critic_grads_acc_init = tree_map(jnp.zeros_like, critic_params_inner)
        metrics_acc_init = (0.0,) * 13
        (grads_sum, critic_grads_sum, metrics_sum, batch_key), _ = lax.scan(
            _micro_step_scan, 
            (grads_acc_init, critic_grads_acc_init, metrics_acc_init, batch_key), 
            micro_batches
        )
        grads_avg = tree_map(lambda x: x / n_micro_splits, grads_sum)
        critic_grads_avg = tree_map(lambda x: x / n_micro_splits, critic_grads_sum)
        pol_sum, act_sum, crit_sum, perc_sum, safety_sum, ent_sum, w_ent_sum, max_w_sum, kl_sum, clip_sum, explained_var_sum, social_actor_sum, navigation_actor_sum = metrics_sum
        grad_norm = optax.global_norm(grads_avg)
        aux_avg = (
            pol_sum / n_micro_splits,
            act_sum / n_micro_splits,
            crit_sum / n_micro_splits,
            perc_sum / n_micro_splits,
            safety_sum / n_micro_splits,
            ent_sum / n_micro_splits,
            w_ent_sum / n_micro_splits,
            max_w_sum / n_micro_splits,
            kl_sum / n_micro_splits,
            clip_sum / n_micro_splits,
            explained_var_sum / n_micro_splits,
            grad_norm,
            social_actor_sum / n_micro_splits,
            navigation_actor_sum / n_micro_splits,
        )
        updates, new_opt_st_inner = optimizer.update(grads_avg, opt_st_inner)
        critic_updates, new_critic_opt_st_inner = critic_optimizer.update(critic_grads_avg, critic_opt_st_inner)
        candidate_params = optax.apply_updates(params_inner, updates)
        candidate_critic_params = optax.apply_updates(critic_params_inner, critic_updates)
        update_is_finite = (
            tree_all_finite(grads_avg)
            & tree_all_finite(critic_grads_avg)
            & tree_all_finite(candidate_params)
            & tree_all_finite(candidate_critic_params)
            & tree_all_finite(new_opt_st_inner)
            & tree_all_finite(new_critic_opt_st_inner)
        )
        # Never allow a bad CUDA/JAX update to corrupt the last known-good
        # training state.  The host loop below observes this flag and aborts with
        # an actionable error after synchronization.
        committed_params = tree_select(update_is_finite, candidate_params, params_inner)
        committed_critic_params = tree_select(
            update_is_finite, candidate_critic_params, critic_params_inner
        )
        committed_opt_state = tree_select(update_is_finite, new_opt_st_inner, opt_st_inner)
        committed_critic_opt_state = tree_select(
            update_is_finite, new_critic_opt_st_inner, critic_opt_st_inner
        )
        aux_avg = (*aux_avg, update_is_finite.astype(jnp.float32))
        return (
            committed_params,
            committed_critic_params,
            committed_opt_state,
            committed_critic_opt_state,
            batch_idx + 1,
            batch_key,
        ), aux_avg

    (new_params, new_critic_params, new_opt_st, new_critic_opt_st, _, _), batch_aux = lax.scan(
        _batch_step, (network_params, critic_network_params, opt_state, critic_opt_state, 0, key), batched_buffer
    )
    epoch_metrics = {
        "loss": jnp.mean(batch_aux[0]),
        "actor": jnp.mean(batch_aux[1]),
        "critic": jnp.mean(batch_aux[2]),
        "perc": jnp.mean(batch_aux[3]),
        "safety": jnp.mean(batch_aux[4]),
        "entropy": jnp.mean(batch_aux[5]),
        "weight_entropy": jnp.mean(batch_aux[6]),
        "max_weight": jnp.mean(batch_aux[7]),
        "approx_kl": jnp.mean(batch_aux[8]),
        "clip_frac": jnp.mean(batch_aux[9]),
        "explained_var": jnp.mean(batch_aux[10]),
        "grad_norm": jnp.mean(batch_aux[11]),
        "actor_social": jnp.mean(batch_aux[12]),
        "actor_navigation": jnp.mean(batch_aux[13]),
        "finite": jnp.all(batch_aux[14] > 0.5),
    }
    return (new_params, new_critic_params, new_opt_st, new_critic_opt_st), epoch_metrics

def get_dynamic_probabilities(success_rates, min_prob=0.03):
    """
    Computes dynamic sampling probabilities for scenarios based on their success rates.
    
    Args:
        success_rates (jnp.array): Array float [N_scenarios] with values between 0 and 1.
        min_prob (float): Minimum guaranteed probability for each scenario.
        
    Returns:
        jnp.array: Normalized probabilities that sum to 1.0.
    """
    n_scenarios = success_rates.shape[0]
    residual_budget = 1.0 - (n_scenarios * min_prob)
    residual_budget = jnp.maximum(residual_budget, 0.0)
    difficulties = (1.0 - success_rates)
    sum_difficulties = jnp.sum(difficulties) + 1e-6 
    variable_share = difficulties / sum_difficulties
    probs = min_prob + (variable_share * residual_budget)
    probs = probs / jnp.sum(probs)
    return probs


def get_social_curriculum_probabilities(
    scenario_keys,
    scenario_success=None,
    update=0,
):
    """Return aligned 90/10 -> 70/30 social-first sampling weights."""
    scenario_keys = tuple(int(key) for key in scenario_keys)
    social_fraction = 0.9 - 0.2 * float(jnp.clip((update - 200) / 300, 0.0, 1.0))
    success = (
        np.zeros((len(scenario_keys),), dtype=np.float32)
        if scenario_success is None
        else np.asarray(device_get(scenario_success), dtype=np.float32)
    )
    result = np.zeros((len(scenario_keys),), dtype=np.float32)

    def fill_group(group, budget):
        indices = [i for i, key in enumerate(scenario_keys) if key in group]
        if not indices:
            return False
        group_success = success[indices]
        difficulty = np.maximum(1.0 - group_success, 0.05)
        adaptive = difficulty / np.sum(difficulty)
        conditional = 0.3 / len(indices) + 0.7 * adaptive
        result[indices] = budget * conditional
        return True

    has_social = any(key in SOCIAL_SCENARIOS for key in scenario_keys)
    has_navigation = any(key in NAVIGATION_SCENARIOS for key in scenario_keys)
    if has_social and has_navigation:
        fill_group(SOCIAL_SCENARIOS, social_fraction)
        fill_group(NAVIGATION_SCENARIOS, 1.0 - social_fraction)
    elif has_social:
        fill_group(SOCIAL_SCENARIOS, 1.0)
    elif has_navigation:
        fill_group(NAVIGATION_SCENARIOS, 1.0)
    else:
        result[:] = 1.0 / len(scenario_keys)
    return jnp.asarray(result / np.sum(result), dtype=jnp.float32)


def summarize_evaluation(evaluation, scenario_keys):
    """Add social/navigation macro and worst-case metrics to an evaluation."""
    result = dict(evaluation)
    per_scenario = result["per_scenario"]
    for name, group in (("social", SOCIAL_SCENARIOS), ("navigation", NAVIGATION_SCENARIOS)):
        selected = [per_scenario[key] for key in scenario_keys if key in group]
        result[f"{name}_present"] = bool(selected)
        successes = [item["success"] for item in selected]
        result[f"{name}_macro_success"] = float(np.mean(successes)) if successes else 0.0
        result[f"{name}_worst_success"] = float(np.min(successes)) if successes else 0.0
        result[f"{name}_human_collision_rate"] = (
            float(np.mean([item["collision_with_human"] for item in selected]))
            if selected else 0.0
        )
    return result


def interpolated_bounds(nominal, lower, upper, fraction):
    """Return explicit (low, high) bounds for one curriculum stage."""
    low = tree_map(
        lambda base, bound: base + fraction * (bound - base), nominal, lower
    )
    high = tree_map(
        lambda base, bound: base + fraction * (bound - base), nominal, upper
    )
    return {name: (low[name], high[name]) for name in nominal}


def get_v4_scenario_probabilities(
    scenario_keys,
    evaluation=None,
    social_mastered=False,
):
    """Social-first fixed-budget probabilities driven only by gate evaluation."""
    scenario_keys = tuple(int(key) for key in scenario_keys)
    social_budget = 0.8 if social_mastered else 0.9
    result = np.zeros((len(scenario_keys),), dtype=np.float32)

    def difficulty(key, group):
        if evaluation is None or key not in evaluation.get("per_scenario", {}):
            return 1.0
        metrics = evaluation["per_scenario"][key]
        if group is SOCIAL_SCENARIOS:
            score = max(
                0.05,
                0.5 * (1.0 - metrics["success"])
                + 0.35 * metrics["collision_with_human"]
                + 0.15 * metrics["timeout"],
            )
            # Parallel traffic was the persistent weak social case in v3.  The
            # uniform half still bounds this modest targeted emphasis.
            return score * (1.25 if key == 1 else 1.0)
        return max(
            0.05,
            0.7 * (1.0 - metrics["success"])
            + 0.2 * metrics["collision_with_obstacle"]
            + 0.1 * metrics["timeout"],
        )

    def fill(group, budget):
        indices = [i for i, key in enumerate(scenario_keys) if key in group]
        if not indices:
            return False
        values = np.asarray([difficulty(scenario_keys[i], group) for i in indices])
        adaptive = values / values.sum()
        conditional = 0.5 / len(indices) + 0.5 * adaptive
        result[indices] = budget * conditional
        return True

    has_social = any(key in SOCIAL_SCENARIOS for key in scenario_keys)
    has_navigation = any(key in NAVIGATION_SCENARIOS for key in scenario_keys)
    if has_social and has_navigation:
        fill(SOCIAL_SCENARIOS, social_budget)
        fill(NAVIGATION_SCENARIOS, 1.0 - social_budget)
    elif has_social:
        fill(SOCIAL_SCENARIOS, 1.0)
    elif has_navigation:
        fill(NAVIGATION_SCENARIOS, 1.0)
    else:
        result[:] = 1.0 / len(scenario_keys)
    return jnp.asarray(result / result.sum(), dtype=jnp.float32)


def initial_v4_curriculum(level=0):
    level = int(np.clip(level, 0, len(CURRICULUM_STAGES) - 1))
    domain_fraction, visibility = CURRICULUM_STAGES[level]
    return {
        "level": level,
        "phase": curriculum_phase(level),
        "domain_fraction": domain_fraction,
        "visibility": visibility,
        "promotion_streak": 0,
        "regression_streak": 0,
        "last_transition_update": -50,
        "social_mastered": False,
        "transition_reason": "initialized",
    }


def update_v4_curriculum(curriculum, evaluation, update):
    """Update one ordered curriculum level from a current-stage evaluation."""
    state = dict(curriculum)
    social_ready = (
        evaluation["social_macro_success"] >= 0.80
        and evaluation["social_worst_success"] >= 0.50
        and evaluation["social_human_collision_rate"] <= 0.15
    )
    navigation_ready = (not evaluation.get("navigation_present", True)) or (
        evaluation["navigation_macro_success"] >= 0.75
        and evaluation["navigation_worst_success"] >= 0.40
    )
    regression = (
        evaluation["social_macro_success"] < 0.65
        or evaluation["social_human_collision_rate"] > 0.25
        or (
            evaluation.get("navigation_present", True)
            and evaluation["navigation_macro_success"] < 0.60
        )
    )
    state["social_mastered"] = bool(social_ready)
    state["promotion_streak"] = state["promotion_streak"] + 1 if social_ready and navigation_ready else 0
    state["regression_streak"] = state["regression_streak"] + 1 if regression else 0
    cooldown_ready = update - state["last_transition_update"] >= 50
    reason = "holding"
    next_level = state["level"]
    if cooldown_ready and state["regression_streak"] >= 2 and state["level"] > 0:
        next_level -= 1
        reason = "regressed: current-stage validation below safety floor"
    elif (
        cooldown_ready
        and state["promotion_streak"] >= 3
        and state["level"] < len(CURRICULUM_STAGES) - 1
    ):
        next_level += 1
        reason = "promoted: three consecutive current-stage validations passed"
    if next_level != state["level"]:
        state["level"] = next_level
        state["phase"] = curriculum_phase(next_level)
        state["domain_fraction"], state["visibility"] = CURRICULUM_STAGES[next_level]
        state["promotion_streak"] = 0
        state["regression_streak"] = 0
        state["social_mastered"] = False
        state["last_transition_update"] = int(update)
    state["transition_reason"] = reason
    return state


def curriculum_phase(level):
    """Human-readable phase persisted in v4 checkpoints and logs."""
    if level <= 5:
        return "domain_ramp"
    if level <= 15:
        return "joint_alternation"
    return "visibility_ramp"


def update_difficulty_curriculum(
    domain_fraction,
    visibility,
    scenario_keys,
    scenario_success,
    human_collision_rate,
    promotion_streak=0,
    visibility_streak=0,
):
    """Advance performance-gated domain and robot-visibility curricula."""
    rates = np.asarray(device_get(scenario_success), dtype=np.float32)
    social_rates = np.asarray(
        [rates[i] for i, key in enumerate(scenario_keys) if key in SOCIAL_SCENARIOS]
    )
    if social_rates.size == 0:
        return domain_fraction, visibility, promotion_streak, visibility_streak
    social_macro = float(np.mean(social_rates))
    social_worst = float(np.min(social_rates))
    promote = social_macro >= 0.70 and social_worst >= 0.25
    promotion_streak = promotion_streak + 1 if promote else 0
    if social_macro < 0.50 or human_collision_rate > 0.35:
        domain_fraction = max(0.0, domain_fraction - 0.05)
        promotion_streak = 0
    elif promotion_streak >= 2:
        domain_fraction = min(1.0, domain_fraction + 0.1)
        promotion_streak = 0

    visibility_ready = (
        social_macro >= 0.80
        and social_worst >= 0.50
        and human_collision_rate <= 0.15
    )
    visibility_streak = visibility_streak + 1 if visibility_ready else 0
    if social_macro < 0.65 or human_collision_rate > 0.25:
        visibility = min(1.0, visibility + 0.1)
        visibility_streak = 0
    elif visibility_streak >= 2:
        visibility = max(0.0, visibility - 0.1)
        visibility_streak = 0
    return domain_fraction, visibility, promotion_streak, visibility_streak


def scenario_curriculum_arrays(scenario_keys, success_rates, episode_counts):
    """Build aligned curriculum arrays for the configured scenario subset.

    Rollout accounting may contain entries for every scenario supported by the
    environment, while a training run can sample only a subset.  Always use the
    configured subset (and its order) for both the rates and observation mask so
    they remain aligned with ``scenarios_prob``.
    """
    return (
        jnp.asarray([success_rates[key] for key in scenario_keys], dtype=jnp.float32),
        jnp.asarray([episode_counts[key] > 0 for key in scenario_keys], dtype=bool),
    )


def evaluate_jessi_s2r_policy(
    params,
    policy,
    env,
    scenario_keys,
    episodes_per_scenario=16,
    seed=10_000,
    robot_param_bounds=None,
    env_param_bounds=None,
    visibility=1.0,
):
    """Deterministically evaluate fixed seeds without touching training state."""
    robot_nominal = validate_robot_params(env.get_default_robot_params())
    env_nominal = validate_env_params(env.get_default_env_params())
    robot_lower, robot_upper = bounds_from_nominal(robot_nominal, robot_param_bounds)
    eval_env_bounds = dict(env_param_bounds or {})
    eval_env_bounds["robot_visibility_probability"] = (visibility, visibility)
    env_lower, env_upper = bounds_from_nominal(env_nominal, eval_env_bounds)
    subset = tuple(int(value) for value in env.hybrid_scenario_subset)
    max_steps = int(np.ceil(env.reward_function.time_limit / env.robot_dt)) + 1
    per_scenario = {}

    for scenario in scenario_keys:
        probabilities = jnp.zeros((len(subset),), dtype=jnp.float32)
        probabilities = probabilities.at[subset.index(int(scenario))].set(1.0)
        reset_seed, env_seed, policy_seed = random.split(
            random.PRNGKey(seed + int(scenario)), 3
        )
        reset_keys = random.split(reset_seed, episodes_per_scenario)
        env_keys = random.split(env_seed, episodes_per_scenario)
        policy_keys = random.split(policy_seed, episodes_per_scenario)
        (
            states,
            reset_keys,
            observations,
            infos,
            robot_params,
            env_params,
            _,
        ) = env.batch_reset_with_params(
            reset_keys,
            robot_param_bounds=robot_param_bounds,
            env_param_bounds=eval_env_bounds,
            scenarios_prob=probabilities,
        )
        completed = jnp.zeros((episodes_per_scenario,), dtype=bool)
        results = {
            name: jnp.zeros((episodes_per_scenario,), dtype=bool)
            for name in ("success", "collision_with_human", "collision_with_obstacle", "timeout")
        }
        for _ in range(max_steps):
            actions, policy_keys, *_ = policy.batch_act_with_params(
                policy_keys,
                observations,
                infos,
                robot_params,
                params,
                sample=False,
            )
            (
                states,
                observations,
                infos,
                robot_params,
                env_params,
                _,
                outcomes,
                (reset_keys, env_keys),
            ) = env.batch_step_with_param_bounds(
                states,
                infos,
                robot_params,
                env_params,
                actions,
                reset_keys,
                env_keys,
                robot_lower,
                robot_upper,
                env_lower,
                env_upper,
                test=False,
                reset_if_done=False,
                scenarios_prob=probabilities,
            )
            newly_completed = (~outcomes["nothing"]) & (~completed)
            for name in results:
                results[name] = results[name] | (newly_completed & outcomes[name])
            completed = completed | newly_completed
            if bool(device_get(jnp.all(completed))):
                break
        per_scenario[int(scenario)] = {
            name: float(device_get(jnp.mean(values.astype(jnp.float32))))
            for name, values in results.items()
        }

    return {
        "per_scenario": per_scenario,
        "macro_success": float(np.mean([
            result["success"] for result in per_scenario.values()
        ])),
        "human_collision_rate": float(np.mean([
            result["collision_with_human"] for result in per_scenario.values()
        ])),
        "obstacle_collision_rate": float(np.mean([
            result["collision_with_obstacle"] for result in per_scenario.values()
        ])),
        "timeout_rate": float(np.mean([
            result["timeout"] for result in per_scenario.values()
        ])),
    }


def evaluate_at_curriculum_stage(
    params,
    policy,
    env,
    scenario_keys,
    stage,
    robot_nominal,
    robot_lower,
    robot_upper,
    env_nominal,
    env_lower,
    env_upper,
    episodes_per_scenario=16,
    seed=10_000,
):
    domain_fraction, visibility = stage
    robot_bounds = interpolated_bounds(
        robot_nominal, robot_lower, robot_upper, domain_fraction
    )
    env_bounds = interpolated_bounds(
        env_nominal, env_lower, env_upper, domain_fraction
    )
    evaluation = evaluate_jessi_s2r_policy(
        params,
        policy,
        env,
        scenario_keys,
        episodes_per_scenario=episodes_per_scenario,
        seed=seed,
        robot_param_bounds=robot_bounds,
        env_param_bounds=env_bounds,
        visibility=visibility,
    )
    evaluation.update({
        "domain_fraction": domain_fraction,
        "visibility": visibility,
    })
    return summarize_evaluation(evaluation, scenario_keys)


def select_warm_start_candidate(
    path,
    policy,
    env,
    scenario_keys,
    robot_nominal,
    robot_lower,
    robot_upper,
    env_nominal,
    env_lower,
    env_upper,
    episodes_per_scenario=16,
):
    """Select final/best legacy weights and the highest already-mastered stage."""
    candidates = load_warm_start_candidates(path)
    matrix_stages = ((0.0, 1.0), (0.5, 0.5), (1.0, 0.5), (1.0, 0.0))
    reports = {}
    scores = {}
    for candidate_name, (actor_params, critic_params) in candidates.items():
        evaluations = [
            evaluate_at_curriculum_stage(
                actor_params, policy, env, scenario_keys, stage,
                robot_nominal, robot_lower, robot_upper,
                env_nominal, env_lower, env_upper,
                episodes_per_scenario=episodes_per_scenario,
                seed=30_000 + index * 1_000,
            )
            for index, stage in enumerate(matrix_stages)
        ]
        target = evaluations[-1]
        scores[candidate_name] = (
            target["social_macro_success"],
            -target["social_human_collision_rate"],
            float(np.mean([item["social_macro_success"] for item in evaluations])),
            target["navigation_macro_success"],
        )
        reports[candidate_name] = evaluations
    selected_name = max(scores, key=scores.get)
    actor_params, critic_params = candidates[selected_name]
    mastered_level = -1
    low, high = 0, len(CURRICULUM_STAGES) - 1
    # The ordered stages are monotonically harder by construction; binary
    # search keeps warm-start evaluation bounded to five stage evaluations.
    while low <= high:
        level = (low + high) // 2
        stage = CURRICULUM_STAGES[level]
        evaluation = evaluate_at_curriculum_stage(
            actor_params, policy, env, scenario_keys, stage,
            robot_nominal, robot_lower, robot_upper,
            env_nominal, env_lower, env_upper,
            episodes_per_scenario=episodes_per_scenario,
            seed=40_000,
        )
        social_ready = (
            evaluation["social_macro_success"] >= 0.80
            and evaluation["social_worst_success"] >= 0.50
            and evaluation["social_human_collision_rate"] <= 0.15
        )
        navigation_ready = (
            not evaluation["navigation_present"]
            or (
                evaluation["navigation_macro_success"] >= 0.75
                and evaluation["navigation_worst_success"] >= 0.40
            )
        )
        if social_ready and navigation_ready:
            mastered_level = level
            low = level + 1
        else:
            high = level - 1
    mastered_level = max(mastered_level, 0)
    return actor_params, critic_params, mastered_level, {
        "selected": selected_name,
        "scores": scores,
        "matrix": reports,
    }


def update_ema(
    ema_success,
    batch_success,
    scenario_ema_success,
    scenario_batch_success,
    scenario_observed=None,
    alpha=0.08,
):
    scenario_batch_success = jnp.asarray(scenario_batch_success, dtype=jnp.float32)
    scenario_observed = (
        jnp.ones_like(scenario_batch_success, dtype=bool)
        if scenario_observed is None
        else jnp.asarray(scenario_observed, dtype=bool)
    )
    # Both branches must preserve the same public tuple contract.  Returning one
    # scalar/array here used to make the first curriculum update fail while its
    # caller attempted to unpack two values.
    new_ema_success = (
        batch_success
        if ema_success is None
        else (1.0 - alpha) * ema_success + alpha * batch_success
    )
    if scenario_ema_success is None:
        new_scenario_ema_success = jnp.where(
            scenario_observed, scenario_batch_success, 0.0
        )
    else:
        candidate_scenario_ema = (
            (1.0 - alpha) * scenario_ema_success
            + alpha * scenario_batch_success
        )
        new_scenario_ema_success = jnp.where(
            scenario_observed, candidate_scenario_ema, scenario_ema_success
        )
    return new_ema_success, new_scenario_ema_success

def jessi_s2r_rl_rollout(
    initial_actor_parameters,
    initial_critic_parameters,
    n_parallel_envs,
    train_updates,
    random_seed,
    actor_network_optimizer,
    critic_network_optimizer,
    total_batch_size,
    mini_batch_size,
    micro_batch_size,
    policy:JESSI_S2R,
    env:LaserNav,
    clip_range,
    n_epochs,
    beta_entropy,
    lambda_gae,
    training_type:str = "multitask",
    target_kl:float = None,
    safety_loss:bool = False,
    safety_mode:str = None,
    debugging:bool = False,
    robot_param_bounds:dict = None,
    env_param_bounds:dict = None,
    checkpoint_dir:str = None,
    checkpoint_every:int = 25,
    resume_from:str = None,
    keep_checkpoints:int = 3,
    checkpoint_config:dict = None,
    evaluation_every:int = 0,
    evaluation_episodes:int = 16,
    audit_every:int = 100,
    audit_episodes:int = 32,
    warm_start_from:str = None,
    curriculum_version:str = "v4",
):
    safety_mode = ("legacy" if safety_loss else "off") if safety_mode is None else safety_mode
    if safety_mode not in ("off", "legacy", "risk_aux"):
        raise ValueError("safety_mode must be one of: off, legacy, risk_aux")
    assert training_type in TRAINING_TYPES, "Invalid training type. Must be one of: " + ", ".join(TRAINING_TYPES)
    assert total_batch_size % n_parallel_envs == 0, "Total batch size must be divisible by number of parallel envs."
    assert mini_batch_size % micro_batch_size == 0, "Mini-batch size must be divisible by micro-batch size."
    assert total_batch_size % mini_batch_size == 0, "Total batch size must be divisible by mini-batch size."
    assert micro_batch_size % device_count() == 0, "Micro-batch size must be divisible by number of devices."
    training_type_name = training_type
    training_type = TRAINING_TYPES.index(training_type)
    n_steps = total_batch_size // n_parallel_envs
    n_minibatches = total_batch_size // mini_batch_size
    n_micro_splits = mini_batch_size // micro_batch_size
    devices = mesh_utils.create_device_mesh((device_count(),))
    mesh = Mesh(devices, axis_names=('env_axis',)) 
    sharding_env = NamedSharding(mesh, PartitionSpec('env_axis'))
    sharding_replicated = NamedSharding(mesh, PartitionSpec())
    key = random.PRNGKey(random_seed)
    robot_nominal = validate_robot_params(env.get_default_robot_params())
    env_nominal = validate_env_params(env.get_default_env_params())
    robot_lower, robot_upper = bounds_from_nominal(robot_nominal, robot_param_bounds)
    env_lower, env_upper = bounds_from_nominal(env_nominal, env_param_bounds)
    scenario_keys = tuple(int(s) for s in env.hybrid_scenario_subset)
    resume_config = {
        "device_count": device_count(),
        "n_parallel_envs": n_parallel_envs,
        "train_updates": train_updates,
        "total_batch_size": total_batch_size,
        "mini_batch_size": mini_batch_size,
        "micro_batch_size": micro_batch_size,
        "n_epochs": n_epochs,
        "training_type": training_type_name,
        "safety_mode": safety_mode,
        "scenario_keys": list(scenario_keys),
        "robot_param_bounds": _plain_value(robot_param_bounds or {}),
        "env_param_bounds": _plain_value(env_param_bounds or {}),
        "extra": _plain_value(checkpoint_config or {}),
        "evaluation_every": evaluation_every,
        "evaluation_episodes": evaluation_episodes,
        "audit_every": audit_every,
        "audit_episodes": audit_episodes,
        "curriculum_version": curriculum_version,
    }
    key, subkey = random.split(key)
    reset_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    key, subkey = random.split(key)
    env_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    curriculum = initial_v4_curriculum()
    scenarios_prob = get_v4_scenario_probabilities(scenario_keys)
    visibility = curriculum["visibility"]
    domain_fraction = curriculum["domain_fraction"]
    nominal_robot_bounds = {
        name: (value, value) for name, value in robot_nominal.items()
    }
    nominal_env_bounds = {
        name: (value, value) for name, value in env_nominal.items()
    }
    nominal_env_bounds["robot_visibility_probability"] = (visibility, visibility)
    (
        states,
        reset_keys,
        obses,
        infos,
        _,
        _,
        init_outcomes,
    ) = env.batch_reset_with_params(
        reset_keys,
        robot_param_bounds=nominal_robot_bounds,
        env_param_bounds=nominal_env_bounds,
        scenarios_prob=scenarios_prob,
    )
    states = tree_map(lambda x: device_put(x, sharding_env), states)
    obses = tree_map(lambda x: device_put(x, sharding_env), obses)
    infos = tree_map(lambda x: device_put(x, sharding_env), infos)
    init_outcomes = tree_map(lambda x: device_put(x, sharding_env), init_outcomes)
    env_state = (states, obses, infos, init_outcomes)
    key, subkey = random.split(key)
    policy_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    logs = {
        "losses": [], "returns": [], "successes": [], "failures": [], "timeouts": [],
        "collisions_humans": [], "collisions_obstacles": [], "times_to_goal": [], "episodes": [], 
        "perception_losses": [], "safety_losses": [], "actor_losses": [], "critic_losses": [], "entropy_losses": [],
        "weight_entropies": [], "max_weights": [],
        "stds": [], "grad_norm": [], "approx_kl": [], "clip_frac": [], "explained_var": [],
        "successes_per_scenario": {s: [] for s in scenario_keys},
        "episodes_per_scenario": {s: [] for s in scenario_keys},
        "domain_randomization_fraction": [],
        "visibility": [], "social_macro_success": [], "navigation_macro_success": [],
        "social_evaluations": [], "robust_evaluations": [],
        "stage_evaluations": [], "nominal_evaluations": [], "audit_evaluations": [],
        "curriculum_level": [], "curriculum_transitions": [],
        "social_sampling_budget": [],
        "actor_losses_social": [], "actor_losses_navigation": [],
        "mean_reward_social": [], "mean_reward_navigation": [],
        "min_clearance_social": [], "ttc_violation_social": [],
        "yielding_violation_social": [], "min_clearance_navigation": [],
        "ttc_violation_navigation": [], "yielding_violation_navigation": [],
        "episodes_social": [], "episodes_navigation": [],
        "human_collisions_social": [], "human_collisions_navigation": [],
        "obstacle_collisions_social": [], "obstacle_collisions_navigation": [],
        "warm_start": None,
    }
    init_beta_entropy = beta_entropy
    scenarios_labels = {}
    for i, scenario_name in enumerate(SCENARIOS):
        words = scenario_name.split('_')
        prefix = words[0][:2].capitalize()
        suffix = "".join([w[0] for w in words[1:]]) 
        scenarios_labels[i] = prefix + suffix
    scenario_ema_success = None
    ema_success = None
    start_update = 0
    if resume_from is None:
        warm_start_report = None
        if warm_start_from is not None:
            params, critic_params, warm_level, warm_start_report = select_warm_start_candidate(
                warm_start_from,
                policy,
                env,
                scenario_keys,
                robot_nominal,
                robot_lower,
                robot_upper,
                env_nominal,
                env_lower,
                env_upper,
                episodes_per_scenario=evaluation_episodes,
            )
            curriculum = initial_v4_curriculum(warm_level)
            visibility = curriculum["visibility"]
            domain_fraction = curriculum["domain_fraction"]
            print(
                f"Warm-start selected {warm_start_report['selected']} at "
                f"curriculum level {warm_level}: domain={domain_fraction:.1f}, "
                f"visibility={visibility:.1f}."
            )
            logs["warm_start"] = warm_start_report
        else:
            params = initial_actor_parameters
            critic_params = initial_critic_parameters
        # These snapshots must own their host memory: the working device arrays
        # are donated to the compiled PPO update.
        best_params = _host_tree(params)
        best_critic_params = _host_tree(critic_params)
        best_score = (-jnp.inf, -jnp.inf, -jnp.inf)
        best_curriculum_level = curriculum["level"]
        nominal_best_params = _host_tree(params)
        nominal_best_critic_params = _host_tree(critic_params)
        nominal_best_score = (-jnp.inf, -jnp.inf)
        robust_best_params = _host_tree(params)
        robust_best_critic_params = _host_tree(critic_params)
        robust_best_score = (-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf)
        opt_state = actor_network_optimizer.init(params)
        critic_opt_state = critic_network_optimizer.init(critic_params)
        params = device_put(params, sharding_replicated)
        critic_params = device_put(critic_params, sharding_replicated)
        opt_state = device_put(opt_state, sharding_replicated)
        critic_opt_state = device_put(critic_opt_state, sharding_replicated)
    else:
        checkpoint = load_training_checkpoint(resume_from, resume_config)
        restored = checkpoint["state"]
        params = device_put(restored["params"], sharding_replicated)
        critic_params = device_put(restored["critic_params"], sharding_replicated)
        opt_state = device_put(restored["opt_state"], sharding_replicated)
        critic_opt_state = device_put(
            restored["critic_opt_state"], sharding_replicated
        )
        best_params = restored["best_params"]
        best_critic_params = restored["best_critic_params"]
        best_score = tuple(restored["best_score"])
        best_curriculum_level = int(
            restored.get("best_curriculum_level", restored["curriculum"]["level"])
        )
        nominal_best_params = restored.get("nominal_best_params", restored["best_params"])
        nominal_best_critic_params = restored.get("nominal_best_critic_params", restored["best_critic_params"])
        nominal_best_score = tuple(restored.get("nominal_best_score", (-jnp.inf, -jnp.inf)))
        robust_best_params = restored.get("robust_best_params", restored["best_params"])
        robust_best_critic_params = restored.get("robust_best_critic_params", restored["best_critic_params"])
        robust_best_score = tuple(restored.get("robust_best_score", (-jnp.inf,) * 4))
        key = device_put(restored["key"])
        policy_keys = device_put(restored["policy_keys"], sharding_env)
        reset_keys = device_put(restored["reset_keys"], sharding_env)
        env_keys = device_put(restored["env_keys"], sharding_env)
        env_state = tree_map(lambda x: device_put(x, sharding_env), restored["env_state"])
        logs = restored["logs"]
        ema_success = restored["ema_success"]
        scenario_ema_success = restored["scenario_ema_success"]
        scenarios_prob = jnp.asarray(restored["scenarios_prob"])
        curriculum = restored["curriculum"]
        curriculum.setdefault("phase", curriculum_phase(curriculum["level"]))
        visibility = float(curriculum["visibility"])
        domain_fraction = float(curriculum["domain_fraction"])
        start_update = int(checkpoint["next_update"])
        print(f"Resuming from {resume_from} at update {start_update}.")
    if resume_from is None and curriculum["level"] > 0:
        stage_robot_bounds = interpolated_bounds(
            robot_nominal, robot_lower, robot_upper, domain_fraction
        )
        stage_env_bounds = interpolated_bounds(
            env_nominal, env_lower, env_upper, domain_fraction
        )
        stage_env_bounds["robot_visibility_probability"] = (visibility, visibility)
        states, reset_keys, obses, infos, _, _, init_outcomes = env.batch_reset_with_params(
            reset_keys,
            robot_param_bounds=stage_robot_bounds,
            env_param_bounds=stage_env_bounds,
            scenarios_prob=scenarios_prob,
        )
        reset_keys = device_put(reset_keys, sharding_env)
        env_state = tuple(
            tree_map(lambda x: device_put(x, sharding_env), value)
            for value in (states, obses, infos, init_outcomes)
        )
    train_one_epoch_sharded = None
    print(f"Starting optimized training loop for {train_updates} updates.")
    print(f"Rollout distributed across {len(devices)} devices.")
    for update in tqdm(range(start_update, train_updates)):
        entropy_progress = min(update / max(int(0.7 * train_updates), 1), 1.0)
        beta_entropy = init_beta_entropy * (1.0 - 0.9 * entropy_progress)
        # A. COLLECT ROLLOUT STEP (Parallel)
        rollout_robot_lower = tree_map(
            lambda nominal, bound: nominal + domain_fraction * (bound - nominal),
            robot_nominal,
            robot_lower,
        )
        rollout_robot_upper = tree_map(
            lambda nominal, bound: nominal + domain_fraction * (bound - nominal),
            robot_nominal,
            robot_upper,
        )
        rollout_env_lower = tree_map(
            lambda nominal, bound: nominal + domain_fraction * (bound - nominal),
            env_nominal,
            env_lower,
        ) | {
            "robot_visibility_probability": jnp.asarray(visibility, dtype=jnp.float32),
        }
        rollout_env_upper = tree_map(
            lambda nominal, bound: nominal + domain_fraction * (bound - nominal),
            env_nominal,
            env_upper,
        ) | {
            "robot_visibility_probability": jnp.asarray(visibility, dtype=jnp.float32),
        }
        env_state, policy_keys, reset_keys, env_keys, history_raw, outcomes_sum, returns, times, success_per_scenario, episodes_per_scenario = collect_rollout_step(
            params,
            critic_params,
            env_state,
            policy_keys,
            reset_keys,
            env_keys,
            init_outcomes,
            policy,
            env,
            n_steps,
            scenarios_prob,
            visibility,
            rollout_robot_lower,
            rollout_robot_upper,
            rollout_env_lower,
            rollout_env_upper,
        )
        current_states, current_obs, current_infos, current_dones = env_state[0], env_state[1], env_state[2], ~(env_state[3]['nothing'])
        n_succ = jnp.sum(outcomes_sum["success"])
        n_coll_hum = jnp.sum(outcomes_sum["collision_with_human"])
        n_coll_obs = jnp.sum(outcomes_sum["collision_with_obstacle"])
        n_timeout = jnp.sum(outcomes_sum["timeout"])
        ep_count = n_succ + n_coll_hum + n_coll_obs + n_timeout
        n_fail = n_coll_hum + n_coll_obs
        batch_mean_return = float(jnp.sum(returns)/ep_count) if ep_count > 0 else 0.0
        batch_mean_time = float(jnp.sum(times)/n_succ) if n_succ > 0 else 0.0
        avg_action_std = device_get(jnp.mean(history_raw["stds"], axis=(0,1)))
        success_per_scenario = {k: int(jnp.sum(success_per_scenario[k])) for k in success_per_scenario}
        episodes_per_scenario = {k: int(jnp.sum(episodes_per_scenario[k])) for k in episodes_per_scenario}
        success_rate_per_scenario = {k: (success_per_scenario[k] / episodes_per_scenario[k]) if episodes_per_scenario[k] > 0 else 0.0 for k in logs["successes_per_scenario"]}
        # B. PROCESS BUFFER (Parallel)
        keys = vmap(random.split)(policy_keys)
        policy_keys, critic_keys = keys[:,0], keys[:,1]
        buffer_gpu = process_buffer_and_gae(
            critic_params, critic_keys, current_states, current_obs, current_infos, current_dones, history_raw, policy, env, env.reward_function.gamma, policy.dt, policy.v_max, lambda_gae
        )
        # C. PREPARE TRAINING DATA
        def get_batched_shape_struct(x):
            target_shape = (n_minibatches, n_micro_splits, micro_batch_size, *x.shape[1:])
            return ShapeDtypeStruct(target_shape, x.dtype)
        if train_one_epoch_sharded is None:
            dummy_buffer_struct = tree_map(get_batched_shape_struct, buffer_gpu)
            train_pure = partial(
                train_one_epoch, 
                policy=policy,
                optimizer=actor_network_optimizer, 
                critic_optimizer=critic_network_optimizer,
                clip_range=clip_range, 
                compute_safety_loss=(safety_mode == "legacy"),
                compute_risk_auxiliary_loss=(safety_mode == "risk_aux"),
                training_type=training_type,
            )
            abstract_train_out = eval_shape(
                train_pure,
                key, params, opt_state, critic_params, critic_opt_state,
                dummy_buffer_struct, beta_entropy=beta_entropy,
                social_weight=0.9,
            )
            out_shardings_train = tree_map(lambda x: sharding_replicated, abstract_train_out)
            train_one_epoch_sharded = jit(
                train_pure,
                donate_argnums=(1, 2, 3, 4), # params, opt, critic_params, critic_opt_state
                out_shardings=out_shardings_train
            )
        epoch_metrics_acc = {
            "loss": [], "perc": [], "safety": [], "actor": [], "critic": [], "entropy": [], "weight_entropy": [], "max_weight": [], "approx_kl": [], "grad_norm": [], "clip_frac": [], "explained_var": [], "actor_social": [], "actor_navigation": []
        }
        # E. UPDATE LOOP
        for epoch in range(n_epochs):
            key, shuffle_key, data_aug_key = random.split(key, 3)
            perm = random.permutation(shuffle_key, total_batch_size)
            def shuffle_and_reshape_gpu(x):
                shuffled = jnp.take(x, perm, axis=0) 
                return jnp.reshape(shuffled, (n_minibatches, n_micro_splits, micro_batch_size, *x.shape[1:]))
            batched_buffer_gpu = tree_map(shuffle_and_reshape_gpu, buffer_gpu)
            (params, critic_params, opt_state, critic_opt_state), metrics_one_epoch = train_one_epoch_sharded(
                data_aug_key, 
                params, 
                opt_state, 
                critic_params,
                critic_opt_state,
                batched_buffer_gpu, 
                beta_entropy=beta_entropy,
                social_weight=(0.8 if curriculum["social_mastered"] else 0.9),
                debugging=(epoch==0) & (debugging),
            )
            metrics_one_epoch["loss"].block_until_ready() # SYNC
            if not bool(device_get(metrics_one_epoch["finite"])):
                raise FloatingPointError(
                    "Rejected a non-finite JESSI-S2R update; parameters and optimizer "
                    "state were rolled back to the last finite minibatch."
                )
            for k in epoch_metrics_acc:
                if k not in metrics_one_epoch: continue
                epoch_metrics_acc[k].append(metrics_one_epoch[k])
            if (target_kl is not None) and (jnp.mean(jnp.array(epoch_metrics_acc["approx_kl"])) > target_kl):
                print(f"Early stopping at epoch {epoch} due to reaching max KL.")
                break
        # F. LOGGING
        grad_norm = float(jnp.mean(jnp.array(epoch_metrics_acc['grad_norm'])))
        logs["losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["loss"]))))
        logs["perception_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["perc"]))))
        logs["safety_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["safety"]))))
        logs["actor_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["actor"]))))
        logs["actor_losses_social"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["actor_social"]))))
        logs["actor_losses_navigation"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["actor_navigation"]))))
        logs["critic_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["critic"]))))
        logs["entropy_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["entropy"]))))
        logs["weight_entropies"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["weight_entropy"]))))
        logs["max_weights"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["max_weight"]))))
        logs["returns"].append(batch_mean_return)
        logs["times_to_goal"].append(batch_mean_time)
        logs["successes"].append(int(n_succ))
        logs["failures"].append(int(n_fail))
        logs["timeouts"].append(int(n_timeout))
        logs["collisions_humans"].append(int(n_coll_hum))
        logs["collisions_obstacles"].append(int(n_coll_obs))
        logs["episodes"].append(int(ep_count))
        logs["domain_randomization_fraction"].append(domain_fraction)
        logs["visibility"].append(visibility)
        logs["curriculum_level"].append(curriculum["level"])
        logs["social_sampling_budget"].append(
            0.8 if curriculum["social_mastered"] else 0.9
        )
        flat_scenarios = history_raw["scenario"].reshape(-1)
        flat_rewards = history_raw["rewards"].reshape(-1)
        social_step_mask = social_mask_from_scenarios(flat_scenarios)
        logs["mean_reward_social"].append(float(group_weighted_mean(
            flat_rewards, flat_scenarios, 1.0
        )))
        logs["mean_reward_navigation"].append(float(group_weighted_mean(
            flat_rewards, flat_scenarios, 0.0
        )))
        logs["min_clearance_social"].append(float(jnp.where(
            jnp.any(social_step_mask),
            jnp.min(jnp.where(
                social_step_mask,
                history_raw["min_human_clearance"].reshape(-1),
                jnp.inf,
            )),
            0.0,
        )))
        logs["min_clearance_navigation"].append(float(jnp.where(
            jnp.any(~social_step_mask),
            jnp.min(jnp.where(
                ~social_step_mask,
                history_raw["min_human_clearance"].reshape(-1),
                jnp.inf,
            )),
            0.0,
        )))
        logs["ttc_violation_social"].append(float(group_weighted_mean(
            history_raw["ttc_violation"].reshape(-1).astype(jnp.float32),
            flat_scenarios,
            1.0,
        )))
        logs["yielding_violation_social"].append(float(group_weighted_mean(
            history_raw["yielding_violation"].reshape(-1).astype(jnp.float32),
            flat_scenarios,
            1.0,
        )))
        logs["ttc_violation_navigation"].append(float(group_weighted_mean(
            history_raw["ttc_violation"].reshape(-1).astype(jnp.float32),
            flat_scenarios,
            0.0,
        )))
        logs["yielding_violation_navigation"].append(float(group_weighted_mean(
            history_raw["yielding_violation"].reshape(-1).astype(jnp.float32),
            flat_scenarios,
            0.0,
        )))
        completed = history_raw["completed_episode"].reshape(-1).astype(jnp.float32)
        human_collision = history_raw["collision_with_human"].reshape(-1).astype(jnp.float32)
        obstacle_collision = history_raw["collision_with_obstacle"].reshape(-1).astype(jnp.float32)

        def episode_group_diagnostics(group_mask):
            group_completed = completed * group_mask.astype(jnp.float32)
            episode_count = jnp.sum(group_completed)
            denominator = jnp.maximum(episode_count, 1.0)
            return (
                episode_count,
                jnp.sum(human_collision * group_mask) / denominator,
                jnp.sum(obstacle_collision * group_mask) / denominator,
            )

        social_episode_count, social_human_rate, social_obstacle_rate = (
            episode_group_diagnostics(social_step_mask)
        )
        navigation_episode_count, navigation_human_rate, navigation_obstacle_rate = (
            episode_group_diagnostics(~social_step_mask)
        )
        logs["episodes_social"].append(int(social_episode_count))
        logs["episodes_navigation"].append(int(navigation_episode_count))
        logs["human_collisions_social"].append(float(social_human_rate))
        logs["human_collisions_navigation"].append(float(navigation_human_rate))
        logs["obstacle_collisions_social"].append(float(social_obstacle_rate))
        logs["obstacle_collisions_navigation"].append(float(navigation_obstacle_rate))
        logs["stds"].append(avg_action_std)
        logs["grad_norm"].append(grad_norm)
        logs["approx_kl"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["approx_kl"]))))
        logs["explained_var"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["explained_var"]))))
        logs["clip_frac"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["clip_frac"]))))
        logs["successes_per_scenario"] = {k: logs["successes_per_scenario"][k] + [success_per_scenario[k]] for k in logs["successes_per_scenario"]}
        logs["episodes_per_scenario"] = {k: logs["episodes_per_scenario"][k] + [episodes_per_scenario[k]] for k in logs["episodes_per_scenario"]}
        # G. EMA UPDATE and CURRICULUM UTILS
        batch_scenario_success_rate, scenario_observed = scenario_curriculum_arrays(
            scenario_keys,
            success_rate_per_scenario,
            episodes_per_scenario,
        )
        has_completed_episode = logs['episodes'][-1] > 0
        batch_success_rate = (
            logs['successes'][-1] / logs['episodes'][-1]
            if has_completed_episode
            else (float(ema_success) if ema_success is not None else 0.0)
        )
        if has_completed_episode:
            ema_success, scenario_ema_success = update_ema(
                ema_success,
                batch_success_rate,
                scenario_ema_success,
                batch_scenario_success_rate,
                scenario_observed,
            )
        social_rates = [
            float(scenario_ema_success[i])
            for i, scenario in enumerate(scenario_keys)
            if scenario in SOCIAL_SCENARIOS
        ] if scenario_ema_success is not None else []
        navigation_rates = [
            float(scenario_ema_success[i])
            for i, scenario in enumerate(scenario_keys)
            if scenario in NAVIGATION_SCENARIOS
        ] if scenario_ema_success is not None else []
        social_macro_success = float(np.mean(social_rates)) if social_rates else 0.0
        navigation_macro_success = float(np.mean(navigation_rates)) if navigation_rates else 0.0
        logs["social_macro_success"].append(social_macro_success)
        logs["navigation_macro_success"].append(navigation_macro_success)
        human_collision_rate = int(n_coll_hum) / max(int(ep_count), 1)
        score = (social_macro_success, -human_collision_rate, batch_success_rate)
        if evaluation_every <= 0 and score > best_score:
            best_score = score
            best_params = device_get(params)
            best_critic_params = device_get(critic_params)
        worst_3_scenarios_success_rate = (
            jnp.mean(jnp.sort(scenario_ema_success)[:3])
            if scenario_ema_success is not None
            else 0.0
        )
        # H. PRINT LOGS
        print(
            f"\nUpd {update}:\n",
            f"| Ret: {logs['returns'][-1]:.3f} | EMA Succ: {(ema_success if ema_success is not None else 0.0):.3f} | Succ: {batch_success_rate:.3f} | Fail: {logs['failures'][-1]/max(logs['episodes'][-1], 1):.3f} (hum {int(n_coll_hum)/max(logs['episodes'][-1], 1):.2f}, obs {int(n_coll_obs)/max(logs['episodes'][-1], 1):.2f}) | Timeouts: {logs['timeouts'][-1]/max(logs['episodes'][-1], 1):.3f}\n",
            f"| Action Stds: {logs['stds'][-1]} | Time to Goal: {logs['times_to_goal'][-1]:.2f}\n",
            f"| Macro SR: social {social_macro_success:.3f}, navigation {navigation_macro_success:.3f}\n",
            f"| SR x scenario - " + ", ".join([f"{scenarios_labels[k]}: {success_rate_per_scenario[k]:.2f}" for k in logs['successes_per_scenario']]) + "\n",
            f"| Episodes x scenario - " + ", ".join([f"{scenarios_labels[k]}: {episodes_per_scenario[k]}" for k in logs['episodes_per_scenario']]) + "\n",
            f"| EMA-SR x scenario - " + ", ".join([f"{scenarios_labels[k]}: {scenario_ema_success[i]:.2f}" for i, k in enumerate(logs['successes_per_scenario'])]) + "\n" if scenario_ema_success is not None else "",
            f"| Scenario Probs - " + ", ".join([f"{scenarios_labels[k]}: {scenarios_prob[i]:.2f}" for i, k in enumerate(logs['successes_per_scenario'])]) + "\n",
            f"| Actor Loss: {logs['actor_losses'][-1]:.4f} | Critic Loss: {logs['critic_losses'][-1]:.4f} | Perc Loss: {logs['perception_losses'][-1]:.4f} | Safety Loss: {logs['safety_losses'][-1]:.4f} |  Entropy Loss: {logs['entropy_losses'][-1]:.4f}\n",
            f"| Weights entropy: {logs['weight_entropies'][-1]:.4f} | Max weights: {logs['max_weights'][-1]:.4f} | Level: {curriculum['level']} ({curriculum['phase']}) | Visibility: {visibility:.2f} | Domain rand: {domain_fraction:.2f} | Social budget: {logs['social_sampling_budget'][-1]:.2f} \n",
            f"| Social diagnostics: reward {logs['mean_reward_social'][-1]:.4f}, min clearance {logs['min_clearance_social'][-1]:.3f}, TTC/yield {logs['ttc_violation_social'][-1]:.3f}/{logs['yielding_violation_social'][-1]:.3f}, episodes {logs['episodes_social'][-1]}, human/obstacle collisions {logs['human_collisions_social'][-1]:.3f}/{logs['obstacle_collisions_social'][-1]:.3f}\n",
            f"| Navigation diagnostics: reward {logs['mean_reward_navigation'][-1]:.4f}, min clearance {logs['min_clearance_navigation'][-1]:.3f}, episodes {logs['episodes_navigation'][-1]}, human/obstacle collisions {logs['human_collisions_navigation'][-1]:.3f}/{logs['obstacle_collisions_navigation'][-1]:.3f}\n",
            f"| Loss: {logs['losses'][-1]:.4f} | Grad Norm: {grad_norm:.4f} | Approx KL: {logs['approx_kl'][-1]:.4f} | Clip frac: {logs['clip_frac'][-1]:.4f} | Explained Var: {logs['explained_var'][-1]:.4f} \n",
        )
        # I. V4 CURRENT-STAGE CURRICULUM
        if evaluation_every > 0 and (
            (update + 1) % evaluation_every == 0
            or (update + 1 == train_updates)
        ):
            evaluated_level = curriculum["level"]
            stage_evaluation = evaluate_at_curriculum_stage(
                params,
                policy,
                env,
                scenario_keys,
                CURRICULUM_STAGES[evaluated_level],
                robot_nominal,
                robot_lower,
                robot_upper,
                env_nominal,
                env_lower,
                env_upper,
                episodes_per_scenario=evaluation_episodes,
                seed=10_000,
            )
            stage_evaluation.update({
                "update": update + 1,
                "curriculum_level": evaluated_level,
            })
            logs["stage_evaluations"].append(stage_evaluation)
            logs["social_evaluations"].append(stage_evaluation)
            if best_curriculum_level != evaluated_level:
                best_curriculum_level = evaluated_level
                best_score = (-jnp.inf, -jnp.inf, -jnp.inf)
            current_score = (
                stage_evaluation["social_macro_success"],
                -stage_evaluation["social_human_collision_rate"],
                stage_evaluation["navigation_macro_success"],
            )
            if current_score > best_score:
                best_score = current_score
                best_params = _host_tree(params)
                best_critic_params = _host_tree(critic_params)
            previous_level = curriculum["level"]
            curriculum = update_v4_curriculum(
                curriculum, stage_evaluation, update + 1
            )
            domain_fraction = curriculum["domain_fraction"]
            visibility = curriculum["visibility"]
            scenarios_prob = get_v4_scenario_probabilities(
                scenario_keys,
                stage_evaluation,
                curriculum["social_mastered"],
            )
            if curriculum["level"] != previous_level:
                transition = {
                    "update": update + 1,
                    "from_level": previous_level,
                    "to_level": curriculum["level"],
                    "reason": curriculum["transition_reason"],
                }
                logs["curriculum_transitions"].append(transition)
                # A current-stage snapshot must be scored at the current stage.
                # Start the new level from the transition parameters and replace
                # it after the first deterministic evaluation at that level.
                best_curriculum_level = curriculum["level"]
                best_score = (-jnp.inf, -jnp.inf, -jnp.inf)
                best_params = _host_tree(params)
                best_critic_params = _host_tree(critic_params)
            print(
                "Current-stage deterministic evaluation: "
                f"level={evaluated_level}, "
                f"social={stage_evaluation['social_macro_success']:.3f} "
                f"(worst={stage_evaluation['social_worst_success']:.3f}, "
                f"human collisions={stage_evaluation['social_human_collision_rate']:.3f}), "
                f"navigation={stage_evaluation['navigation_macro_success']:.3f} "
                f"(worst={stage_evaluation['navigation_worst_success']:.3f}), "
                f"promotion={curriculum['promotion_streak']}/3, "
                f"regression={curriculum['regression_streak']}/2; "
                f"{curriculum['transition_reason']}"
            )
            print(
                "Evaluation SR x scenario - "
                + ", ".join(
                    f"{scenarios_labels[key]}: {stage_evaluation['per_scenario'][key]['success']:.2f}"
                    for key in scenario_keys
                )
            )

            if audit_every > 0 and (update + 1) % audit_every == 0:
                nominal_evaluation = evaluate_at_curriculum_stage(
                    params, policy, env, scenario_keys, (0.0, 1.0),
                    robot_nominal, robot_lower, robot_upper,
                    env_nominal, env_lower, env_upper,
                    episodes_per_scenario=audit_episodes,
                    seed=20_000,
                )
                nominal_evaluation["update"] = update + 1
                logs["nominal_evaluations"].append(nominal_evaluation)
                nominal_score = (
                    nominal_evaluation["social_macro_success"],
                    -nominal_evaluation["social_human_collision_rate"],
                )
                if nominal_score > nominal_best_score:
                    nominal_best_score = nominal_score
                    nominal_best_params = _host_tree(params)
                    nominal_best_critic_params = _host_tree(critic_params)

                audit_evaluation = evaluate_at_curriculum_stage(
                    params,
                    policy,
                    env,
                    scenario_keys,
                    CURRICULUM_STAGES[-1],
                    robot_nominal,
                    robot_lower,
                    robot_upper,
                    env_nominal,
                    env_lower,
                    env_upper,
                    episodes_per_scenario=audit_episodes,
                    seed=50_000,
                )
                audit_evaluation["update"] = update + 1
                logs["audit_evaluations"].append(audit_evaluation)
                logs["robust_evaluations"].append(audit_evaluation)
                audit_score = (
                    audit_evaluation["social_macro_success"],
                    -audit_evaluation["social_human_collision_rate"],
                    audit_evaluation["navigation_macro_success"],
                    -audit_evaluation["timeout_rate"],
                )
                if audit_score > robust_best_score:
                    robust_best_score = audit_score
                    robust_best_params = _host_tree(params)
                    robust_best_critic_params = _host_tree(critic_params)
                print(
                    "Audit: "
                    f"nominal social={nominal_evaluation['social_macro_success']:.3f}; "
                    f"target social={audit_evaluation['social_macro_success']:.3f}, "
                    f"navigation={audit_evaluation['navigation_macro_success']:.3f}, "
                    f"human collisions={audit_evaluation['social_human_collision_rate']:.3f}"
                )

        if checkpoint_dir is not None and (
            ((update + 1) % max(int(checkpoint_every), 1) == 0)
            or (update + 1 == train_updates)
        ):
            checkpoint_state = {
                "params": params,
                "critic_params": critic_params,
                "opt_state": opt_state,
                "critic_opt_state": critic_opt_state,
                "best_params": best_params,
                "best_critic_params": best_critic_params,
                "best_score": best_score,
                "best_curriculum_level": best_curriculum_level,
                "nominal_best_params": nominal_best_params,
                "nominal_best_critic_params": nominal_best_critic_params,
                "nominal_best_score": nominal_best_score,
                "robust_best_params": robust_best_params,
                "robust_best_critic_params": robust_best_critic_params,
                "robust_best_score": robust_best_score,
                "key": key,
                "policy_keys": policy_keys,
                "reset_keys": reset_keys,
                "env_keys": env_keys,
                "env_state": env_state,
                "logs": logs,
                "ema_success": ema_success,
                "scenario_ema_success": scenario_ema_success,
                "scenarios_prob": scenarios_prob,
                "curriculum": curriculum,
            }
            saved_path = save_training_checkpoint(
                checkpoint_dir,
                update + 1,
                checkpoint_state,
                resume_config,
                keep=keep_checkpoints,
            )
            print(f"Saved checkpoint: {saved_path}")


    if checkpoint_dir is not None:
        atomic_pickle(os.path.join(checkpoint_dir, "best_current_stage.pkl"), {
            "params": best_params, "critic_params": best_critic_params,
            "score": best_score,
        })
        atomic_pickle(os.path.join(checkpoint_dir, "best_nominal.pkl"), {
            "params": nominal_best_params, "critic_params": nominal_best_critic_params,
            "score": nominal_best_score,
        })
        atomic_pickle(os.path.join(checkpoint_dir, "best_target_robust.pkl"), {
            "params": robust_best_params, "critic_params": robust_best_critic_params,
            "score": robust_best_score,
        })
    return best_params, device_get(params), best_critic_params, device_get(critic_params), logs
