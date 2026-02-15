import optax
from jax import jit, lax, random, vmap, device_put, device_get, device_count, eval_shape, ShapeDtypeStruct, debug
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax import nn
import numpy as np
from tqdm import tqdm
from functools import partial
from jax import value_and_grad
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

from jhsfm.hsfm import get_linear_velocity
from socialjym.envs.base_env import SCENARIOS
from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi import JESSI


TRAINING_TYPES = ["multitask", "modular", "policy"]

@partial(jit, static_argnames=("policy", "env", "n_steps"))
def collect_rollout_step(
    network_params, 
    env_state, 
    policy_keys, 
    reset_keys, 
    env_keys,
    template_outcomes,
    policy:JESSI, 
    env:LaserNav, 
    n_steps,
    scenarios_prob,
):
    def _scan_step(carry, _):
        (states, obses, infos, outcomes, returns, times, success_per_scenario, episodes_per_scenario, p_keys, r_keys, e_keys, outcomes_acc) = carry
        actions, new_p_keys, inputs0, inputs1, _, sampled_actions, _, actor_distrs, values, masks = policy.batch_act(
            p_keys, obses, infos, network_params, sample=True
        )
        new_states, new_obses, new_infos, rewards, new_outcomes, (new_r_keys, new_e_keys) = env.batch_step(
            states, infos, actions, reset_keys=r_keys, env_keys=e_keys, test=False, reset_if_done=True, scenarios_prob=scenarios_prob
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
        humans_visibility, _ = env.batch_object_visibility(
            rc_humans_positions, infos["humans_parameters"][:,:,0], rc_obstacles
        )
        humans_in_range = env.batch_humans_inside_lidar_range(
            rc_humans_positions, infos["humans_parameters"][:,:,0]
        )
        step_data = {
            # "obs": obses,
            # "robot_goal": infos["robot_goal"],
            "inputs0": inputs0,
            "inputs1": inputs1,
            "masks": masks.astype(jnp.bool),
            "gt_poses": rc_humans_positions,
            "gt_vels": rc_humans_velocities,
            "gt_mask": humans_visibility & humans_in_range,
            "values": values,
            "actions": sampled_actions,
            "rewards": rewards,
            "dones": ~(outcomes["nothing"]),
            "neglogpdfs": policy.dirichlet.batch_neglogp(actor_distrs, sampled_actions),
            "stds": policy.dirichlet.batch_std(actor_distrs)
        }
        new_times = times + (new_outcomes["success"]) * (infos['time'] + policy.dt)
        new_returns = returns + (~new_outcomes["nothing"]) * (infos['return'] + jnp.power(policy.gamma, (infos['step']+1) * policy.dt * policy.v_max) * rewards)
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

@partial(jit, static_argnames=("policy",))
def process_buffer_and_gae(
    network_params, 
    last_obs, 
    last_info, 
    last_dones,
    history, 
    policy:JESSI, 
    gamma, 
    dt,
    vmax,
    lambda_gae
):
    """
    Calcola l'ultimo value, GAE, Returns e appiattisce il buffer.
    """
    perception_inputs, robot_state_inputs = vmap(policy.compute_e2e_input, in_axes=(0,0))(
        last_obs, last_info['robot_goal']
    )
    _, _, _, _, _, last_values, _ = policy.e2e.apply(
        network_params, None, perception_inputs, robot_state_inputs
    )
    rewards = history["rewards"]
    values = history["values"]
    dones = history["dones"]
    values_ext = jnp.concatenate([values, last_values[None, :]], axis=0)
    dones_ext = jnp.concatenate([dones, last_dones[None, :]], axis=0)
    def _gae_step(gae_carry, i):
        adv_next = gae_carry
        mask = 1.0 - dones_ext[i+1].astype(jnp.float32)
        delta = rewards[i] + jnp.power(gamma,dt*vmax) * values_ext[i+1] * mask - values_ext[i]
        advantage = delta + jnp.power(gamma*lambda_gae,dt*vmax) * adv_next * mask
        return advantage, advantage # Carry, Output
    n_steps = rewards.shape[0]
    _, advantages = lax.scan(_gae_step, jnp.zeros_like(values[0]), jnp.arange(n_steps)[::-1])
    advantages = advantages[::-1]
    critic_targets = advantages + values
    def flatten(x):
        return jnp.reshape(x, (-1, *x.shape[2:]))
    flattened_buffer = {
        # "observations": flatten(history["obs"]),
        # "robot_goals": flatten(history["robot_goal"]),
        "inputs0": flatten(history["inputs0"]),
        "inputs1": flatten(history["inputs1"]),
        "masks": flatten(history["masks"]),
        "gt_poses": flatten(history["gt_poses"]),
        "gt_vels": flatten(history["gt_vels"]),
        "gt_mask": flatten(history["gt_mask"]),
        "actions": flatten(history["actions"]),
        "values": flatten(history["values"]),
        "neglogpdfs": flatten(history["neglogpdfs"]),
        "critic_targets": flatten(critic_targets),
        "advantages": flatten(advantages),
    }
    
    return flattened_buffer

@partial(
    jit,
    static_argnames=("policy", "optimizer", "clip_range", "beta_entropy", "compute_safety_loss", "training_type"),
    donate_argnums=(1, 2, 3)
)
def train_one_epoch(
    key,
    network_params,
    opt_state,
    batched_buffer,
    policy:JESSI,
    optimizer,
    clip_range,
    beta_entropy,
    compute_safety_loss,
    training_type,
    debugging=False,
):
    multitask_training = (training_type == TRAINING_TYPES.index("multitask"))
    modular_training = (training_type == TRAINING_TYPES.index("modular"))
    policy_training = (training_type == TRAINING_TYPES.index("policy"))

    n_minibatches = batched_buffer["actions"].shape[0]
    n_micro_splits = batched_buffer["actions"].shape[1]

    def _batch_step(carry_inner, micro_batches): 
        params_inner, opt_st_inner, batch_idx, batch_key = carry_inner 

        # Normalize advantages within mini-batch
        all_mb_advantages = micro_batches["advantages"]
        norm_advantages = (all_mb_advantages - jnp.mean(all_mb_advantages)) / (jnp.std(all_mb_advantages) + 1e-8)
        # We clip the normalized advantages to avoid too large policy updates
        micro_batches["advantages"] = micro_batches["advantages"].at[:].set(jnp.clip(norm_advantages, -5, 5))

        def micro_batch_loss_fn(p, u_mb, micro_batch_key):
            inputs0, inputs1, masks = u_mb["inputs0"], u_mb["inputs1"], u_mb["masks"]
            # Lowe input precision to save memory
            if multitask_training or modular_training:
                inputs0_f16 = inputs0.astype(jnp.bfloat16)
                inputs1_f16 = inputs1.astype(jnp.bfloat16)
                # Forward pass (For Actor/Critic)
                (safety_perc_dist, _, _, actor_dist, _, pred_val, _) = policy.e2e.apply(
                    p, None, inputs0_f16, inputs1_f16, stop_perception_gradient=~(multitask_training), external_mask=masks
                )
            else:
                # Forward pass (For Actor/Critic)
                (safety_perc_dist, _, _, actor_dist, _, pred_val, _) = policy.e2e.apply(
                    p, None, inputs0, inputs1, stop_perception_gradient=~(multitask_training), external_mask=masks
                )
            # Cast back to higher precision for loss computation
            if multitask_training or modular_training:
                pred_val = pred_val.astype(jnp.float32)
                def dist_to_f32(dist):
                    return tree_map(lambda x: x.astype(jnp.float32), dist)
                actor_dist = dist_to_f32(actor_dist)  
                safety_perc_dist = dist_to_f32(safety_perc_dist)
            # Actor
            new_neglogp = policy.dirichlet.batch_neglogp(actor_dist, u_mb["actions"])
            log_ratio = u_mb["neglogpdfs"] - new_neglogp
            # log_ratio = jnp.clip(log_ratio, -10, 10) # MORE STABLE
            ratio = jnp.exp(log_ratio)
            lax.cond(
                debugging & (batch_idx == 0),
                lambda : debug.print("Mean Ratio is: {m} - Std Ratio is: {s}", m=jnp.mean(ratio), s=jnp.std(ratio)),
                lambda : None,
            )
            surr1 = ratio * u_mb["advantages"]
            surr2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * u_mb["advantages"]
            actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            approx_kl = jnp.mean((ratio - 1) - log_ratio)
            clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_range)
            # Critic
            v_loss = jnp.square(pred_val - u_mb["critic_targets"])
            v_clipped = u_mb["values"] + jnp.clip(pred_val - u_mb["values"], -clip_range, clip_range)
            v_loss_clipped = jnp.square(v_clipped - u_mb["critic_targets"])
            critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss, v_loss_clipped))
            y_true = u_mb["critic_targets"].flatten()
            y_pred = pred_val.flatten()
            var_y = jnp.var(y_true)
            explained_var = 1 - jnp.var(y_true - y_pred) / (var_y + 1e-8)
            # Entropy
            entropy = jnp.mean(policy.dirichlet.batch_entropy(actor_dist))
            entropy_loss = -beta_entropy * entropy
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
                (perc_dist, _, _, _, _, _, _) = policy.e2e.apply(
                    p, None, inputs0_corrupt_f16, inputs1, stop_perception_gradient=False, only_perception=True, perception_key=mask_drop_key
                )
                # Compute perception loss
                perc_dist = dist_to_f32(perc_dist)
                batch_perc_loss = policy._perception_loss(perc_dist, gt_dict)
                perception_loss = jnp.mean(batch_perc_loss)
            else:
                perception_loss = 0.0
            # Safety loss (optional)
            if compute_safety_loss:
                safety_loss = policy._safety_loss(
                    actor_dist,
                    safety_perc_dist,
                )
            else:
                safety_loss = 0.0
            # Total loss
            total_loss = policy_loss + .5 * critic_loss + .05 * perception_loss + safety_loss
            return total_loss, (actor_loss, critic_loss, perception_loss, safety_loss, entropy_loss, approx_kl, clip_frac, explained_var)

        def _micro_step_scan(carry, u_mb):
            current_grads_acc, current_metrics_acc, batch_key = carry
            batch_key, sub_key = random.split(batch_key)
            (loss, aux), grads = value_and_grad(micro_batch_loss_fn, has_aux=True)(params_inner, u_mb, sub_key)
            new_grads_acc = tree_map(lambda acc, g: acc + g, current_grads_acc, grads)
            l_act, l_crit, l_perc, l_safety, l_ent, approx_kl, clip_frac, explained_var = aux
            acc_loss, (acc_act, acc_crit, acc_perc, acc_safety, acc_ent, acc_kl, acc_clip, acc_explained_var) = current_metrics_acc
            new_metrics_acc = (
                acc_loss + loss,
                (acc_act + l_act, acc_crit + l_crit, acc_perc + l_perc, acc_safety + l_safety, acc_ent + l_ent, acc_kl + approx_kl, acc_clip + clip_frac, acc_explained_var + explained_var)
            )
            return (new_grads_acc, new_metrics_acc, batch_key), None

        grads_acc_init = tree_map(jnp.zeros_like, params_inner)
        metrics_acc_init = (0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) # loss, (actor, critic, perc, safety, ent, kl, clip, explained_var)
        (grads_sum, metrics_sum, batch_key), _ = lax.scan(
            _micro_step_scan, 
            (grads_acc_init, metrics_acc_init, batch_key), 
            micro_batches
        )
        grads_avg = tree_map(lambda x: x / n_micro_splits, grads_sum)
        loss_sum, (act_sum, crit_sum, perc_sum, safety_sum, ent_sum, kl_sum, clip_sum, explained_var_sum) = metrics_sum
        loss_avg = loss_sum / n_micro_splits
        grad_norm = optax.global_norm(grads_avg)
        aux_avg = (
            act_sum / n_micro_splits,
            crit_sum / n_micro_splits,
            perc_sum / n_micro_splits,
            safety_sum / n_micro_splits,
            ent_sum / n_micro_splits,
            kl_sum / n_micro_splits,
            clip_sum / n_micro_splits,
            explained_var_sum / n_micro_splits,
            grad_norm,
        )
        updates, new_opt_st_inner = optimizer.update(grads_avg, opt_st_inner)
        new_params_inner = optax.apply_updates(params_inner, updates)
        return (new_params_inner, new_opt_st_inner, batch_idx + 1, batch_key), (loss_avg, aux_avg)

    (new_params, new_opt_st, _, _), (batch_losses, batch_aux) = lax.scan(
        _batch_step, (network_params, opt_state, 0, key), batched_buffer
    )
    epoch_metrics = {
        "loss": jnp.mean(batch_losses),
        "actor": jnp.mean(batch_aux[0]),
        "critic": jnp.mean(batch_aux[1]),
        "perc": jnp.mean(batch_aux[2]),
        "safety": jnp.mean(batch_aux[3]),
        "entropy": jnp.mean(batch_aux[4]),
        "approx_kl": jnp.mean(batch_aux[5]),
        "clip_frac": jnp.mean(batch_aux[6]),
        "explained_var": jnp.mean(batch_aux[7]),
        "grad_norm": jnp.mean(batch_aux[8]),
    }
    return (new_params, new_opt_st), epoch_metrics

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

def jessi_multitask_rl_rollout(
    initial_network_params,
    n_parallel_envs,
    train_updates,
    random_seed,
    network_optimizer,
    total_batch_size,
    mini_batch_size,
    micro_batch_size,
    policy:JESSI,
    env:LaserNav,
    clip_range,
    n_epochs,
    beta_entropy,
    lambda_gae,
    training_type:str = "multitask",
    target_kl:float = None,
    safety_loss:bool = False,
    debugging:bool = False,
):
    assert training_type in TRAINING_TYPES, "Invalid training type. Must be one of: " + ", ".join(TRAINING_TYPES)
    assert total_batch_size % n_parallel_envs == 0, "Total batch size must be divisible by number of parallel envs."
    assert mini_batch_size % micro_batch_size == 0, "Mini-batch size must be divisible by micro-batch size."
    assert total_batch_size % mini_batch_size == 0, "Total batch size must be divisible by mini-batch size."
    assert micro_batch_size % device_count() == 0, "Micro-batch size must be divisible by number of devices."
    training_type = TRAINING_TYPES.index(training_type)
    n_steps = total_batch_size // n_parallel_envs
    n_minibatches = total_batch_size // mini_batch_size
    n_micro_splits = mini_batch_size // micro_batch_size
    devices = mesh_utils.create_device_mesh((device_count(),))
    mesh = Mesh(devices, axis_names=('env_axis',)) 
    sharding_env = NamedSharding(mesh, PartitionSpec('env_axis'))
    sharding_replicated = NamedSharding(mesh, PartitionSpec())
    key = random.PRNGKey(random_seed)
    key, subkey = random.split(key)
    reset_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    key, subkey = random.split(key)
    env_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    states, reset_keys, obses, infos, init_outcomes = env.batch_reset(reset_keys)
    states = tree_map(lambda x: device_put(x, sharding_env), states)
    obses = tree_map(lambda x: device_put(x, sharding_env), obses)
    infos = tree_map(lambda x: device_put(x, sharding_env), infos)
    init_outcomes = tree_map(lambda x: device_put(x, sharding_env), init_outcomes)
    env_state = (states, obses, infos, init_outcomes)
    key, subkey = random.split(key)
    policy_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    params = initial_network_params
    best_params = initial_network_params.copy()
    best_return = -jnp.inf
    opt_state = network_optimizer.init(params)
    params = device_put(params, sharding_replicated)
    opt_state = device_put(opt_state, sharding_replicated)
    logs = {
        "losses": [], "returns": [], "successes": [], "failures": [], "timeouts": [],
        "collisions_humans": [], "collisions_obstacles": [], "times_to_goal": [], "episodes": [], 
        "perception_losses": [], "safety_losses": [], "actor_losses": [], "critic_losses": [], "entropy_losses": [],
        "stds": [], "grad_norm": [], "approx_kl": [], "clip_frac": [], "explained_var": [],
        "successes_per_scenario": {int(s): [] for s in env.hybrid_scenario_subset},
        "episodes_per_scenario": {int(s): [] for s in env.hybrid_scenario_subset},
        
    }
    scenarios_labels = {}
    for i, scenario_name in enumerate(SCENARIOS):
        words = scenario_name.split('_')
        prefix = words[0][:2].capitalize()
        suffix = "".join([w[0] for w in words[1:]]) 
        scenarios_labels[i] = prefix + suffix
    scenarios_prob = jnp.array([1.0 / (len(env.hybrid_scenario_subset))] * len(env.hybrid_scenario_subset))
    print(f"Starting optimized training loop for {train_updates} updates.")
    print(f"Rollout distributed across {len(devices)} devices.")
    for update in tqdm(range(train_updates)):
        # A. COLLECT ROLLOUT STEP (Parallel)
        env_state, policy_keys, reset_keys, env_keys, history_raw, outcomes_sum, returns, times, success_per_scenario, episodes_per_scenario = collect_rollout_step(
            params, env_state, policy_keys, reset_keys, env_keys, init_outcomes, policy, env, n_steps, scenarios_prob
        )
        current_obs, current_infos, current_dones = env_state[1], env_state[2], ~(env_state[3]['nothing'])
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
        # A.5 SAVE BEST PARAMS
        if batch_mean_return > best_return:
            best_return = batch_mean_return
            best_params = device_get(params)
        # B. PROCESS BUFFER (Parallel)
        buffer_gpu = process_buffer_and_gae(
            params, current_obs, current_infos, current_dones, history_raw, policy, policy.gamma, policy.dt, policy.v_max, lambda_gae
        )
        # C. PREPARE TRAINING DATA
        def get_batched_shape_struct(x):
            target_shape = (n_minibatches, n_micro_splits, micro_batch_size, *x.shape[1:])
            return ShapeDtypeStruct(target_shape, x.dtype)
        if update == 0:
            dummy_buffer_struct = tree_map(get_batched_shape_struct, buffer_gpu)
            train_pure = partial(
                train_one_epoch, 
                policy=policy,
                optimizer=network_optimizer, 
                clip_range=clip_range, 
                beta_entropy=beta_entropy,
                compute_safety_loss=safety_loss,
                training_type=training_type,
            )
            abstract_train_out = eval_shape(
                train_pure,
                key, params, opt_state, dummy_buffer_struct 
            )
            out_shardings_train = tree_map(lambda x: sharding_replicated, abstract_train_out)
            train_one_epoch_sharded = jit(
                train_pure,
                donate_argnums=(1, 2), # params, opt
                out_shardings=out_shardings_train
            )
        epoch_metrics_acc = {
            "loss": [], "perc": [], "safety": [], "actor": [], "critic": [], "entropy": [], "approx_kl": [], "grad_norm": [], "clip_frac": [], "explained_var": []
        }
        # E. UPDATE LOOP
        for epoch in range(n_epochs):
            key, shuffle_key, data_aug_key = random.split(key, 3)
            perm = random.permutation(shuffle_key, total_batch_size)
            def shuffle_and_reshape_gpu(x):
                shuffled = jnp.take(x, perm, axis=0) 
                return jnp.reshape(shuffled, (n_minibatches, n_micro_splits, micro_batch_size, *x.shape[1:]))
            batched_buffer_gpu = tree_map(shuffle_and_reshape_gpu, buffer_gpu)
            (params, opt_state), metrics_one_epoch = train_one_epoch_sharded(
                data_aug_key, 
                params, 
                opt_state, 
                batched_buffer_gpu, 
                debugging=(epoch==0) & (debugging),
            )
            metrics_one_epoch["loss"].block_until_ready() # SYNC
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
        logs["critic_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["critic"]))))
        logs["entropy_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["entropy"]))))
        logs["returns"].append(batch_mean_return)
        logs["times_to_goal"].append(batch_mean_time)
        logs["successes"].append(int(n_succ))
        logs["failures"].append(int(n_fail))
        logs["timeouts"].append(int(n_timeout))
        logs["collisions_humans"].append(int(n_coll_hum))
        logs["collisions_obstacles"].append(int(n_coll_obs))
        logs["episodes"].append(int(ep_count))
        logs["stds"].append(avg_action_std)
        logs["grad_norm"].append(grad_norm)
        logs["approx_kl"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["approx_kl"]))))
        logs["explained_var"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["explained_var"]))))
        logs["clip_frac"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["clip_frac"]))))
        logs["successes_per_scenario"] = {k: logs["successes_per_scenario"][k] + [success_per_scenario[k]] for k in logs["successes_per_scenario"]}
        logs["episodes_per_scenario"] = {k: logs["episodes_per_scenario"][k] + [episodes_per_scenario[k]] for k in logs["episodes_per_scenario"]}
        # G. PRINT LOGS
        print(
            f"Upd {update}:\n",
            f"| Ret: {logs['returns'][-1]:.3f} | Succ: {logs['successes'][-1]/logs['episodes'][-1]:.3f} | Fail: {logs['failures'][-1]/logs['episodes'][-1]:.3f} (hum {int(n_coll_hum)/logs['episodes'][-1]:.2f}, obs {int(n_coll_obs)/logs['episodes'][-1]:.2f}) | Timeouts: {logs['timeouts'][-1]/logs['episodes'][-1]:.3f}\n",
            f"| Action Stds: {logs['stds'][-1]} | Time to Goal: {logs['times_to_goal'][-1]:.2f}\n",
            f"| SR x scenario - " + ", ".join([f"{scenarios_labels[k]}: {success_rate_per_scenario[k]:.2f}" for k in logs['successes_per_scenario']]) + "\n",
            f"| Scenario Probs - " + ", ".join([f"{scenarios_labels[k]}: {scenarios_prob[i]:.2f}" for i, k in enumerate(logs['successes_per_scenario'])]) + "\n",
            f"| Actor Loss: {logs['actor_losses'][-1]:.4f} | Critic Loss: {logs['critic_losses'][-1]:.4f} | Perc Loss: {logs['perception_losses'][-1]:.4f} | Safety Loss: {logs['safety_losses'][-1]:.4f} |  Entropy Loss: {logs['entropy_losses'][-1]:.4f}\n",
            f"| Loss: {logs['losses'][-1]:.4f} | Grad Norm: {grad_norm:.4f} | Approx KL: {logs['approx_kl'][-1]:.4f} | Clip frac: {logs['clip_frac'][-1]:.4f} | Explained Var: {logs['explained_var'][-1]:.4f} \n",
        )
        # H. UPDATE SCENARIO PROBS
        scenarios_prob = get_dynamic_probabilities(jnp.array([success_rate_per_scenario[k] for k in sorted(success_rate_per_scenario)]))
    return best_params, device_get(params), logs 
             