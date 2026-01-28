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


@partial(jit, static_argnames=("policy", "env", "n_steps"))
def collect_rollout_step(
    network_params, 
    env_state, 
    policy_keys, 
    reset_keys, 
    env_keys,
    template_outcomes,
    policy, 
    env, 
    n_steps
):
    
    def _scan_step(carry, _):
        (states, obses, infos, outcomes, returns, times, success_per_scenario, episodes_per_scenario, p_keys, r_keys, e_keys, outcomes_acc) = carry
        actions, new_p_keys, _, sampled_actions, _, actor_distrs, values = policy.batch_act(
            p_keys, obses, infos, network_params, sample=True
        )
        new_states, new_obses, new_infos, rewards, new_outcomes, (new_r_keys, new_e_keys) = env.batch_step(
            states, infos, actions, reset_keys=r_keys, env_keys=e_keys, test=False, reset_if_done=True
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
            "obs": obses,
            "robot_goal": infos["robot_goal"],
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
    policy, 
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
    _, _, _, _, _, last_values = policy.e2e.apply(
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
    flat_adv = flatten(advantages)
    flattened_normalized_advantages = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)
    flattened_buffer = {
        "observations": flatten(history["obs"]),
        "robot_goals": flatten(history["robot_goal"]),
        "gt_poses": flatten(history["gt_poses"]),
        "gt_vels": flatten(history["gt_vels"]),
        "gt_mask": flatten(history["gt_mask"]),
        "actions": flatten(history["actions"]),
        "values": flatten(history["values"]),
        "neglogpdfs": flatten(history["neglogpdfs"]),
        "critic_targets": flatten(critic_targets),
        "advantages": flattened_normalized_advantages,
    }
    
    return flattened_buffer

@partial(jit,static_argnames=("policy", "optimizer", "clip_range", "beta_entropy"),donate_argnums=(1, 2, 3))
def train_one_epoch(
    key,
    network_params,
    opt_state,
    batched_buffer,
    policy,
    optimizer,
    clip_range,
    beta_entropy
):

    n_minibatches = batched_buffer["actions"].shape[0]
    n_micro_splits = batched_buffer["actions"].shape[1]

    def _batch_step(carry_inner, micro_batches): 
        params_inner, opt_st_inner = carry_inner 
        def micro_batch_loss_fn(p, u_mb):
            inputs0, inputs1 = vmap(policy.compute_e2e_input, in_axes=(0,0))(
                u_mb["observations"], u_mb["robot_goals"]
            )
            # Lowe input precision to save memory
            p = tree_map(lambda x: x.astype(jnp.bfloat16), p)
            inputs0 = inputs0.astype(jnp.bfloat16)
            inputs1 = inputs1.astype(jnp.bfloat16)
            # Forward pass
            (perc_dist, _, _, actor_dist, _, pred_val) = policy.e2e.apply(
                p, None, inputs0, inputs1
            )
            # Cast back to higher precision
            pred_val = pred_val.astype(jnp.float32)
            def dist_to_f32(dist):
                return tree_map(lambda x: x.astype(jnp.float32), dist)
            actor_dist = dist_to_f32(actor_dist)
            perc_dist = dist_to_f32(perc_dist)
            # Actor
            new_neglogp = policy.dirichlet.batch_neglogp(actor_dist, u_mb["actions"])
            log_ratio = u_mb["neglogpdfs"] - new_neglogp
            ratio = jnp.exp(log_ratio)
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
            # Entropy
            entropy = jnp.mean(policy.dirichlet.batch_entropy(actor_dist))
            entropy_loss = -beta_entropy * entropy
            policy_loss = actor_loss + entropy_loss
            # Perception
            gt_dict = {"gt_mask": u_mb["gt_mask"], "gt_poses": u_mb["gt_poses"], "gt_vels": u_mb["gt_vels"]}
            batch_perc_loss = policy._encoder_loss(perc_dist, gt_dict)
            perception_loss = jnp.mean(batch_perc_loss)
            total_loss = policy_loss + .5 * critic_loss + .05 * perception_loss
            return total_loss, (actor_loss, critic_loss, perception_loss, entropy_loss, approx_kl, clip_frac)

        def _micro_step_scan(carry, u_mb):
            current_grads_acc, current_metrics_acc = carry
            (loss, aux), grads = value_and_grad(micro_batch_loss_fn, has_aux=True)(params_inner, u_mb)
            new_grads_acc = tree_map(lambda acc, g: acc + g, current_grads_acc, grads)
            l_act, l_crit, l_perc, l_ent, approx_kl, clip_frac = aux
            acc_loss, (acc_act, acc_crit, acc_perc, acc_ent, acc_kl, acc_clip) = current_metrics_acc
            new_metrics_acc = (
                acc_loss + loss,
                (acc_act + l_act, acc_crit + l_crit, acc_perc + l_perc, acc_ent + l_ent, acc_kl + approx_kl, acc_clip + clip_frac)
            )
            return (new_grads_acc, new_metrics_acc), None

        grads_acc_init = tree_map(jnp.zeros_like, params_inner)
        metrics_acc_init = (0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) # loss, (actor, critic, perc, ent, kl, clip)
        (grads_sum, metrics_sum), _ = lax.scan(
            _micro_step_scan, 
            (grads_acc_init, metrics_acc_init), 
            micro_batches
        )
        grads_avg = tree_map(lambda x: x / n_micro_splits, grads_sum)
        loss_sum, (act_sum, crit_sum, perc_sum, ent_sum, kl_sum, clip_sum) = metrics_sum
        loss_avg = loss_sum / n_micro_splits
        grad_norm = optax.global_norm(grads_avg)
        aux_avg = (
            act_sum / n_micro_splits,
            crit_sum / n_micro_splits,
            perc_sum / n_micro_splits,
            ent_sum / n_micro_splits,
            kl_sum / n_micro_splits,
            clip_sum / n_micro_splits,
            grad_norm,
        )
        updates, new_opt_st_inner = optimizer.update(grads_avg, opt_st_inner)
        new_params_inner = optax.apply_updates(params_inner, updates)
        return (new_params_inner, new_opt_st_inner), (loss_avg, aux_avg)

    (new_params, new_opt_st), (batch_losses, batch_aux) = lax.scan(
        _batch_step, (network_params, opt_state), batched_buffer
    )
    epoch_metrics = {
        "loss": jnp.mean(batch_losses),
        "actor": jnp.mean(batch_aux[0]),
        "critic": jnp.mean(batch_aux[1]),
        "perc": jnp.mean(batch_aux[2]),
        "entropy": jnp.mean(batch_aux[3]),
        "approx_kl": jnp.mean(batch_aux[4]),
        "clip_frac": jnp.mean(batch_aux[5]),
        "grad_norm": jnp.mean(batch_aux[6]),
    }
    return (key, new_params, new_opt_st), epoch_metrics

def jessi_multitask_rl_rollout(
    initial_network_params,
    n_parallel_envs,
    train_updates,
    random_seed,
    network_optimizer,
    total_batch_size,
    mini_batch_size,
    micro_batch_size,
    policy,
    env,
    clip_range,
    n_epochs,
    beta_entropy,
    lambda_gae,
):
    assert total_batch_size % n_parallel_envs == 0, "Total batch size must be divisible by number of parallel envs."
    assert mini_batch_size % micro_batch_size == 0, "Mini-batch size must be divisible by micro-batch size."
    assert total_batch_size % mini_batch_size == 0, "Total batch size must be divisible by mini-batch size."
    assert micro_batch_size % device_count() == 0, "Micro-batch size must be divisible by number of devices."
    n_steps = total_batch_size // n_parallel_envs
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
        "perception_losses": [], "actor_losses": [], "critic_losses": [], "entropy_losses": [],
        "stds": [], "grad_norm": [], "approx_kl": [], "clip_frac": [],
        "successes_per_scenario": {int(s): [] for s in env.hybrid_scenario_subset},
        "episodes_per_scenario": {int(s): [] for s in env.hybrid_scenario_subset},
        
    }
    scenarios_labels = {}
    for i, scenario_name in enumerate(SCENARIOS):
        words = scenario_name.split('_')
        prefix = words[0][:2].capitalize()
        suffix = "".join([w[0] for w in words[1:]]) 
        scenarios_labels[i] = prefix + suffix
    print(f"Starting optimized training loop for {train_updates} updates.")
    print(f"Rollout distributed across {len(devices)} devices.")
    for update in tqdm(range(train_updates)):
        # A. COLLECT ROLLOUT STEP (Parallel)
        env_state, policy_keys, reset_keys, env_keys, history_raw, outcomes_sum, returns, times, success_per_scenario, episodes_per_scenario = collect_rollout_step(
            params, env_state, policy_keys, reset_keys, env_keys, init_outcomes, policy, env, n_steps
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
        # A.5 SAVE BEST PARAMS
        if batch_mean_return > best_return:
            best_return = batch_mean_return
            best_params = device_get(params)
        # B. PROCESS BUFFER (Parallel)
        buffer_gpu = process_buffer_and_gae(
            params, current_obs, current_infos, current_dones, history_raw, policy, policy.gamma, policy.dt, policy.v_max, lambda_gae
        )
        # C. COPY TO CPU (Only for efficient Shuffle)
        buffer_cpu = device_get(buffer_gpu)
        # D. PREPARE TRAINING DATA
        total_samples = buffer_cpu["actions"].shape[0]
        n_minibatches = total_samples // mini_batch_size
        n_micro_splits = mini_batch_size // micro_batch_size
        def reshape_helper_np(x):
            return x.reshape((n_minibatches, n_micro_splits, micro_batch_size, *x.shape[1:]))
        sharding_train = NamedSharding(mesh, PartitionSpec(None, None, 'env_axis')) 
        if update == 0:
            dummy_buffer_struct = tree_map(
                lambda x: ShapeDtypeStruct(x.shape, x.dtype), 
                tree_map(reshape_helper_np, buffer_cpu)
            )
            train_pure = partial(
                train_one_epoch, 
                policy=policy,
                optimizer=network_optimizer, 
                clip_range=clip_range, 
                beta_entropy=beta_entropy
            )
            abstract_train_out = eval_shape(
                train_pure,
                key, params, opt_state, dummy_buffer_struct # Passiamo strutture astratte
            )
            out_shardings_train = tree_map(lambda x: sharding_replicated, abstract_train_out)
            train_one_epoch_sharded = jit(
                train_pure,
                donate_argnums=(1, 2), # params, opt
                out_shardings=out_shardings_train
            )
        epoch_metrics_acc = {
            "loss": [], "perc": [], "actor": [], "critic": [], "entropy": [], "approx_kl": [], "grad_norm": [], "clip_frac": []
        }
        # E. UPDATE LOOP
        for epoch in range(n_epochs):
            key, subkey = random.split(key)
            perm = np.random.permutation(total_samples)
            shuffled_buffer_cpu = tree_map(lambda x: x[perm], buffer_cpu)
            batched_buffer_cpu = tree_map(reshape_helper_np, shuffled_buffer_cpu)
            batched_buffer_gpu = device_put(batched_buffer_cpu, sharding_train)
            (key, params, opt_state), metrics_one_epoch = train_one_epoch_sharded(
                key, 
                params, 
                opt_state, 
                batched_buffer_gpu, 
                # policy, 
                # network_optimizer, 
                # clip_range, 
                # beta_entropy
            )
            metrics_one_epoch["loss"].block_until_ready() # SYNC
            for k in epoch_metrics_acc:
                if k not in metrics_one_epoch: continue
                epoch_metrics_acc[k].append(metrics_one_epoch[k])
        # F. LOGGING
        grad_norm = float(jnp.mean(jnp.array(epoch_metrics_acc['grad_norm'])))
        logs["losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["loss"]))))
        logs["perception_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["perc"]))))
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
        logs["clip_frac"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["clip_frac"]))))
        logs["successes_per_scenario"] = {k: logs["successes_per_scenario"][k] + [success_per_scenario[k]] for k in logs["successes_per_scenario"]}
        logs["episodes_per_scenario"] = {k: logs["episodes_per_scenario"][k] + [episodes_per_scenario[k]] for k in logs["episodes_per_scenario"]}
        # G. PRINT LOGS
        if update % 1 == 0:
             print(
                f"Upd {update}:\n",
                f"| Ret: {logs['returns'][-1]:.3f} | Succ: {logs['successes'][-1]/logs['episodes'][-1]:.3f} | Fail: {logs['failures'][-1]/logs['episodes'][-1]:.3f} (hum {int(n_coll_hum)/logs['episodes'][-1]:.2f}, obs {int(n_coll_obs)/logs['episodes'][-1]:.2f}) | Timeouts: {logs['timeouts'][-1]/logs['episodes'][-1]:.3f}\n",
                f"| Action Stds: {logs['stds'][-1]} | Time to Goal: {logs['times_to_goal'][-1]:.2f}\n",
                f"| SR x scenario - " + ", ".join([f"{scenarios_labels[k]}: {logs['successes_per_scenario'][k][-1]/logs['episodes_per_scenario'][k][-1]:.2f}" for k in logs['successes_per_scenario']]) + "\n",
                f"| Actor Loss: {logs['actor_losses'][-1]:.4f} | Critic Loss: {logs['critic_losses'][-1]:.4f} | Perc Loss: {logs['perception_losses'][-1]:.4f} |  Entropy Loss: {logs['entropy_losses'][-1]:.4f}\n",
                f"| Loss: {logs['losses'][-1]:.4f} | Grad Norm: {grad_norm:.4f} | Approx KL: {logs['approx_kl'][-1]:.4f} | Clip frac: {logs['clip_frac'][-1]:.4f} \n",
             )
    return best_params, device_get(params), logs 
             