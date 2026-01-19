import optax
from jax import jit, lax, random, vmap, device_put, device_get, device_count, eval_shape, ShapeDtypeStruct
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from jax import value_and_grad
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

from jhsfm.hsfm import get_linear_velocity


def jessi_multitask_rl_rollout(
        initial_network_params: dict,
        n_parallel_envs:int,
        train_updates: int, # Number of episodes to train on, exploration episodes excluded
        random_seed: int,
        network_optimizer: optax.GradientTransformation,
        total_batch_size: int,
        mini_batch_size: int,
        policy: JESSI,
        env: LaserNav,
        clip_range: float,
        n_epochs: int,
        beta_entropy: float,
        lambda_gae: float,
        debugging: bool = False,
        debugging_interval: int = 100,
) -> dict:
        assert policy.name == "JESSI", "This function is only compatible with JESSI."
        # Compute number of steps to simulate per update for each parallel env
        assert total_batch_size % n_parallel_envs == 0, "The buffer capacity must be a multiple of the number of parallel environments. Otherwise you will trow away experiences."
        assert total_batch_size % mini_batch_size == 0, "The total batch size must be a multiple of the mini-batch size."
        assert env.humans_policy == HUMAN_POLICIES.index("hsfm"), "This training is only compatible with HSFM humans policy."
        n_steps = int(total_batch_size / n_parallel_envs)
        n_batches = int(total_batch_size / mini_batch_size)

        # Define the main rollout episode loop
        @loop_tqdm(train_updates)
        @jit
        def _update_for_loop_body(upd_idx:int, val:tuple):
                # Retrieve data from the tuple
                policy_keys, reset_keys, shuffle_key, states, obses, infos, outcomes, network_params, net_opt_state, current_buffer_size, aux_data = val
                ## Generate experience
                @jit
                def _step_fori_body(step:int, val:tuple):
                        # Retrieve data from the tuple
                        network_params, states, obses, infos, init_outcomes, policy_keys, reset_keys, episode_count, batch_robot_goals, batch_supervised_target, batch_observations, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, successes, failures, timeouts, returns, batch_stds = val
                        ## Step
                        actions, policy_keys, _, sampled_actions, _, actor_distrs, values = policy.batch_act(policy_keys, obses, infos, network_params, sample=True)
                        new_states, new_obses, new_infos, rewards, outcomes, reset_keys = env.batch_step(states, infos, actions, reset_keys, test=False, reset_if_done=True)
                        ## Compute Supervised targets
                        rc_humans_positions, _, rc_humans_velocities, rc_obstacles, _ = env.batch_robot_centric_transform(
                                states[:,:-1,:2], # humans_positions,
                                states[:,:-1,4], # humans_orientations,
                                vmap(vmap(get_linear_velocity))(states[:,:-1,4], states[:,:-1,2:4]), # humans_velocities,
                                infos["static_obstacles"][:,-1], # static_obstacles,
                                states[:,-1,:2], # robot_positions,
                                states[:,-1,4], # robot_orientations,
                                infos["robot_goal"], # robot_goals,
                        )
                        humans_visibility, _ = env.batch_object_visibility(
                                rc_humans_positions,
                                infos["humans_parameters"][:,:,0],
                                rc_obstacles,
                        )
                        humans_in_range = env.batch_humans_inside_lidar_range(
                                rc_humans_positions,
                                infos["humans_parameters"][:,:,0],
                        )
                        ## Save data to later add to the buffer
                        batch_robot_goals = batch_robot_goals.at[:,step].set(infos["robot_goal"])
                        batch_supervised_target["gt_poses"] = batch_supervised_target["gt_poses"].at[:,step].set(rc_humans_positions)
                        batch_supervised_target["gt_vels"] = batch_supervised_target["gt_vels"].at[:,step].set(rc_humans_velocities)
                        batch_supervised_target["gt_mask"] = batch_supervised_target["gt_mask"].at[:,step].set(humans_visibility & humans_in_range)
                        batch_observations = batch_observations.at[:,step].set(obses)
                        batch_values = batch_values.at[:,step].set(values)
                        batch_actions = batch_actions.at[:,step].set(sampled_actions)
                        batch_rewards = batch_rewards.at[:,step].set(rewards)
                        batch_dones = batch_dones.at[:,step].set(~(init_outcomes["nothing"])) # Pay attention to this: we are using the initial outcomes to compute the dones
                        batch_neglogpdfs = batch_neglogpdfs.at[:,step].set(policy.dirichlet.batch_neglogp(actor_distrs, sampled_actions))
                        ## Compute dones and auxiliary data
                        successes += jnp.sum(outcomes["success"])
                        failures += jnp.sum(outcomes["collision_with_human"] | outcomes["collision_with_obstacle"])
                        timeouts += jnp.sum(outcomes["timeout"])
                        # returns = returns.at[:].set(returns + rewards * jnp.power(jnp.broadcast_to(policy.gamma,(n_parallel_envs,)), policy.v_max * (infos["time"] - policy.dt)))  
                        returns = returns.at[:].set(returns + new_infos["return"] * (~outcomes["nothing"]))  
                        episode_count += jnp.sum(~(outcomes["nothing"]))
                        batch_stds = batch_stds.at[:,step,:].set(policy.dirichlet.batch_std(actor_distrs))
                        return (network_params, new_states, new_obses, new_infos, outcomes, policy_keys, reset_keys, episode_count, batch_robot_goals, batch_supervised_target, batch_observations, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, successes, failures, timeouts, returns, batch_stds)
                batch_robot_goals = jnp.zeros((n_parallel_envs, n_steps, 2))
                batch_supervised_target = {
                        "gt_poses": jnp.zeros((n_parallel_envs, n_steps, env.n_humans, 2)),
                        "gt_vels": jnp.zeros((n_parallel_envs, n_steps, env.n_humans, 2)),
                        "gt_mask": jnp.zeros((n_parallel_envs, n_steps, env.n_humans), dtype=bool),
                }
                batch_observations = jnp.zeros((n_parallel_envs, n_steps, policy.n_stack, policy.lidar_num_rays + 6))
                batch_values = jnp.zeros((n_parallel_envs, n_steps))
                batch_actions = jnp.zeros((n_parallel_envs, n_steps, 2))
                batch_rewards = jnp.zeros((n_parallel_envs, n_steps))
                batch_dones = jnp.zeros((n_parallel_envs, n_steps))
                batch_neglogpdfs = jnp.zeros((n_parallel_envs, n_steps))
                batch_stds = jnp.zeros((n_parallel_envs, n_steps, 2)) # (WARNING: could be state dependent or not)
                val_init = (network_params, states, obses, infos, init_outcomes, policy_keys, reset_keys, 0, batch_robot_goals, batch_supervised_target, batch_observations, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, 0, 0, 0, jnp.zeros(n_parallel_envs,), batch_stds)
                network_params, states, obses, infos, outcomes, policy_keys, reset_keys, episode_count, batch_robot_goals, batch_supervised_target, batch_observations, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, succ_count, fail_count, timeo_count, cum_rewards, batch_stds = lax.fori_loop(
                        0,
                        n_steps, 
                        _step_fori_body, 
                        val_init
                )           
                ### Save auxiliary data
                aux_data["successes"] = aux_data["successes"].at[upd_idx].set(succ_count)
                aux_data["failures"] = aux_data["failures"].at[upd_idx].set(fail_count)
                aux_data["timeouts"] = aux_data["timeouts"].at[upd_idx].set(timeo_count)
                # aux_data["returns"] = aux_data["returns"].at[upd_idx].set(jnp.mean(cum_rewards))
                aux_data["returns"] = aux_data["returns"].at[upd_idx].set(jnp.sum(cum_rewards)/episode_count)
                aux_data["episodes"] = aux_data["episodes"].at[upd_idx].set(episode_count)
                aux_data["stds"] = aux_data["stds"].at[upd_idx].set(jnp.mean(batch_stds, axis=(0,1)))
                ### Add experiences to the buffer
                # Compute the value of the last batched_states and dones
                perception_inputs, robot_state_inputs = vmap(policy.compute_e2e_input, in_axes=(0,0))(
                        obses,
                        infos['robot_goal']
                )
                # Compute the value of the last batch  
                _, _, _, _, _ , _, last_values, _ = policy.e2e.apply(
                        network_params,
                        None,
                        perception_inputs,
                        robot_state_inputs
                )
                last_dones = ~(outcomes["nothing"])
                batch_values = jnp.append(batch_values, last_values[:,None], axis=1)
                batch_dones = jnp.column_stack((batch_dones, last_dones))
                # Compute advatages with GAE
                @jit
                def _gae_fori_loop(t:int, val:tuple):
                        rt = n_steps - t - 1 # reverse the time index
                        # Retrieve data from the tuple
                        values, rewards, dones, advantages = val
                        # Compute the advantage
                        delta = rewards[:,rt] + pow(policy.gamma, policy.dt * policy.v_max) * values[:,rt+1] * (1 - dones[:,rt+1]) - values[:,rt]
                        advantages = advantages.at[:,rt].set(delta + pow(policy.gamma * lambda_gae, policy.dt * policy.v_max) * advantages[:,rt+1] * (1 - dones[:,rt+1]))
                        return values, rewards, dones, advantages
                _, _, _, advatages = lax.fori_loop(
                        0,
                        n_steps,
                        _gae_fori_loop,
                        (batch_values, batch_rewards, batch_dones, jnp.zeros((n_parallel_envs, n_steps+1))),
                )
                critic_targets = advatages[:,:-1] + batch_values[:,:-1] # These are the returns for each state
                ## Debugging
                # debug.print("Advantages: {x}", x=advatages[0,:-1])
                # debug.print("Critic estimates: {x}", x=batch_values[0,:-1])
                # debug.print("Returns: {x}", x=critic_targets[0])
                # debug.print("Dones: {x}", x=batch_dones[0,:-1])
                # debug.print("Batch actions: {x}", x=batch_actions[0])
                # debug.print("Batch neglogp: {x}", x=batch_neglogpdfs[0])
                # Add all experiences to the buffer
                training_data = {
                        "robot_goals": jnp.reshape(batch_robot_goals, (n_parallel_envs*n_steps, 2)),
                        "gt_mask": jnp.reshape(batch_supervised_target["gt_mask"], (n_parallel_envs*n_steps, env.n_humans)),
                        "gt_poses": jnp.reshape(batch_supervised_target["gt_poses"], (n_parallel_envs*n_steps, env.n_humans, 2)),
                        "gt_vels": jnp.reshape(batch_supervised_target["gt_vels"], (n_parallel_envs*n_steps, env.n_humans, 2)),
                        "observations": jnp.reshape(batch_observations, (n_parallel_envs*n_steps, policy.n_stack, policy.lidar_num_rays + 6)),
                        "critic_targets": jnp.reshape(critic_targets, (n_parallel_envs*n_steps,)),
                        "actions": jnp.reshape(batch_actions, (n_parallel_envs*n_steps, 2)),
                        "values": jnp.reshape(batch_values[:,:-1], (n_parallel_envs*n_steps,)),
                        "neglogpdfs": jnp.reshape(batch_neglogpdfs, (n_parallel_envs*n_steps,)),
                }
                ### Update model parameters
                @jit
                def _epoch_body(epoch:int, epoch_val:tuple):
                        # Retrieve data from the tuple
                        shuffle_key, network_params, net_opt_state, training_data, pre_aux_data = epoch_val
                        # Shuffle the buffer
                        shuffle_key, subkey = random.split(shuffle_key)
                        indexes = jnp.arange(total_batch_size)
                        shuffled_indexes = random.permutation(subkey, indexes)
                        training_data = tree_map(lambda x: x[shuffled_indexes], training_data)
                        # Compute the number of mini-batches
                        @jit
                        def _batch_body(batch:int, batch_val:tuple):
                                # Retrieve data from the tuple
                                shuffle_key, network_params, net_opt_state, training_data, pre_aux_data = batch_val
                                # Retrieve data from the buffer
                                indexes = (jnp.arange(mini_batch_size) + batch * mini_batch_size).astype(jnp.int32)
                                experiences = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, training_data)
                                experiences["inputs0"], experiences["inputs1"] = vmap(policy.compute_e2e_input, in_axes=(0,0))(
                                        experiences["observations"],
                                        experiences["robot_goals"],
                                )
                                # Debugging
                                update_debug = (debugging) & (upd_idx % debugging_interval == 0) & (epoch + batch == 0) & (upd_idx != 0)
                                lax.cond(
                                        update_debug,
                                        lambda _: debug.print("\n### Update {x}", x=upd_idx),
                                        lambda _: None,
                                        None,
                                )
                                # Update the networks
                                new_network_params, net_opt_state, loss, perception_loss, critic_loss, actor_loss, entropy_loss, loss_stds = policy.update(
                                        network_params,
                                        network_optimizer,
                                        net_opt_state,
                                        experiences,
                                        beta_entropy,
                                        clip_range,
                                        debugging = False, #(debugging) & (upd_idx % debugging_interval == 0) & (epoch + batch == 0), 
                                )
                                # Save aux data
                                pre_aux_data["losses"] = pre_aux_data["losses"].at[epoch,batch].set(loss)
                                pre_aux_data["actor_losses"] = pre_aux_data["actor_losses"].at[epoch,batch].set(actor_loss)
                                pre_aux_data["critic_losses"] = pre_aux_data["critic_losses"].at[epoch,batch].set(critic_loss)
                                pre_aux_data["entropy_losses"] = pre_aux_data["entropy_losses"].at[epoch,batch].set(entropy_loss)
                                pre_aux_data["perception_losses"] = pre_aux_data["perception_losses"].at[epoch,batch].set(perception_loss)
                                pre_aux_data["loss_stds"] = pre_aux_data["loss_stds"].at[epoch,batch].set(loss_stds)
                                return shuffle_key, new_network_params, net_opt_state, training_data, pre_aux_data
                        epoch_val = lax.fori_loop(
                                0,
                                n_batches,
                                _batch_body,
                                (shuffle_key, network_params, net_opt_state, training_data, pre_aux_data)
                        )
                        return epoch_val
                pre_aux_data = {
                        "losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "actor_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "critic_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "entropy_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "perception_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "loss_stds": jnp.zeros((n_epochs,n_batches, 3), dtype=jnp.float32),
                }
                shuffle_key, network_params, net_opt_state, training_data, pre_aux_data = lax.fori_loop(
                        0,
                        n_epochs,
                        _epoch_body,
                        (shuffle_key, network_params, net_opt_state, training_data, pre_aux_data)
                )
                # Save data
                aux_data["losses"] = aux_data["losses"].at[upd_idx].set(jnp.mean(pre_aux_data["losses"], axis=(0,1)))
                aux_data["actor_losses"] = aux_data["actor_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["actor_losses"], axis=(0,1)))
                aux_data["critic_losses"] = aux_data["critic_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["critic_losses"], axis=(0,1)))
                aux_data["entropy_losses"] = aux_data["entropy_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["entropy_losses"], axis=(0,1)))
                aux_data["perception_losses"] = aux_data["perception_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["perception_losses"], axis=(0,1)))
                aux_data["loss_stds"] = aux_data["loss_stds"].at[upd_idx].set(jnp.mean(pre_aux_data["loss_stds"], axis=(0,1)))
                lax.cond(
                        (debugging) & (upd_idx % debugging_interval == 0) & (upd_idx != 0),   
                        lambda _: debug.print(
                                "Episodes {w}\nLoss: {l}\nPerception loss: {p}\nActor loss: {y}\nCritic loss: {z}\nLoss stds: {llv}\nEntropy: {e}\nActor Std: {std}\nReturn: {r}\nSucc.Rate: {s}\nFail.Rate: {f}\nTim.Rate: {t}", 
                                l=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["losses"])) <= upd_idx), aux_data["losses"], jnp.nan)),
                                p=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["perception_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["perception_losses"])) <= upd_idx), aux_data["perception_losses"], jnp.nan)),
                                y=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["actor_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["actor_losses"])) <= upd_idx), jnp.abs(aux_data["actor_losses"]), jnp.nan)),
                                z=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["critic_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["critic_losses"])) <= upd_idx), aux_data["critic_losses"], jnp.nan)), 
                                llv=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["loss_stds"]))[:,None] > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["loss_stds"]))[:,None] <= upd_idx), aux_data["loss_stds"], jnp.array([jnp.nan,jnp.nan,jnp.nan])), axis=0),
                                w=jnp.sum(aux_data["episodes"]),
                                r=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["returns"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["returns"])) <= upd_idx), aux_data["returns"], jnp.nan)),
                                s=jnp.nansum(jnp.where((jnp.arange(len(aux_data["successes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["successes"])) <= upd_idx), aux_data["successes"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                f=jnp.nansum(jnp.where((jnp.arange(len(aux_data["failures"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["failures"])) <= upd_idx), aux_data["failures"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                t=jnp.nansum(jnp.where((jnp.arange(len(aux_data["timeouts"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["timeouts"])) <= upd_idx), aux_data["timeouts"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                e=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["entropy_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["entropy_losses"])) <= upd_idx), aux_data["entropy_losses"], jnp.nan)),
                                std=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["stds"]))[:,None] > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["stds"]))[:,None] <= upd_idx), aux_data["stds"], jnp.array([jnp.nan,jnp.nan])), axis=0),              
                                ),
                        lambda x: x, 
                        None
                )
                return policy_keys, reset_keys, shuffle_key, states, obses, infos, outcomes, network_params, net_opt_state, current_buffer_size, aux_data

        # Initialize the auxiliary data array
        aux_data = {
                "losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "perception_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "actor_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "critic_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "entropy_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "loss_stds": jnp.zeros([train_updates, 3], dtype=jnp.float32),
                "returns": jnp.zeros([train_updates], dtype=jnp.float32),
                "successes": jnp.zeros([train_updates], dtype=int),
                "failures": jnp.zeros([train_updates], dtype=int),
                "timeouts": jnp.zeros([train_updates], dtype=int),
                "episodes": jnp.zeros([train_updates], dtype=int),
                "stds": jnp.zeros([train_updates, 2], dtype=jnp.float32),
        }
        new_outcomes_acc = {k: outcomes_acc[k] + outcomes[k] for k in outcomes}
        return (new_states, new_obses, new_infos, new_p_keys, new_r_keys, new_outcomes_acc), step_data
    
    init_outcomes_acc = {
        k: jnp.zeros_like(template_outcomes[k], dtype=jnp.int32) 
        for k in template_outcomes
    }
    init_carry = (env_state[0], env_state[1], env_state[2], policy_keys, reset_keys, init_outcomes_acc)
    final_carry, history = lax.scan(_scan_step, init_carry, None, length=n_steps)
    (final_states, final_obses, final_infos, final_p_keys, final_r_keys, sum_outcomes) = final_carry
    next_env_state = (final_states, final_obses, final_infos)
    return next_env_state, final_p_keys, final_r_keys, history, sum_outcomes

@partial(jit, static_argnames=("policy",), donate_argnums=(3,))
def process_buffer_and_gae(
    network_params, 
    last_obs, 
    last_info, 
    history, 
    policy, 
    gamma, 
    lambda_gae
):
    """
    Calcola l'ultimo value, GAE, Returns e appiattisce il buffer.
    """
    perception_inputs, robot_state_inputs = vmap(policy.compute_e2e_input, in_axes=(0,0))(
        last_obs, last_info['robot_goal']
    )
    _, _, _, _, _ , _, last_values, _ = policy.e2e.apply(
        network_params, None, perception_inputs, robot_state_inputs
    )
    rewards = history["rewards"]
    values = history["values"]
    dones = history["dones"]
    values_ext = jnp.concatenate([values, last_values[None, :]], axis=0)
    def _gae_step(gae_carry, i):
        adv_next = gae_carry
        mask = 1.0 - dones[i].astype(jnp.float32)
        delta = rewards[i] + gamma * values_ext[i+1] * mask - values_ext[i]
        advantage = delta + gamma * lambda_gae * adv_next * mask
        return advantage, advantage # Carry, Output
    n_steps = rewards.shape[0]
    _, advantages = lax.scan(_gae_step, jnp.zeros_like(values[0]), jnp.arange(n_steps)[::-1])
    advantages = advantages[::-1]
    critic_targets = advantages + values
    def flatten(x):
        return jnp.reshape(x, (-1, *x.shape[2:]))
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
        "advantages": flatten(advantages),
    }
    flat_adv = flattened_buffer["advantages"]
    flattened_buffer["advantages"] = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)
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
            (perc_dist, _, _, actor_dist, _, _, pred_val, log_vars) = policy.e2e.apply(
                p, None, inputs0, inputs1
            )
            # Actor
            new_neglogp = vmap(policy.dirichlet.neglogp)(actor_dist, u_mb["actions"])
            ratio = jnp.exp(u_mb["neglogpdfs"] - new_neglogp)
            surr1 = ratio * u_mb["advantages"]
            surr2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * u_mb["advantages"]
            actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            # Critic
            v_loss = jnp.square(pred_val - u_mb["critic_targets"])
            v_clipped = u_mb["values"] + jnp.clip(pred_val - u_mb["values"], -clip_range, clip_range)
            v_loss_clipped = jnp.square(v_clipped - u_mb["critic_targets"])
            critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss, v_loss_clipped))
            # Entropy
            entropy = jnp.mean(policy.dirichlet.entropy(actor_dist))
            entropy_loss = -beta_entropy * entropy
            # Perception
            gt_dict = {"gt_mask": u_mb["gt_mask"], "gt_poses": u_mb["gt_poses"], "gt_vels": u_mb["gt_vels"]}
            batch_perc_loss = policy._encoder_loss(perc_dist, gt_dict)
            perception_loss = jnp.mean(batch_perc_loss)
            # Total
            w_actor = jnp.exp(-log_vars[0]) * actor_loss + log_vars[0]
            w_critic = jnp.exp(-log_vars[1]) * critic_loss + log_vars[1]
            w_perc = jnp.exp(-log_vars[2]) * perception_loss + log_vars[2]
            total_loss = 0.5 * (w_actor + w_critic + w_perc) + entropy_loss
            return total_loss, (actor_loss, critic_loss, perception_loss, entropy_loss, log_vars)

        def _micro_step_scan(carry, u_mb):
            current_grads_acc, current_metrics_acc = carry
            (loss, aux), grads = value_and_grad(micro_batch_loss_fn, has_aux=True)(params_inner, u_mb)
            new_grads_acc = tree_map(lambda acc, g: acc + g, current_grads_acc, grads)
            l_act, l_crit, l_perc, l_ent, l_log = aux
            acc_loss, (acc_act, acc_crit, acc_perc, acc_ent, acc_log) = current_metrics_acc
            new_metrics_acc = (
                acc_loss + loss,
                (acc_act + l_act, acc_crit + l_crit, acc_perc + l_perc, acc_ent + l_ent, acc_log + l_log)
            )
            return (new_grads_acc, new_metrics_acc), None

        grads_acc_init = tree_map(jnp.zeros_like, params_inner)
        metrics_acc_init = (0.0, (0.0, 0.0, 0.0, 0.0, jnp.zeros((3,))))
        (grads_sum, metrics_sum), _ = lax.scan(
            _micro_step_scan, 
            (grads_acc_init, metrics_acc_init), 
            micro_batches
        )
        grads_avg = tree_map(lambda x: x / n_micro_splits, grads_sum)
        loss_sum, (act_sum, crit_sum, perc_sum, ent_sum, log_sum) = metrics_sum
        loss_avg = loss_sum / n_micro_splits
        aux_avg = (
            act_sum / n_micro_splits,
            crit_sum / n_micro_splits,
            perc_sum / n_micro_splits,
            ent_sum / n_micro_splits,
            log_sum / n_micro_splits
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
        "log_vars": jnp.mean(batch_aux[4], axis=0)
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
    n_steps = total_batch_size // n_parallel_envs
    devices = mesh_utils.create_device_mesh((device_count(),))
    mesh = Mesh(devices, axis_names=('env_axis',)) 
    sharding_env = NamedSharding(mesh, PartitionSpec('env_axis'))
    sharding_replicated = NamedSharding(mesh, PartitionSpec())
    key = random.PRNGKey(random_seed)
    key, subkey = random.split(key)
    reset_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    states, reset_keys, obses, infos, init_outcomes = env.batch_reset(reset_keys)
    states = tree_map(lambda x: device_put(x, sharding_env), states)
    obses = tree_map(lambda x: device_put(x, sharding_env), obses)
    infos = tree_map(lambda x: device_put(x, sharding_env), infos)
    init_outcomes = tree_map(lambda x: device_put(x, sharding_env), init_outcomes)
    env_state = (states, obses, infos)
    key, subkey = random.split(key)
    policy_keys = device_put(random.split(subkey, n_parallel_envs), sharding_env)
    params = initial_network_params
    opt_state = network_optimizer.init(params)
    params = device_put(params, sharding_replicated)
    opt_state = device_put(opt_state, sharding_replicated)
    logs = {
        "losses": [], "returns": [], "successes": [], "failures": [], "timeouts": [],
        "episodes": [], "perception_losses": [], "actor_losses": [], "critic_losses": [],
        "entropy_losses": [], "loss_stds": [], "stds": []
    }
    print(f"Starting optimized training loop for {train_updates} updates.")
    print(f"Rollout distributed across {len(devices)} devices.")
    for update in tqdm(range(train_updates)):
        # A. COLLECT ROLLOUT STEP (Parallel)
        env_state, policy_keys, reset_keys, history_raw, outcomes_sum = collect_rollout_step(
            params, env_state, policy_keys, reset_keys, init_outcomes, policy, env, n_steps
        )
        batch_mean_return = float(jnp.mean(jnp.sum(history_raw["rewards"], axis=0)))
        avg_action_std = device_get(jnp.mean(history_raw["stds"], axis=(0,1)))
        # B. PROCESS BUFFER (Parallel)
        current_obs, current_infos = env_state[1], env_state[2]
        buffer_gpu = process_buffer_and_gae(
            params, current_obs, current_infos, history_raw, policy, policy.gamma, lambda_gae
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
            "loss": [], "perc": [], "actor": [], "critic": [], "entropy": [], "log_vars": []
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
                epoch_metrics_acc[k].append(metrics_one_epoch[k])
        # F. LOGGING
        logs["losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["loss"]))))
        logs["perception_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["perc"]))))
        logs["actor_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["actor"]))))
        logs["critic_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["critic"]))))
        logs["entropy_losses"].append(float(jnp.mean(jnp.stack(epoch_metrics_acc["entropy"]))))
        avg_log_vars = jnp.mean(jnp.stack(epoch_metrics_acc["log_vars"]), axis=0)
        logs["loss_stds"].append(jnp.exp(0.5 * avg_log_vars))
        logs["returns"].append(batch_mean_return)
        n_succ = jnp.sum(outcomes_sum["success"])
        n_coll_hum = jnp.sum(outcomes_sum["collision_with_human"])
        n_coll_obs = jnp.sum(outcomes_sum["collision_with_obstacle"])
        n_timeout = jnp.sum(outcomes_sum["timeout"])
        ep_count = n_succ + n_coll_hum + n_coll_obs + n_timeout
        n_fail = n_coll_hum + n_coll_obs
        logs["successes"].append(int(n_succ))
        logs["failures"].append(int(n_fail))
        logs["timeouts"].append(int(n_timeout))
        logs["episodes"].append(int(ep_count))
        logs["stds"].append(avg_action_std)
        if update % 1 == 0:
             print(f"Upd {update} | Ret: {logs['returns'][-1]:.2f} | Loss: {logs['losses'][-1]:.4f} | Succ: {logs['successes'][-1]} | Fail: {logs['failures'][-1]} | Timeouts: {logs['timeouts'][-1]}")
    return params, logs 
             