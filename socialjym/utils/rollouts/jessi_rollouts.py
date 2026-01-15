import optax
from jax import jit, lax, random, vmap, debug
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax_tqdm import loop_tqdm, scan_tqdm

from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi import JESSI
from socialjym.envs.base_env import HUMAN_POLICIES
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
        # Initialize the optimizer state
        network_optimizer_state = network_optimizer.init(initial_network_params)
        # Initialize the random keys
        policy_keys = vmap(random.PRNGKey)(jnp.arange(n_parallel_envs, dtype=int) + random_seed)
        reset_keys = vmap(random.PRNGKey)(jnp.arange(n_parallel_envs, dtype=int) + random_seed)
        shuffle_key = random.PRNGKey(random_seed)
        # Initialize environments
        states, reset_keys, obses, infos, init_outcomes = env.batch_reset(reset_keys)
        # Create initial values for training loop
        val_init = (
                policy_keys,
                reset_keys,
                shuffle_key,
                states,
                obses,
                infos,
                init_outcomes,
                initial_network_params, 
                network_optimizer_state,
                0,
                aux_data
        )
        # Execute the training loop
        debug.print("Performing RL training for {x} updates...", x=train_updates)
        vals = lax.fori_loop(
                0,
                train_updates, 
                _update_for_loop_body, 
                val_init
        )
        output_dict = {}
        keys = [
                "policy_keys",
                "reset_keys",
                "shuffle_key",
                "states",
                "obses",
                "infos",
                "outcomes",
                "network_params", 
                "network_optimizer_state",
                "current_buffer_size",
                "aux_data"
        ]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value
        return output_dict