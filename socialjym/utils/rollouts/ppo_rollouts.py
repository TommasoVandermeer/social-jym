import haiku as hk
import optax
from typing import Callable
from jax import jit, lax, random, vmap, debug, nn
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
import pickle

from socialjym.envs.base_env import BaseEnv
from socialjym.utils.distributions.base_distribution import DISTRIBUTIONS
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer


def ppo_rl_rollout(
        initial_actor_params: dict,
        initial_critic_params: dict,
        n_parallel_envs:int,
        train_updates: int, # Number of episodes to train on, exploration episodes excluded
        random_seed: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        buffer_state: dict,
        buffer_capacity: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: PPOBuffer,
        clip_range: float,
        n_epochs: int,
        beta_entropy: float,
        lambda_gae: float,
        debugging: bool = False,
        debugging_interval: int = 100,
) -> dict:
        
        assert policy.name == "SARL-PPO", "This function is only compatible with PPO policies."
        # Compute number of steps to simulate per update for each parallel env
        assert buffer_capacity % n_parallel_envs == 0, "The buffer capacity must be a multiple of the number of parallel environments. Otherwise you will trow away experiences."
        assert buffer_capacity == replay_buffer.buffer_size, "The buffer capacity must be equal to the buffer size of the replay buffer."
        n_steps = int(buffer_capacity / n_parallel_envs)
        n_batches = int(buffer_capacity / replay_buffer.batch_size)

        # Define the main rollout episode loop
        @loop_tqdm(train_updates)
        @jit
        def _update_for_loop_body(upd_idx:int, val:tuple):
                # Retrieve data from the tuple
                policy_keys, reset_keys, shuffle_key, states, obses, infos, outcomes, actor_params, critic_params, act_opt_state, cri_opt_state, buffer_state, current_buffer_size, aux_data = val
                ## Generate experience
                @jit
                def _step_fori_body(step:int, val:tuple):
                        # Retrieve data from the tuple
                        actor_params, critic_params, states, obses, infos, init_outcomes, policy_keys, reset_keys, episode_count, batch_inputs, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, successes, failures, timeouts, returns, batch_stds = val
                        ## Step
                        actions, policy_keys, inputs, sampled_actions, distrs = policy.batch_act(policy_keys, obses, infos, actor_params, sample=True)
                        # Compute state values
                        values = vmap(policy.critic.apply, in_axes=(None,None,0))(
                                critic_params, 
                                None, 
                                inputs, 
                        ) 
                        states, obses, infos, rewards, outcomes, reset_keys = env.batch_step(states, infos, actions, reset_keys, test=False, reset_if_done=True)
                        ## Save data to later add to the buffer
                        batch_inputs = batch_inputs.at[:,step].set(inputs)
                        batch_values = batch_values.at[:,step].set(jnp.squeeze(values))
                        batch_actions = batch_actions.at[:,step].set(sampled_actions)
                        batch_rewards = batch_rewards.at[:,step].set(rewards)
                        batch_dones = batch_dones.at[:,step].set(~(init_outcomes["nothing"])) # Pay attention to this: we are using the initial outcomes to compute the dones
                        batch_neglogpdfs = batch_neglogpdfs.at[:,step].set(policy.distr.batch_neglogp(distrs, sampled_actions))
                        ## Compute dones and auxiliary data
                        successes += jnp.sum(outcomes["success"])
                        failures += jnp.sum(outcomes["failure"])
                        timeouts += jnp.sum(outcomes["timeout"])
                        returns = returns.at[:].set(returns + rewards * jnp.power(jnp.broadcast_to(policy.gamma,(n_parallel_envs,)), policy.v_max * (infos["time"] - policy.dt)))  
                        episode_count += jnp.sum(~(outcomes["nothing"]))
                        batch_stds = batch_stds.at[:,step,:].set(policy.distr.batch_std(distrs))
                        return (actor_params, critic_params, states, obses, infos, outcomes, policy_keys, reset_keys, episode_count, batch_inputs, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, successes, failures, timeouts, returns, batch_stds)
                batch_inputs = jnp.zeros((n_parallel_envs, n_steps, env.n_humans, policy.vnet_input_size))
                batch_values = jnp.zeros((n_parallel_envs, n_steps))
                batch_actions = jnp.zeros((n_parallel_envs, n_steps, 2))
                batch_rewards = jnp.zeros((n_parallel_envs, n_steps))
                batch_dones = jnp.zeros((n_parallel_envs, n_steps))
                batch_neglogpdfs = jnp.zeros((n_parallel_envs, n_steps))
                batch_stds = jnp.zeros((n_parallel_envs, n_steps, 2)) # (WARNING: could be state dependent or not)
                val_init = (actor_params, critic_params, states, obses, infos, init_outcomes, policy_keys, reset_keys, 0, batch_inputs, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, 0, 0, 0, jnp.zeros(n_parallel_envs,), batch_stds)
                actor_params, critic_params, states, obses, infos, outcomes, policy_keys, reset_keys, episode_count, batch_inputs, batch_values, batch_actions, batch_rewards, batch_dones, batch_neglogpdfs, succ_count, fail_count, timeo_count, cum_rewards, batch_stds = lax.fori_loop(
                        0,
                        n_steps, 
                        _step_fori_body, 
                        val_init
                )           
                ### Save auxiliary data
                aux_data["successes"] = aux_data["successes"].at[upd_idx].set(succ_count)
                aux_data["failures"] = aux_data["failures"].at[upd_idx].set(fail_count)
                aux_data["timeouts"] = aux_data["timeouts"].at[upd_idx].set(timeo_count)
                aux_data["returns"] = aux_data["returns"].at[upd_idx].set(jnp.mean(cum_rewards))
                aux_data["episodes"] = aux_data["episodes"].at[upd_idx].set(episode_count)
                aux_data["stds"] = aux_data["stds"].at[upd_idx].set(jnp.mean(batch_stds, axis=(0,1)))
                ### Add experiences to the buffer
                # Compute the value of the last batched_states and dones
                last_values = vmap(policy.critic.apply, in_axes=(None,None,0))(
                        critic_params,
                        None,
                        vmap(policy.batch_compute_vnet_input, in_axes=(0,0,0))(
                                obses[:,-1], 
                                obses[:,:-1], 
                                infos
                        ),
                )
                last_dones = ~(outcomes["nothing"])
                batch_values = jnp.append(batch_values, last_values, axis=1)
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
                buffer_state = replay_buffer.batch_add(
                        buffer_state,
                        jnp.reshape(batch_inputs, (n_parallel_envs*n_steps, env.n_humans, policy.vnet_input_size)),
                        jnp.reshape(critic_targets, (n_parallel_envs*n_steps,)),
                        jnp.reshape(batch_actions, (n_parallel_envs*n_steps, 2)),
                        jnp.reshape(batch_values[:,:-1], (n_parallel_envs*n_steps,)),
                        jnp.reshape(batch_neglogpdfs, (n_parallel_envs*n_steps,)),
                        jnp.arange(n_parallel_envs*n_steps) + current_buffer_size,
                )
                current_buffer_size += n_parallel_envs * n_steps
                ### Update model parameters
                @jit
                def _epoch_body(epoch:int, epoch_val:tuple):
                        # Retrieve data from the tuple
                        shuffle_key, actor_params, critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data = epoch_val
                        # Shuffle the buffer
                        buffer_state, shuffle_key = replay_buffer.shuffle(buffer_state, shuffle_key)
                        # Compute the number of mini-batches
                        @jit
                        def _batch_body(batch:int, batch_val:tuple):
                                # Retrieve data from the tuple
                                shuffle_key, actor_params, critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data = batch_val
                                # Retrieve data from the buffer
                                experiences = replay_buffer.iterate(buffer_state, current_buffer_size, batch)
                                # Debugging
                                update_debug = (debugging) & (upd_idx % debugging_interval == 0) & (epoch + batch == 0) & (upd_idx != 0)
                                lax.cond(
                                        update_debug,
                                        lambda _: debug.print("\n### Update {x}", x=upd_idx),
                                        lambda _: None,
                                        None,
                                )
                                # Update the networks
                                new_critic_params, new_actor_params, cri_opt_state, act_opt_state, critic_loss, actor_loss, entropy_loss = policy.update(
                                        critic_params,
                                        actor_params,
                                        actor_optimizer,
                                        act_opt_state,
                                        critic_optimizer,
                                        cri_opt_state,
                                        experiences,
                                        beta_entropy,
                                        clip_range,
                                        debugging = False, #(debugging) & (upd_idx % debugging_interval == 0) & (epoch + batch == 0), 
                                )
                                # Save aux data
                                pre_aux_data["actor_losses"] = pre_aux_data["actor_losses"].at[epoch,batch].set(actor_loss)
                                pre_aux_data["critic_losses"] = pre_aux_data["critic_losses"].at[epoch,batch].set(critic_loss)
                                pre_aux_data["entropy_losses"] = pre_aux_data["entropy_losses"].at[epoch,batch].set(entropy_loss)
                                return shuffle_key, new_actor_params, new_critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data
                        epoch_val = lax.fori_loop(
                                0,
                                n_batches,
                                _batch_body,
                                (shuffle_key, actor_params, critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data)
                        )
                        return epoch_val
                pre_aux_data = {
                        "actor_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "critic_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                        "entropy_losses": jnp.zeros((n_epochs,n_batches), dtype=jnp.float32),
                }
                shuffle_key, actor_params, critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data = lax.fori_loop(
                        0,
                        n_epochs,
                        _epoch_body,
                        (shuffle_key, actor_params, critic_params, cri_opt_state, act_opt_state, buffer_state, current_buffer_size, pre_aux_data)
                )
                # Empty the buffer
                buffer_state = replay_buffer.empty(buffer_state)
                current_buffer_size = 0
                # Save data
                aux_data["actor_losses"] = aux_data["actor_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["actor_losses"], axis=(0,1)))
                aux_data["critic_losses"] = aux_data["critic_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["critic_losses"], axis=(0,1)))
                aux_data["entropy_losses"] = aux_data["entropy_losses"].at[upd_idx].set(jnp.mean(pre_aux_data["entropy_losses"], axis=(0,1)))
                # DEBUG: print the mean loss for this episode and the return
                lax.cond(
                        (debugging) & (upd_idx % debugging_interval == 0) & (upd_idx != 0),   
                        lambda _: debug.print(
                                "Episodes {w}\nActor loss: {y}\nCritic loss: {z}\nEntropy: {e}\nStd: {std}\nReturn: {r}\nSucc.Rate: {s}\nFail.Rate: {f}\nTim.Rate: {t}", 
                                y=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["actor_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["actor_losses"])) <= upd_idx), jnp.abs(aux_data["actor_losses"]), jnp.nan)),
                                z=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["critic_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["critic_losses"])) <= upd_idx), aux_data["critic_losses"], jnp.nan)), 
                                w=jnp.sum(aux_data["episodes"]),
                                r=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["returns"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["returns"])) <= upd_idx), aux_data["returns"], jnp.nan)),
                                s=jnp.nansum(jnp.where((jnp.arange(len(aux_data["successes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["successes"])) <= upd_idx), aux_data["successes"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                f=jnp.nansum(jnp.where((jnp.arange(len(aux_data["failures"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["failures"])) <= upd_idx), aux_data["failures"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                t=jnp.nansum(jnp.where((jnp.arange(len(aux_data["timeouts"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["timeouts"])) <= upd_idx), aux_data["timeouts"], jnp.nan)) / jnp.nansum(jnp.where((jnp.arange(len(aux_data["episodes"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["episodes"])) <= upd_idx), aux_data["episodes"], jnp.nan)),
                                e=jnp.nanmean(jnp.where((jnp.arange(len(aux_data["entropy_losses"])) > upd_idx-debugging_interval) & (jnp.arange(len(aux_data["entropy_losses"])) <= upd_idx), aux_data["entropy_losses"], jnp.nan)),
                                std=jnp.nanmean(jnp.where(((jnp.column_stack([jnp.arange(aux_data["stds"].shape[0]),jnp.arange(aux_data["stds"].shape[0])])) > upd_idx-debugging_interval) & (jnp.column_stack([jnp.arange(aux_data["stds"].shape[0]),jnp.arange(aux_data["stds"].shape[0])]) <= upd_idx), aux_data["stds"], jnp.array([jnp.nan,jnp.nan])), axis=0),
                        ),
                        lambda x: x, 
                        None
                )
                return policy_keys, reset_keys, shuffle_key, states, obses, infos, outcomes, actor_params, critic_params, act_opt_state, cri_opt_state, buffer_state, current_buffer_size, aux_data

        # Initialize the auxiliary data array
        aux_data = {
                "actor_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "critic_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "entropy_losses": jnp.zeros([train_updates], dtype=jnp.float32),
                "returns": jnp.zeros([train_updates], dtype=jnp.float32),
                "successes": jnp.zeros([train_updates], dtype=int),
                "failures": jnp.zeros([train_updates], dtype=int),
                "timeouts": jnp.zeros([train_updates], dtype=int),
                "episodes": jnp.zeros([train_updates], dtype=int),
                "stds": jnp.zeros([train_updates, 2], dtype=jnp.float32),
        }
        # Initialize the optimizer state
        actor_optimizer_state = actor_optimizer.init(initial_actor_params)
        critic_optimizer_state = critic_optimizer.init(initial_critic_params)
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
                initial_actor_params, 
                initial_critic_params, 
                actor_optimizer_state, 
                critic_optimizer_state, 
                buffer_state, 
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
                "actor_params", 
                "critic_params", 
                "actor_optimizer_state", 
                "critic_optimizer_state", 
                "buffer_state", 
                "current_buffer_size",
                "aux_data",
        ]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value
        return output_dict