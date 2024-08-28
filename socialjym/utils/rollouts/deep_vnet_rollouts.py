import haiku as hk
import optax
from typing import Callable
from jax import jit, lax, random, vmap
import jax.numpy as jnp
from jax_tqdm import loop_tqdm

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.policies.cadrl import CADRL
from socialjym.utils.replay_buffers.base_vnet_replay_buffer import BaseVNetReplayBuffer


def deep_vnet_rl_rollout(
        initial_vnet_params: dict,
        train_episodes: int,
        random_seed: int,
        model: hk.Transformed,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        buffer_size: int,
        num_batches: int,
        epsilon_decay_fn: Callable,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
) -> dict:
        
        # If policy is CADRL, the number of humans in the training environment must be 1
        if isinstance(policy, CADRL): assert env.n_humans == 1, "CADRL policy only supports training with one human."

        # Define the main rollout episode loop
        @loop_tqdm(train_episodes)
        @jit
        def _fori_body(i: int, val:tuple):
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, done, policy_key, buffer_state, current_buffer_size, cumulative_reward = val
                        # Step
                        epsilon = epsilon_decay_fn(epsilon_start, epsilon_end, i, decay_rate)
                        action, policy_key, vnet_input = policy.act(policy_key, obs, info, model_params, epsilon)
                        state, obs, info, reward, done = env.step(state, info, action)
                        # Update buffer state
                        cumulative_reward += policy.gamma **(info["time"] * policy.v_max) * reward
                        next_vnet_input = jnp.squeeze(policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info))
                        target = reward + (1 - done) * pow(policy.gamma, policy.dt * policy.v_max) * model.apply(model_params, None, next_vnet_input)
                        experience = (jnp.squeeze(vnet_input), target)
                        buffer_state = replay_buffer.add(buffer_state, experience, current_buffer_size)
                        current_buffer_size += 1
                        return (state, obs, info, done, policy_key, buffer_state, current_buffer_size, cumulative_reward)
                # Retrieve data from the tuple
                model_params, optimizer_state, buffer_state, current_buffer_size, policy_key, buffer_key, reset_key, losses, returns = val
                # Reset the environment
                state, reset_key, obs, info = env.reset(reset_key)
                # Episode loop
                val_init = (state, obs, info, False, policy_key, buffer_state, current_buffer_size, 0.)
                _, _, _, _, policy_key, buffer_state, current_buffer_size, cumulative_reward = lax.while_loop(lambda x: x[3] == False, _while_body, val_init)           
                # Update model parameters
                @jit
                def _model_update_fori_body(j:int, val:tuple):
                        # Retrieve data from the tuple
                        buffer_key, buffer_state, size, model_params, optimizer_state, losses = val
                        # Sample a batch of experiences from the replay buffer
                        experiences_batch, buffer_key = replay_buffer.sample(
                        buffer_key,
                        buffer_state,
                        size)
                        # Update the model parameters
                        model_params, optimizer_state, loss = policy.update(
                        model_params,
                        optimizer,
                        optimizer_state,
                        experiences_batch)
                        # Save the losses
                        losses = losses.at[j].set(loss)
                        return (buffer_key, buffer_state, size, model_params, optimizer_state, losses)
                all_losses = jnp.empty([num_batches])
                val_init = (buffer_key, buffer_state, jnp.min(jnp.array([current_buffer_size, buffer_size])), model_params, optimizer_state, all_losses)
                buffer_key, buffer_state, _, model_params, optimizer_state, all_losses = lax.fori_loop(0, num_batches,_model_update_fori_body, val_init)
                losses = losses.at[i].set(jnp.mean(all_losses))
                returns = returns.at[i].set(cumulative_reward)
                # Return the updated values
                val = (model_params, optimizer_state, buffer_state, current_buffer_size, policy_key, buffer_key, reset_key, losses, returns)
                return val

        # Initialize the random keys
        policy_key, buffer_key, reset_key = vmap(random.PRNGKey)(jnp.arange(3) + random_seed)
        # Initialize the array where to save data
        losses = jnp.empty([train_episodes], dtype=jnp.float32)
        returns = jnp.empty([train_episodes], dtype=jnp.float32)
        # Initialize the model parameters
        optimizer_state = optimizer.init(initial_vnet_params)
        # Create initial values for training loop
        val_init = (initial_vnet_params, optimizer_state, buffer_state, 0, policy_key, buffer_key, reset_key, losses, returns)
        # Execute the training loop
        vals = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        output_dict = {}
        keys = ["model_params", "optimizer_state", "buffer_state", "current_buffer_size", "policy_key", "buffer_key", "reset_key", "losses", "returns"]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value

        return output_dict

def deep_vnet_il_rollout(
        initial_vnet_params: dict,
        train_episodes: int,
        random_seed: int,
        model: hk.Transformed,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        buffer_size: int,
        num_epochs: int,
) -> dict:
        
        # If policy is CADRL, the number of humans in the training environment must be 1
        if isinstance(policy, CADRL): assert env.n_humans == 1, "CADRL policy only supports training with one human."

        # Define the main rollout episode loop
        @loop_tqdm(train_episodes)
        @jit
        def _fori_body(i: int, val:tuple):
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, done, policy_key, vnet_inputs, rewards, steps = val
                        # Step
                        vnet_input = policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
                        state, obs, info, reward, done = env.imitation_learning_step(state, info)
                        # Save data
                        vnet_inputs = vnet_inputs.at[steps].set(jnp.squeeze(vnet_input))
                        rewards = rewards.at[steps].set(reward)
                        steps += 1
                        return (state, obs, info, done, policy_key, vnet_inputs, rewards, steps)
                # Retrieve data from the tuple
                model_params, buffer_state, current_buffer_size, policy_key, reset_key, returns = val
                # Reset the environment
                state, reset_key, obs, info = env.reset(reset_key)
                # For imitation learning, set the humans' safety space to 0.1
                info["humans_parameters"] = info["humans_parameters"].at[:,18].set(jnp.ones((env.n_humans,)) * 0.1) 
                # Episode loop
                vnet_inputs = jnp.empty((int(env.time_limit/env.robot_dt), policy.vnet_input_size))
                rewards = jnp.empty((int(env.time_limit/env.robot_dt),))
                val_init = (state, obs, info, False, policy_key, vnet_inputs, rewards, 0)
                _, _, _, _, policy_key, vnet_inputs, rewards, episode_steps = lax.while_loop(lambda x: x[3] == False, _while_body, val_init)           
                # Update buffer state - During IL we feed the entire state value in the reward key
                @jit
                def _compute_state_value_for_body(j:int, t:int, value:float):
                        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
                        return value
                ############
                # Standard way to update the buffer state 
                buffer_state, current_buffer_size = lax.fori_loop(
                        0,
                        episode_steps,
                        lambda k, buff: (replay_buffer.add(buff[0], (vnet_inputs[k], lax.fori_loop(k,episode_steps,lambda j, val: _compute_state_value_for_body(j, k, val),0.)), buff[1]), buff[1]+1),
                        (buffer_state, current_buffer_size))
                ## Alternative way to update the buffer state 
                # next_vnet_inputs = jnp.empty((int(env.time_limit/env.robot_dt), policy.vnet_input_size))
                # next_vnet_inputs = lax.fori_loop(0, episode_steps, lambda k, val: val.at[k].set(vnet_inputs[k+1]), next_vnet_inputs)
                # buffer_state, current_buffer_size = lax.fori_loop(
                #         0,
                #         episode_steps,
                #         lambda k, buff: (replay_buffer.add(buff[0], (vnet_inputs[k], lax.cond(k!=episode_steps-1, lambda _: jnp.squeeze(rewards[k] + pow(policy.gamma, policy.dt * policy.v_max) * model.apply(model_params, None, next_vnet_inputs[k])), lambda _: rewards[k], None)), buff[1]), buff[1]+1),
                #         (buffer_state, current_buffer_size))
                ############
                # Compute episode return
                cumulative_episode_reward = lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.)
                returns = returns.at[i].set(cumulative_episode_reward)
                # Return the updated values
                val = (model_params, buffer_state, current_buffer_size, policy_key, reset_key, returns)
                return val

        # Initialize the random keys
        policy_key, buffer_key, reset_key = vmap(random.PRNGKey)(jnp.arange(3) + random_seed)
        # Initialize the array where to save data
        returns = jnp.empty([train_episodes])
        # Initialize the optimizer
        optimizer_state = optimizer.init(initial_vnet_params)
        # Create initial values for training loop
        val_init = (initial_vnet_params, buffer_state, 0, policy_key, reset_key, returns)
        # Execute the training loop
        model_params, buffer_state, current_buffer_size, policy_key, reset_key, returns = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        # Update model parameters
        @jit
        def _model_update_fori_body(j:int, val:tuple):
                # Retrieve data from the tuple
                buffer_key, buffer_state, size, model_params, optimizer_state, losses = val
                # Sample a batch of experiences from the replay buffer
                experiences_batch, buffer_key = replay_buffer.sample(
                buffer_key,
                buffer_state,
                size)
                # Update the model parameters
                model_params, optimizer_state, loss = policy.update(
                model_params,
                optimizer,
                optimizer_state,
                experiences_batch)
                # Save the losses
                losses = losses.at[j].set(loss)
                return (buffer_key, buffer_state, size, model_params, optimizer_state, losses)
        losses = jnp.empty([num_epochs])
        val_init = (buffer_key, buffer_state, jnp.min(jnp.array([current_buffer_size, buffer_size])), model_params, optimizer_state, losses)
        buffer_key, buffer_state, _, model_params, optimizer_state, all_losses = lax.fori_loop(0, num_epochs,_model_update_fori_body, val_init)

        output_dict = {
                "model_params": model_params,
                "optimizer_state": optimizer_state,
                "buffer_state": buffer_state,
                "current_buffer_size": current_buffer_size,
                "policy_key": policy_key,
                "buffer_key": buffer_key,
                "reset_key": reset_key,
                "loss": jnp.mean(all_losses),
                "returns": returns}

        return output_dict