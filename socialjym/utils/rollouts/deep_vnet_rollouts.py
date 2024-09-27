import haiku as hk
import optax
from typing import Callable
from jax import jit, lax, random, vmap, debug
import jax.numpy as jnp
from jax_tqdm import loop_tqdm

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
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
        current_buffer_size: int,
        num_batches: int,
        epsilon_decay_fn: Callable,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
        time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
        target_update_interval: int = 50,
        debugging: bool = False
) -> dict:
        
        # If policy is CADRL, the number of humans in the training environment must be 1
        if policy.name == "CADRL": assert env.n_humans == 1, "CADRL policy only supports training with one human."
        # For a correct shuffling of the buffer, the latter size must be a multiple of the number of batches times the batch size
        assert buffer_size % (num_batches * replay_buffer.batch_size) == 0, "The buffer size must be a multiple of the number of batches times the batch size."

        # Define the main rollout episode loop
        @loop_tqdm(train_episodes)
        @jit
        def _fori_body(i: int, val:tuple):
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, outcome, policy_key, buffer_state, current_buffer_size, vnet_inputs, rewards, dones, steps = val
                        # Step
                        epsilon = epsilon_decay_fn(epsilon_start, epsilon_end, i, decay_rate)
                        action, policy_key, vnet_input = policy.act(policy_key, obs, info, model_params, epsilon)
                        state, obs, info, reward, outcome = env.step(state, info, action)
                        # Save data
                        vnet_inputs = vnet_inputs.at[steps].set(vnet_input)
                        rewards = rewards.at[steps].set(reward)
                        dones = dones.at[steps].set(jnp.logical_not(outcome["nothing"]))
                        steps += 1
                        return (state, obs, info, outcome, policy_key, buffer_state, current_buffer_size, vnet_inputs, rewards, dones, steps)
                # Retrieve data from the tuple
                model_params, target_model_params, optimizer_state, buffer_state, current_buffer_size, losses, returns = val
                # Initialize the random keys
                policy_key, buffer_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i)
                # Reset the environment
                state, reset_key, obs, info = env.reset(reset_key)
                # Episode loop
                vnet_inputs = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans, policy.vnet_input_size))
                rewards = jnp.empty((int(time_limit/env.robot_dt)+1,))
                init_outcome = {"nothing": True, "success": False, "failure": False, "timeout": False}
                dones = jnp.empty((int(time_limit/env.robot_dt)+1,))
                val_init = (state, obs, info, init_outcome, policy_key, buffer_state, current_buffer_size, vnet_inputs, rewards, dones, 0)
                _, _, _, outcome, policy_key, buffer_state, current_buffer_size, vnet_inputs, rewards, dones, episode_steps = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, val_init)           
                # Compute episode return
                @jit
                def _compute_state_value_for_body(j:int, t:int, value:float):
                        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
                        return value 
                cumulative_episode_reward = lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.)
                # Update buffer state
                buffer_state, current_buffer_size = lax.cond(
                        jnp.any(jnp.array([outcome["success"], outcome["failure"]])), # Add only experiences of successful or failed episodes
                        lambda x: lax.fori_loop(
                                0,
                                episode_steps,
                                lambda k, buff: (replay_buffer.add(buff[0], (vnet_inputs[k], rewards[k] + (1 - dones[k]) * pow(policy.gamma, policy.dt * policy.v_max) * jnp.squeeze(model.apply(target_model_params, None, vnet_inputs[k+1]))), buff[1]), buff[1]+1),
                                (x[0], x[1])),
                        lambda x: x,
                        (buffer_state, current_buffer_size))
                # Shuffle the buffer if this is full (otherwise it is not possible without making a mess) and the number of experiences used is equal to buffer size
                actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_size]))
                buffer_state, buffer_key = lax.cond(
                        jnp.all(jnp.array([actual_buffer_size == buffer_size, ((i * num_batches * replay_buffer.batch_size) % buffer_size) == 0]),), 
                        lambda _: replay_buffer.shuffle(buffer_state, buffer_key), 
                        lambda x: x, 
                        (buffer_state,buffer_key))
                ## Update model parameters - Iterate over the buffer in num_batches
                @jit
                def _model_update_fori_body(j:int, val:tuple):
                        # Retrieve data from the tuple
                        buffer_state, size, model_params, optimizer_state, losses = val
                        # Sample a batch of experiences from the replay buffer
                        experiences_batch = replay_buffer.iterate(
                        buffer_state,
                        size,
                        ((((i * num_batches * replay_buffer.batch_size) % buffer_size) / replay_buffer.batch_size) + j)) 
                        # Update the model parameters
                        model_params, optimizer_state, loss = policy.update(
                        model_params,
                        optimizer,
                        optimizer_state,
                        experiences_batch)
                        # Save the losses
                        losses = losses.at[j].set(loss)
                        return (buffer_state, size, model_params, optimizer_state, losses)
                all_losses = jnp.empty([num_batches])
                val_init = (buffer_state, actual_buffer_size, model_params, optimizer_state, all_losses)
                buffer_state, _, model_params, optimizer_state, all_losses = lax.fori_loop(0, num_batches,_model_update_fori_body, val_init)
                ## Update model parameters - Randomly sample experiences from the buffer
                # @jit
                # def _model_update_fori_body(j:int, val:tuple):
                #         # Retrieve data from the tuple
                #         buffer_key, buffer_state, size, model_params, optimizer_state, losses = val
                #         # Sample a batch of experiences from the replay buffer
                #         experiences_batch, buffer_key = replay_buffer.sample(
                #         buffer_key,
                #         buffer_state,
                #         size)
                #         # Update the model parameters
                #         model_params, optimizer_state, loss = policy.update(
                #         model_params,
                #         optimizer,
                #         optimizer_state,
                #         experiences_batch)
                #         # Save the losses
                #         losses = losses.at[j].set(loss)
                #         return (buffer_key, buffer_state, size, model_params, optimizer_state, losses)
                # all_losses = jnp.empty([num_batches])
                # val_init = (buffer_key, buffer_state, actual_buffer_size, model_params, optimizer_state, all_losses)
                # buffer_key, buffer_state, _, model_params, optimizer_state, all_losses = lax.fori_loop(0, num_batches,_model_update_fori_body, val_init)
                # Update target network
                target_model_params = lax.cond(i % target_update_interval == 0, lambda x: model_params, lambda x: x, target_model_params)
                # Save data
                losses = losses.at[i].set(jnp.mean(all_losses))
                returns = returns.at[i].set(cumulative_episode_reward)
                # DEBUG: print the mean loss for this episode and the return
                lax.cond(
                        jnp.all(jnp.array([debugging, i % 100 == 0])),
                        lambda _: debug.print("Episode {x} - Mean loss over {y} batches: {z} - Return: {w}", x=i, y=num_batches, z=losses[i], w=returns[i]),
                        lambda x: x, 
                        None)
                # Return the updated values
                val = (model_params, target_model_params, optimizer_state, buffer_state, current_buffer_size, losses, returns)
                return val

        # Initialize the array where to save data
        losses = jnp.empty([train_episodes], dtype=jnp.float32)
        returns = jnp.empty([train_episodes], dtype=jnp.float32)
        # Initialize the model parameters
        optimizer_state = optimizer.init(initial_vnet_params)
        # Create initial values for training loop
        val_init = (initial_vnet_params, initial_vnet_params, optimizer_state, buffer_state, current_buffer_size, losses, returns)
        # Execute the training loop
        vals = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        output_dict = {}
        keys = ["model_params", "target_model_params", "optimizer_state", "buffer_state", "current_buffer_size", "losses", "returns"]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value

        return output_dict

def deep_vnet_il_rollout(
        initial_vnet_params: dict,
        train_episodes: int,
        random_seed: int,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        current_buffer_size: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        buffer_size: int,
        num_epochs: int,
        batch_size: int,
        time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
) -> dict:
        
        # If policy is CADRL, the number of humans in the training environment must be 1
        if policy.name == "CADRL": assert env.n_humans == 1, "CADRL policy only supports training with one human."

        # Define the main rollout episode loop
        @loop_tqdm(train_episodes)
        @jit
        def _fori_body(i: int, val:tuple):
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, outcome, policy_key, vnet_inputs, rewards, steps = val
                        # Step
                        vnet_input = policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
                        state, obs, info, reward, outcome = env.imitation_learning_step(state, info)
                        # Save data
                        vnet_inputs = vnet_inputs.at[steps].set(vnet_input)
                        rewards = rewards.at[steps].set(reward)
                        steps += 1
                        return (state, obs, info, outcome, policy_key, vnet_inputs, rewards, steps)
                # Retrieve data from the tuple
                model_params, buffer_state, current_buffer_size, returns = val
                # Initialize the random keys
                policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
                # Reset the environment
                state, reset_key, obs, info = env.reset(reset_key)
                # For imitation learning, set the humans' safety space to 0.1
                info["humans_parameters"] = info["humans_parameters"].at[:,-1].set(jnp.ones((env.n_humans,)) * 0.1) 
                # Episode loop
                vnet_inputs = jnp.empty((int(time_limit/env.robot_dt), env.n_humans, policy.vnet_input_size))
                rewards = jnp.empty((int(time_limit/env.robot_dt),))
                init_outcome = {"nothing": True, "success": False, "failure": False, "timeout": False}
                val_init = (state, obs, info, init_outcome, policy_key, vnet_inputs, rewards, 0)
                _, _, _, outcome, policy_key, vnet_inputs, rewards, episode_steps = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, val_init)           
                # Update buffer state
                @jit
                def _compute_state_value_for_body(j:int, t:int, value:float):
                        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
                        return value 
                buffer_state, current_buffer_size = lax.cond(
                        jnp.any(jnp.array([outcome["success"], outcome["failure"]])), # Add only experiences of successful or failed episodes
                        lambda x: lax.fori_loop(
                                0,
                                episode_steps,
                                lambda k, buff: (replay_buffer.add(buff[0], (vnet_inputs[k], lax.fori_loop(k,episode_steps,lambda j, val: _compute_state_value_for_body(j, k, val),0.)), buff[1]), buff[1]+1),
                                (x[0], x[1])),
                        lambda x: x,
                        (buffer_state, current_buffer_size))
                # Compute episode return
                cumulative_episode_reward = lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.)
                returns = returns.at[i].set(cumulative_episode_reward)
                # Return the updated values
                val = (model_params, buffer_state, current_buffer_size, returns)
                return val

        # Initialize the array where to save data
        returns = jnp.empty([train_episodes])
        # Initialize the optimizer
        optimizer_state = optimizer.init(initial_vnet_params)
        # Create initial values for training loop
        val_init = (initial_vnet_params, buffer_state, current_buffer_size, returns)
        # Execute the training loop
        debug.print("Simulating IL episodes...")
        model_params, buffer_state, current_buffer_size, returns = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        # Initialize the buffer key
        buffer_key = random.PRNGKey(random_seed)

        # Update model parameters
        actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_size]))
        debug.print("Buffer size after IL: {x}", x=actual_buffer_size)
        optimization_steps = (actual_buffer_size / batch_size).astype(int)
        @loop_tqdm(num_epochs)
        @jit
        def _epoch_fori_body(j:int, val:tuple):
                @jit
                def _model_update_fori_body(k:int, val:tuple):
                        # Retrieve data from the tuple
                        buffer_state, size, model_params, optimizer_state, losses = val
                        # Sample a batch of experiences from the replay buffer
                        experiences_batch = replay_buffer.iterate(
                        buffer_state,
                        size,
                        k)
                        # Update the model parameters
                        new_model_params, new_optimizer_state, loss = policy.update(
                        model_params,
                        optimizer,
                        optimizer_state,
                        experiences_batch)
                        new_losses = losses.at[j,k].set(loss)
                        # debug.print("Epoch {x} - Iteration {y} - Loss {z}", x=j, y=k, z=loss)
                        return (buffer_state, size, new_model_params, new_optimizer_state, new_losses)
                # Shuffle the buffer if this is full (otherwise it is not possible without making a mess)
                shuffled_buffer_state, buffer_key = lax.cond(val[2] == buffer_size, lambda _: replay_buffer.shuffle(val[1], val[0]), lambda x: x, (val[1],val[0]))
                val_init = (shuffled_buffer_state, *val[2:])
                val_end = lax.fori_loop(0, optimization_steps, _model_update_fori_body, val_init)
                val = (buffer_key, *val_end)
                return val
        losses = jnp.empty([num_epochs,optimization_steps])
        val_init = (buffer_key, buffer_state, actual_buffer_size, model_params, optimizer_state, losses)
        debug.print("Optimizing model on generated experiences for {x} epochs...", x=num_epochs)
        buffer_key, buffer_state, _, updated_model_params, optimizer_state, all_losses = lax.fori_loop(0, num_epochs,_epoch_fori_body, val_init)

        output_dict = {
                "model_params": updated_model_params,
                "optimizer_state": optimizer_state,
                "buffer_state": buffer_state,
                "current_buffer_size": current_buffer_size,
                "buffer_key": buffer_key,
                "losses": jnp.mean(all_losses, axis=1),
                "returns": returns}

        return output_dict