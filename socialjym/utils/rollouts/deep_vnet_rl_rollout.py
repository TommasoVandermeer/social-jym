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
        train_episodes: int,
        random_seed: int,
        model: hk.Transformed,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        vnet_input_shape: int,
        buffer_size: int,
        epsilon_decay_fn: Callable,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
        imitation_learning: bool = False
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
                        if not imitation_learning:
                                action, policy_key, vnet_input = policy.act(policy_key, obs, info, model_params, epsilon)
                                state, obs, info, reward, done = env.step(state, info, action)
                        else:
                                vnet_input = policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
                                state, obs, info, reward, done = env.imitation_learning_step(state, info)
                        cumulative_reward += policy.gamma **(info["time"] * policy.v_max) * reward
                        experience = (jnp.squeeze(vnet_input), reward, jnp.squeeze(policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info)), done)
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
                # Sample a batch of experiences from the replay buffer
                experiences_batch, buffer_key = replay_buffer.sample(
                buffer_key,
                buffer_state,
                jnp.min(jnp.array([current_buffer_size, buffer_size])))
                # Update the model parameters
                model_params, optimizer_state, loss = policy.update(
                model_params,
                optimizer,
                optimizer_state,
                experiences_batch)
                # Save the losses
                losses = losses.at[i].set(loss)
                returns = returns.at[i].set(cumulative_reward)
                # Return the updated values
                val = (model_params, optimizer_state, buffer_state, current_buffer_size, policy_key, buffer_key, reset_key, losses, returns)
                return val

        # Initialize the random keys
        params_key, policy_key, buffer_key, reset_key = vmap(random.PRNGKey)(jnp.arange(4) + random_seed)
        # Initialize the array where to save data
        losses = jnp.empty([train_episodes], dtype=jnp.float32)
        returns = jnp.empty([train_episodes], dtype=jnp.float32)
        # Initialize the model parameters
        model_params = model.init(params_key, jnp.zeros(vnet_input_shape))
        optimizer_state = optimizer.init(model_params)
        # Create initial values for training loop
        val_init = (model_params, optimizer_state, buffer_state, 0, policy_key, buffer_key, reset_key, losses, returns)
        # Execute the training loop
        vals = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        output_dict = {}
        keys = ["model_params", "optimizer_state", "buffer_state", "current_buffer_size", "policy_key", "buffer_key", "reset_key", "losses", "returns"]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value

        return output_dict