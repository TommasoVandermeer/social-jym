import haiku as hk
import optax
from typing import Callable
from jax import jit, lax, random, vmap, debug, tree_map
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
import pickle

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.base_a2c_buffer import BaseA2CBuffer


def deep_a2c_rl_rollout(
        initial_actor_params: dict,
        initial_critic_params: dict,
        train_updates: int, # Number of episodes to train on, exploration episodes excluded
        random_seed: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        buffer_state: dict,
        buffer_capacity: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseA2CBuffer,
        sigma_decay_fn: Callable,
        sigma_start: float,
        sigma_end: float,
        sigma_decay: int,
        beta_entropy: float,
        debugging: bool = False,
) -> dict:
        
        assert policy.name == "SARL-A2C", "This function is only compatible with A2C policies."

        # Define the main rollout episode loop
        @jit
        def _update_loop_body(val:tuple):
                ## Generate experience
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, outcome, policy_key, buffer_state, current_buffer_size, inputs, rewards, sampled_actions, dones, steps, sigma = val
                        # Step
                        action, policy_key, input, sampled_action, _ = policy.act(policy_key, obs, info, actor_params, sigma=sigma)
                        state, obs, info, reward, outcome, _ = env.step(state, info, action)
                        # Save data
                        inputs = inputs.at[steps].set(input)
                        rewards = rewards.at[steps].set(reward)
                        sampled_actions = sampled_actions.at[steps].set(sampled_action)
                        dones = dones.at[steps].set(jnp.logical_not(outcome["nothing"]))
                        steps += 1
                        return (state, obs, info, outcome, policy_key, buffer_state, current_buffer_size, inputs, rewards, sampled_actions, dones, steps, sigma)
                # Retrieve data from the tuple
                update_count, episode_count, actor_params, critic_params, act_opt_state, cri_opt_state, buffer_state, current_buffer_size, actor_losses, critic_losses, entropy_losses, returns, successes, failures, timeouts, = val
                # Initialize the random keys
                policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + episode_count)
                # Reset the environment
                state, reset_key, obs, info, init_outcome = env.reset(reset_key)
                # Compute sigma for the episode
                sigma = sigma_decay_fn(sigma_start, sigma_end, update_count, sigma_decay)
                # Episode loop
                inputs = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, env.n_humans, policy.vnet_input_size))
                rewards = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
                sampled_actions = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
                dones = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
                val_init = (state, obs, info, init_outcome, policy_key, buffer_state, current_buffer_size, inputs, rewards, sampled_actions, dones, 0, sigma)
                _, _, _, outcome, policy_key, buffer_state, current_buffer_size, inputs, rewards, sampled_actions, dones, episode_steps, _ = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, val_init)           
                # Compute episode return
                @jit
                def _compute_state_value_for_body(j:int, t:int, value:float):
                        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
                        return value 
                cumulative_episode_reward = lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.)
                # Save episode data
                returns = returns.at[episode_count].set(cumulative_episode_reward)
                successes = successes.at[episode_count].set(outcome["success"])
                failures = failures.at[episode_count].set(outcome["failure"])
                timeouts = timeouts.at[episode_count].set(outcome["timeout"])
                # Update buffer state
                buffer_state, current_buffer_size = lax.cond(
                        (outcome["success"]) | (outcome["failure"]) | (outcome["timeout"]), # Add all experiences
                        lambda x: lax.fori_loop(
                                0,
                                episode_steps,
                                lambda k, buff: (
                                        replay_buffer.add(
                                                buff[0], 
                                                (
                                                        inputs[k], 
                                                        rewards[k] + (1 - dones[k]) * pow(policy.gamma, policy.dt * policy.v_max) * jnp.squeeze(policy.critic.apply(critic_params, None, inputs[k+1])), 
                                                        sampled_actions[k]
                                                ), 
                                                buff[1]
                                        ), 
                                        buff[1]+1),
                                (x[0], x[1])
                        ),
                        lambda x: x,
                        (buffer_state, current_buffer_size)
                )
                actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_capacity]))
                ### Update model parameters
                @jit
                def _model_update_cond_body(cond_val:tuple):
                        update_count, episode_count, actor_params, critic_params, actor_optimizer_state, critic_optimizer_state, buffer_state, current_buffer_size, actor_losses, critic_losses, entropy_losses, returns, successes, failures, timeouts, sigma = cond_val
                        # Sample a batch of experiences from the replay buffer
                        experiences_batch = replay_buffer.iterate(
                                buffer_state,
                                actual_buffer_size,
                                0) 
                        # Update the model parameters
                        critic_params, actor_params, critic_optimizer_state, actor_optimizer_state, critic_loss, actor_loss, entropy_loss = policy.update(
                                critic_params,
                                actor_params,
                                actor_optimizer,
                                actor_optimizer_state,
                                critic_optimizer,
                                critic_optimizer_state,
                                experiences_batch,
                                sigma,
                                beta_entropy,
                                imitation_learning = False
                        )
                        # Empty the buffer
                        buffer_state = replay_buffer.empty(buffer_state)
                        current_buffer_size = 0
                        # Save data
                        actor_losses = actor_losses.at[update_count].set(actor_loss)
                        critic_losses = critic_losses.at[update_count].set(critic_loss)
                        entropy_losses = entropy_losses.at[update_count].set(entropy_loss)
                        # DEBUG: print the mean loss for this episode and the return
                        lax.cond(
                                (debugging) & (update_count % 100 == 0) & (episode_count > 500),   
                                lambda _: debug.print(
                                        "### Update {x} - Episodes {w}\nActor loss: {y} - Critic loss: {z} - Entropy: {e} - Avg. Return: {r}\nSuccess rate: {s} - Failure rate: {f} - Timeout rate: {t}", 
                                        x=(update_count), 
                                        y=actor_losses[update_count], 
                                        z=critic_losses[update_count], 
                                        w=episode_count,
                                        r=jnp.nanmean(jnp.where((jnp.arange(len(returns)) > episode_count-500) & (jnp.arange(len(returns)) < episode_count), returns, jnp.nan)),
                                        s=jnp.nanmean(jnp.where((jnp.arange(len(successes)) > episode_count-500) & (jnp.arange(len(successes)) < episode_count), successes, jnp.nan)),
                                        f=jnp.nanmean(jnp.where((jnp.arange(len(failures)) > episode_count-500) & (jnp.arange(len(failures)) < episode_count), failures, jnp.nan)),
                                        t=jnp.nanmean(jnp.where((jnp.arange(len(timeouts)) > episode_count-500) & (jnp.arange(len(timeouts)) < episode_count), timeouts, jnp.nan)),
                                        e=entropy_losses[update_count],
                                        ),
                                lambda x: x, 
                                None
                        )
                        # Increment update count
                        update_count += 1
                        return (update_count, episode_count, actor_params, critic_params, actor_optimizer_state, critic_optimizer_state, buffer_state, current_buffer_size, actor_losses, critic_losses, entropy_losses, returns, successes, failures, timeouts, sigma)
                val = lax.cond(
                        actual_buffer_size > replay_buffer.batch_size,
                        _model_update_cond_body, 
                        lambda x: x, 
                        (update_count, episode_count, actor_params, critic_params, act_opt_state, cri_opt_state, buffer_state, current_buffer_size, actor_losses, critic_losses, entropy_losses, returns, successes, failures, timeouts, sigma))
                # Increment episode count
                val = (val[0], episode_count+1, *val[2:-1])
                return val

        # Initialize the array where to save data
        actor_losses = jnp.empty([train_updates], dtype=jnp.float32)
        critic_losses = jnp.empty([train_updates], dtype=jnp.float32)
        entropy_losses = jnp.empty([train_updates], dtype=jnp.float32)
        returns = jnp.empty([int(train_updates*replay_buffer.batch_size)], dtype=jnp.float32)
        successes = jnp.empty([int(train_updates*replay_buffer.batch_size)], dtype=int)
        failures = jnp.empty([int(train_updates*replay_buffer.batch_size)], dtype=int)
        timeouts = jnp.empty([int(train_updates*replay_buffer.batch_size)], dtype=int)
        # Initialize the model parameters
        actor_optimizer_state = actor_optimizer.init(initial_actor_params)
        critic_optimizer_state = critic_optimizer.init(initial_critic_params)
        # Create initial values for training loop
        val_init = (
                0,
                0,
                initial_actor_params, 
                initial_critic_params, 
                actor_optimizer_state, 
                critic_optimizer_state, 
                buffer_state, 
                0,
                actor_losses, 
                critic_losses, 
                entropy_losses,
                returns,
                successes,
                failures,
                timeouts,
        )
        # Execute the training loop
        debug.print("Performing RL training for {x} updates...", x=train_updates)
        vals = lax.while_loop(
                lambda x: x[0] < train_updates, 
                _update_loop_body, 
                val_init
        )
        output_dict = {}
        keys = [
                "update_count",
                "episode_count",
                "actor_params", 
                "critic_params", 
                "actor_optimizer_state", 
                "critic_optimizer_state", 
                "buffer_state", 
                "current_buffer_size",
                "actor_losses", 
                "critic_losses", 
                "entropy_losses",
                "returns",
                "successes",
                "failures",
                "timeouts",
        ]
        for idx, value in enumerate(vals): output_dict[keys[idx]] = value
        return output_dict

def deep_a2c_il_rollout(
        initial_actor_params: dict,
        initial_critic_params: dict,
        train_episodes: int,
        random_seed: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        buffer_state: dict,
        current_buffer_size: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseA2CBuffer,
        buffer_capacity: int,
        num_epochs: int,
        batch_size: int,
        custom_episodes: str = None,
) -> dict:
        
        assert policy.name == "SARL-A2C", "This function is only compatible with the SARL-A2C policy."

        # Load custom episodes if provided
        if custom_episodes is not None: 
                with open(custom_episodes, 'rb') as f:
                        custom_episode_data = pickle.load(f)
                train_episodes = len(custom_episode_data)
                print(f"Custom episodes loaded: {train_episodes}")
                # Since jax does not allow to loop over a dict, we have to decompose it in singular jax numpy arrays
                custom_states = jnp.array([custom_episode_data[i]["full_state"] for i in range(train_episodes)])
                custom_robot_goals = jnp.array([custom_episode_data[i]["robot_goal"] for i in range(train_episodes)])
                custom_humans_goals = jnp.array([custom_episode_data[i]["humans_goal"] for i in range(train_episodes)])
                custom_static_obstacles = jnp.array([custom_episode_data[i]["static_obstacles"] for i in range(train_episodes)])
                custom_scenario = jnp.array([custom_episode_data[i]["scenario"] for i in range(train_episodes)])
                custom_humans_radius = jnp.array([custom_episode_data[i]["humans_radius"] for i in range(train_episodes)])
                custom_humans_speed = jnp.array([custom_episode_data[i]["humans_speed"] for i in range(train_episodes)])
        else:
                # Dummy variables
                custom_states = jnp.empty((train_episodes, env.n_humans+1, 6))
                custom_robot_goals = jnp.empty((train_episodes, 2))
                custom_humans_goals = jnp.empty((train_episodes, env.n_humans, 2))
                custom_static_obstacles = jnp.empty((train_episodes, 1, 1, 2, 2))
                custom_scenario = jnp.empty((train_episodes,), dtype=int)
                custom_humans_radius = jnp.empty((train_episodes, env.n_humans))
                custom_humans_speed = jnp.empty((train_episodes, env.n_humans))

        # Define the main rollout episode loop
        @loop_tqdm(train_episodes)
        @jit
        def _fori_body(i: int, val:tuple):
                @jit
                def _while_body(val:tuple):
                        # Retrieve data from the tuple
                        state, obs, info, outcome, inputs, rewards, actions, steps = val
                        # Step
                        input = policy.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
                        state, obs, info, reward, outcome = env.imitation_learning_step(state, info)
                        # Save data
                        inputs = inputs.at[steps].set(input)
                        rewards = rewards.at[steps].set(reward)
                        actions = actions.at[steps].set(obs[-1,2:4])
                        steps += 1
                        return (state, obs, info, outcome, inputs, rewards, actions, steps)
                # Retrieve data from the tuple
                buffer_state, current_buffer_size, returns = val
                # Initialize the random keys
                reset_key = random.PRNGKey(random_seed + i)
                # Reset the environment
                state, reset_key, obs, info, init_outcome = lax.cond(
                        custom_episodes is not None, 
                        lambda x: env.reset_custom_episode(
                                x,
                                {"full_state": custom_states[i], 
                                "robot_goal": custom_robot_goals[i], 
                                "humans_goal": custom_humans_goals[i], 
                                "static_obstacles": custom_static_obstacles[i], 
                                "scenario": custom_scenario[i], 
                                "humans_radius": custom_humans_radius[i], 
                                "humans_speed": custom_humans_speed[i],
                                "humans_delay": jnp.zeros(env.n_humans)}),
                        lambda x: env.reset(x), 
                        reset_key)
                # For imitation learning, set the humans' safety space to 0.1
                info["humans_parameters"] = info["humans_parameters"].at[:,-1].set(jnp.ones((env.n_humans,)) * 0.1) 
                # Episode loop
                inputs = jnp.empty((int(env.reward_function.time_limit/env.robot_dt), env.n_humans, policy.vnet_input_size))
                rewards = jnp.empty((int(env.reward_function.time_limit/env.robot_dt),))
                actions = jnp.empty((int(env.reward_function.time_limit/env.robot_dt), 2))
                val_init = (state, obs, info, init_outcome, inputs, rewards, actions, 0)
                _, _, _, outcome, inputs, rewards, actions, episode_steps = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, val_init)           
                # Update buffer state
                @jit
                def _compute_state_value_for_body(j:int, t:int, value:float):
                        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
                        return value 
                buffer_state, current_buffer_size = lax.cond(
                        outcome["success"], # Add only experiences of successful episodes
                        lambda x: lax.fori_loop(
                                0,
                                episode_steps,
                                lambda k, buff: (replay_buffer.add(
                                        buff[0], (
                                                inputs[k], 
                                                lax.fori_loop(
                                                        k,
                                                        episode_steps,
                                                        lambda j, val: _compute_state_value_for_body(j, k, val),
                                                        0.
                                                ), 
                                                actions[k],
                                                ), 
                                        buff[1]), buff[1]+1),
                                (x[0], x[1])),
                        lambda x: x,
                        (buffer_state, current_buffer_size))
                # Compute episode return
                cumulative_episode_reward = lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.)
                returns = returns.at[i].set(cumulative_episode_reward)
                # Return the updated values
                val = (buffer_state, current_buffer_size, returns)
                return val

        # Initialize the array where to save data
        returns = jnp.empty([train_episodes])
        # Initialize the optimizer
        actor_optimizer_state = actor_optimizer.init(initial_actor_params)
        critic_optimizer_state = critic_optimizer.init(initial_critic_params)
        # Create initial values for training loop
        val_init = (buffer_state, current_buffer_size, returns)
        # Execute the training loop
        debug.print("Simulating IL episodes...")
        buffer_state, current_buffer_size, returns = lax.fori_loop(0, train_episodes, _fori_body, val_init)

        # Initialize the buffer key
        buffer_key = random.PRNGKey(0) # We do not care to control the buffer shuffle

        # Update model parameters
        actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_capacity]))
        debug.print("Buffer size after IL: {x}", x=actual_buffer_size)
        optimization_steps = (actual_buffer_size / batch_size).astype(int)
        @loop_tqdm(num_epochs)
        @jit
        def _epoch_fori_body(j:int, val:tuple):
                ### Loop over optimization steps
                @jit
                def _model_update_fori_body(k:int, val:tuple):
                        # Retrieve data from the tuple
                        buffer_state, size, actor_params, critic_params, act_opt_state, cri_opt_state, act_losses, cri_losses = val
                        # Sample a batch of experiences from the replay buffer
                        experiences_batch = replay_buffer.iterate(
                                buffer_state,
                                size,
                                k
                        )
                        # Update the model parameters
                        critic_params, actor_params, cri_opt_state, act_opt_state, cri_loss, act_loss, _ = policy.update(
                                critic_params,
                                actor_params,
                                actor_optimizer,
                                act_opt_state,
                                critic_optimizer,
                                cri_opt_state,
                                experiences_batch,
                                sigma = 0.,
                                beta_entropy = 0.,
                                imitation_learning = True
                        )
                        act_losses = act_losses.at[j,k].set(act_loss)
                        cri_losses = cri_losses.at[j,k].set(cri_loss)
                        # debug.print("Epoch {x} - Iteration {y} - Loss {z}", x=j, y=k, z=loss)
                        return (buffer_state, size, actor_params, critic_params, act_opt_state, cri_opt_state, act_losses, cri_losses)
                # Shuffle the buffer if this is full (otherwise it is not possible without making a mess)
                shuffled_buffer_state, buffer_key = lax.cond(val[2] == buffer_capacity, lambda _: replay_buffer.shuffle(val[1], val[0]), lambda x: x, (val[1],val[0]))
                val_init = (shuffled_buffer_state, *val[2:])
                val_end = lax.fori_loop(0, optimization_steps, _model_update_fori_body, val_init)
                val = (buffer_key, *val_end)
                return val
        
        debug.print("Pre-shuffling the buffer...")
        buffer_state, buffer_key = lax.cond(
                actual_buffer_size == buffer_capacity, 
                lambda x: replay_buffer.shuffle(x[0], x[1], 100),
                lambda x: x,
                (buffer_state, buffer_key))
        actor_losses = jnp.empty([num_epochs,optimization_steps])
        critic_losses = jnp.empty([num_epochs,optimization_steps])
        val_init = (
                buffer_key, 
                buffer_state, 
                actual_buffer_size, 
                initial_actor_params, 
                initial_critic_params, 
                actor_optimizer_state,
                critic_optimizer_state,
                actor_losses,
                critic_losses,
        )
        debug.print("Optimizing model on generated experiences for {x} epochs...", x=num_epochs)
        buffer_key, buffer_state, _, actor_params, critic_params, actor_opt_state, critic_opt_state, act_losses, cri_losses = lax.fori_loop(
                0, 
                num_epochs,
                _epoch_fori_body, 
                val_init
        )
        output_dict = {
                "actor_params": actor_params,
                "critic_params": critic_params,
                "actor_optimizer_state": actor_opt_state,
                "critic_optimizer_state": critic_opt_state,
                "buffer_state": buffer_state,
                "current_buffer_size": current_buffer_size,
                "buffer_key": buffer_key,
                "actor_losses": jnp.mean(act_losses, axis=1),
                "critic_losses": jnp.mean(cri_losses, axis=1),
                "returns": returns}
        return output_dict