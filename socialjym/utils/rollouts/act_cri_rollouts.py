import optax
from typing import Callable
from jax import jit, lax, random, vmap, debug
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
import pickle

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer


def actor_critic_il_rollout(
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
        replay_buffer: BaseACBuffer,
        buffer_capacity: int,
        num_epochs: int,
        batch_size: int,
        custom_episodes: str = None,
) -> dict:
        policies = ["SARL-PPO", "SARL-A2C"]
        assert policy.name in policies, "This function is only compatible with the Actor-Critic policies."
        
        # Policy index
        policy_index = policies.index(policy.name)

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
                                lambda k, buff: (
                                        replay_buffer.add(
                                                buff[0],
                                                inputs[k], 
                                                lax.fori_loop(
                                                        k,
                                                        episode_steps,
                                                        lambda j, val: _compute_state_value_for_body(j, k, val),
                                                        0.,
                                                ), 
                                                actions[k], 
                                                buff[1],
                                        ), 
                                        buff[1]+1
                                ),
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
        if policy.normalize_and_clip_obs:
                actor_optimizer_state = actor_optimizer.init(initial_actor_params["actor_params"])
        else:
                actor_optimizer_state = actor_optimizer.init(initial_actor_params)
        critic_optimizer_state = critic_optimizer.init(initial_critic_params)
        # Create initial values for training loop
        val_init = (buffer_state, current_buffer_size, returns)
        # Execute the training loop
        debug.print("Simulating IL episodes...")
        buffer_state, current_buffer_size, returns = lax.fori_loop(0, train_episodes, _fori_body, val_init)
        actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_capacity]))

        # Normalizing networks inputs (if necessary)
        if policy.normalize_and_clip_obs:
                norm_state = initial_actor_params["norm_state"]
                inputs = buffer_state["inputs"][:actual_buffer_size].reshape((actual_buffer_size * env.n_humans, policy.vnet_input_size))
                updated_norm_state = policy.update_norm_state(
                        inputs,
                        norm_state,
                )
                normalized_inputs = policy.normalize_and_clip_inputs(
                        inputs,
                        updated_norm_state,
                )
                normalized_inputs = normalized_inputs.reshape((actual_buffer_size, env.n_humans, policy.vnet_input_size))
                buffer_state["inputs"] = buffer_state["inputs"].at[:actual_buffer_size].set(normalized_inputs)
                initial_actor_params["norm_state"] = updated_norm_state
                # print("Batch norm layer state: ", initial_actor_params["norm_state"])

        # Initialize the buffer key
        buffer_key = random.PRNGKey(0) # We do not care to control the buffer shuffle

        # Update model parameters
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
                        if policy_index == policies.index("SARL-PPO"):
                                critic_params, actor_params, cri_opt_state, act_opt_state, cri_loss, act_loss, _ = policy.update_il(
                                        critic_params,
                                        actor_params,
                                        actor_optimizer,
                                        act_opt_state,
                                        critic_optimizer,
                                        cri_opt_state,
                                        experiences_batch,
                                )
                        elif policy_index == policies.index("SARL-A2C"):
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
                initial_actor_params["actor_params"] if policy.normalize_and_clip_obs else initial_actor_params, 
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
                "actor_params": {"actor_params": actor_params, "norm_state": initial_actor_params["norm_state"]} if policy.normalize_and_clip_obs else actor_params,
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