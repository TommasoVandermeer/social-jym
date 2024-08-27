from jax import random
import jax.numpy as jnp
import optax

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rl_rollout import deep_vnet_rl_rollout

# Stochasticity seed
random_seed = 1
# Environment parameters
env_params = {
    'robot_radius': 0.3,
    'n_humans': 1,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'circular_crossing',
    'humans_policy': 'hsfm',
    'circle_radius': 3,
}
# Hyperparameters
training_episodes = 4000
checkpoint_interval = 1000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 4000
learning_rate = 0.001
buffer_size = 100000
batch_size = 100

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = CADRL(env.reward_function, dt=env_params['robot_dt'])
initial_vnet_params = policy.model.init(random.key(random_seed), jnp.zeros((policy.vnet_input_size,)))

# Initialize optimizer
optimizer = optax.adam(learning_rate=learning_rate)
optimizer_state = optimizer.init(initial_vnet_params)

# Define epsilon decay function
def epsilon_scaling_decay(epsilon_start, epsilon_end, current_episode, decay_rate):
    return jnp.max(jnp.array([epsilon_start + (epsilon_end - epsilon_start) / decay_rate * current_episode,0]))

# Initialize replay buffer
replay_buffer = UniformVNetReplayBuffer(buffer_size, batch_size)
buffer_state = {
    'vnet_inputs': jnp.empty((buffer_size, policy.vnet_input_size)),
    'rewards': jnp.empty((buffer_size, 1)),
    'next_vnet_inputs': jnp.empty((buffer_size, policy.vnet_input_size)),
    'dones': jnp.empty((buffer_size, 1)),
}

# Initialize rollout params
rollout_params = {
    'train_episodes': training_episodes,
    'random_seed': random_seed,
    'model': policy.model,
    'optimizer': optimizer,
    'buffer_state': buffer_state,
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'vnet_input_shape': policy.vnet_input_size,
    'buffer_size': buffer_size,
    'epsilon_decay_fn': epsilon_scaling_decay,
    'epsilon_start': epsilon_start,
    'epsilon_end': epsilon_end,
    'decay_rate': epsilon_decay
}

# Perform the training
out = deep_vnet_rl_rollout(**rollout_params)

# Save the final model
final_model_params = out['model_params']
reset_key = out['reset_key']
policy_key = out['policy_key']

# Plot the losses and returns
import numpy as np
import time
from jax import device_get
from socialjym.utils.aux_functions import plot_state, plot_trajectory
import matplotlib.pyplot as plt
figure, ax = plt.subplots(figsize=(10,10))
window = 500
ax.plot(np.arange(len(out['losses'])-(window-1))+window, jnp.convolve(out['losses'], jnp.ones(window,), 'valid') / window)
ax.set(xlabel='Episodes', ylabel='Loss', title='Loss moving average over {} episodes'.format(window))
plt.show()
figure, ax = plt.subplots(figsize=(10,10))
ax.set(xlabel='Episodes', ylabel='Return', title='Return moving average over {} episodes'.format(window))
ax.plot(np.arange(len(out['returns'])-(window-1))+window, jnp.convolve(out['returns'], jnp.ones(window,), 'valid') / window)
plt.show()

# Simulate the policy with final model parameters in new episodes
n_episodes = 5
env_params["n_humans"] = 1
env = SocialNav(**env_params)
# Simulate some episodes
episode_simulation_times = np.empty((n_episodes,))
for i in range(n_episodes):
    done = False
    episode_start_time = time.time()
    state, reset_key, obs, info = env.reset(reset_key)
    all_states = np.array([state])
    while not done:
        # action = jnp.array([0.,1.]) # Move north
        action, policy_key, _ = policy.act(policy_key, obs, info, final_model_params, 0.)
        state, obs, info, reward, done = env.step(state,info,action) 
        all_states = np.vstack((all_states, [state]))
    episode_simulation_times[i] = round(time.time() - episode_start_time,2)
    all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting
    print(f"Episode {i} ended - Execution time {episode_simulation_times[i]} seconds - Plotting trajectory...")
    ## Plot episode trajectory
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
        plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius)
    # plot last state
    plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius)
    plt.show()
# Print simulation times
print(f"Average time per episode: {round(np.mean(episode_simulation_times),2)} seconds")
print(f"Total time for {n_episodes} episodes: {round(np.sum(episode_simulation_times),2)} seconds")

