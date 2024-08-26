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
}
# Hyperparameters
training_episodes = 10000
checkpoint_interval = 1000
epsilon_start = 0.5
epsilon_end = 0.1
epsilon_decay = 4000
learning_rate = 0.001
buffer_size = 10000
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
def epsilon_scaling_decay(epsilon_start, epsilon_end, current_step, decay_rate):
    return epsilon_start + (epsilon_end - epsilon_start) / decay_rate * current_step

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

