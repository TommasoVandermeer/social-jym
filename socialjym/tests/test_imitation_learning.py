from jax import random
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rollouts import deep_vnet_rl_rollout, deep_vnet_il_rollout
from socialjym.utils.aux_functions import epsilon_scaling_decay, plot_state, plot_trajectory, test_k_trials, save_policy_params
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1

n_seeds = 10
n_trials = 1000
losses_at_last_epoch  = np.empty((n_seeds,))
returns = np.empty((n_seeds,n_trials))

for seed in range(n_seeds):
    print(f"\n\nSEED {seed}")
    training_hyperparams = {
        'random_seed': seed,
        'policy_name': 'cadrl', # 'cadrl' or 'sarl'
        'n_humans': 1,  # CADRL uses 1, SARL uses 5
        'il_training_episodes': 3_000,
        'il_learning_rate': 0.01,
        'il_num_epochs': 50, # Number of epochs to train the model after ending IL
        'rl_training_episodes': 10_000,
        'rl_learning_rate': 0.001,
        'rl_num_batches': 100, # Number of batches to train the model after each RL episode
        'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
        'epsilon_start': 0.5,
        'epsilon_end': 0.1,
        'epsilon_decay': 4_000,
        'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
        'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
        'humans_policy': 'sfm',
        'scenario': 'hybrid_scenario',
        'hybrid_scenario_subset': jnp.array([0,1], np.int32), # Subset of the hybrid scenarios to use for training
        'reward_function': 'socialnav_reward1',
        'custom_episodes': True, # If True, the episodes are loaded from a predefined set
    }
    # Initialize reward function
    if training_hyperparams['reward_function'] == 'socialnav_reward1': 
        reward_function = Reward1()
    else:
        raise ValueError(f"{training_hyperparams['reward_function']} is not a valid reward function")
    # Environment parameters
    env_params = {
        'robot_radius': 0.3,
        'n_humans': training_hyperparams['n_humans'],
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': False,
        'scenario': training_hyperparams['scenario'],
        'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
        'humans_policy': training_hyperparams['humans_policy'],
        'circle_radius': 7,
        'reward_function': reward_function,
    }
    # Initialize environment
    env = SocialNav(**env_params)
    # Initialize robot policy and vnet params
    if training_hyperparams['policy_name'] == "cadrl": 
        policy = CADRL(env.reward_function, dt=env_params['robot_dt'])
        initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((policy.vnet_input_size,)))
    elif training_hyperparams['policy_name'] == "sarl":
        policy = SARL(env.reward_function, dt=env_params['robot_dt'])
        initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
    else: raise ValueError(f"{training_hyperparams['policy_name']} is not a valid policy name")
    # Initialize replay buffer
    replay_buffer = UniformVNetReplayBuffer(training_hyperparams['buffer_size'], training_hyperparams['batch_size'])
    # Initialize IL optimizer
    optimizer = optax.sgd(learning_rate=training_hyperparams['il_learning_rate'], momentum=0.9)
    # Initialize buffer state
    buffer_state = {
        'vnet_inputs': jnp.empty((training_hyperparams['buffer_size'], env.n_humans, policy.vnet_input_size)),
        'targets': jnp.empty((training_hyperparams['buffer_size'],1)),
    }
    # Initialize custom episodes path
    if training_hyperparams['custom_episodes']:
        il_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/il_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
    else:
        il_custom_episodes_path = None
    # Initialize IL rollout params
    il_rollout_params = {
        'initial_vnet_params': initial_vnet_params,
        'train_episodes': training_hyperparams['il_training_episodes'],
        'random_seed': training_hyperparams['random_seed'],
        'optimizer': optimizer,
        'buffer_state': buffer_state,
        'current_buffer_size': 0,
        'policy': policy,
        'env': env,
        'replay_buffer': replay_buffer,
        'buffer_size': training_hyperparams['buffer_size'],
        'num_epochs': training_hyperparams['il_num_epochs'],
        'batch_size': training_hyperparams['batch_size'],
        'custom_episodes': il_custom_episodes_path
    }

    # IMITATION LEARNING ROLLOUT
    il_out = deep_vnet_il_rollout(**il_rollout_params)

    # Save the IL model parameters, buffer state, and keys
    il_model_params = il_out['model_params']
    buffer_state = il_out['buffer_state']
    current_buffer_size = il_out['current_buffer_size']

    losses_at_last_epoch[seed] = il_out['losses'][-1]

    metrics = test_k_trials(
        n_trials, 
        training_hyperparams['il_training_episodes'], 
        env, 
        policy, 
        il_model_params, 
        reward_function.time_limit)
    
    returns[seed] = metrics['returns']

# Print the losses at the last epoch for each seed
print("Loss at last epoch for each seed: \n", losses_at_last_epoch)
# Plot boxplot of the returns for each seed
figure, ax = plt.subplots(figsize=(10,10))
ax.set(xlabel='Seed', ylabel='Return', xticks=np.arange(n_seeds+1), xticklabels=[""] + [str(i) for i in np.arange(n_seeds)], title='Seeds only change the weights and biases initialization')
bplot = ax.boxplot(np.transpose(returns), showmeans=True, tick_labels=[str(i) for i in np.arange(n_seeds)], patch_artist=True, showfliers=False)
for patch, color in zip(bplot['boxes'], list(mcolors.TABLEAU_COLORS.values())):
    patch.set_facecolor(color)
plt.show()