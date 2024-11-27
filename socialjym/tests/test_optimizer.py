from jax import random
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rollouts import deep_vnet_rl_rollout, deep_vnet_il_rollout
from socialjym.utils.aux_functions import epsilon_scaling_decay, plot_state, plot_trajectory, test_k_trials, save_policy_params
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1

n_seeds = 10
n_il_epochs = 50
n_trials = 1000
rl_training_episodes = 10_000
espilon_start = 0.5
epsilon_end = 0.1
kinematics = 'holonomic'

loss_during_il = np.empty((2,n_seeds,n_il_epochs))
returns_after_il = np.empty((2,n_seeds,n_trials))
returns_during_rl = np.empty((2,n_seeds,rl_training_episodes))
returns_after_rl = np.empty((2,n_seeds,n_trials))

for opt in range(2):
    for seed in range(n_seeds):
        print(f"\n\nSEED {seed}")
        training_hyperparams = {
            'random_seed': seed,
            'kinematics': kinematics,
            'policy_name': 'cadrl', # 'cadrl' or 'sarl'
            'n_humans': 1,  # CADRL uses 1, SARL uses 5
            'il_training_episodes': 2_000,
            'il_learning_rate': 0.01,
            'il_num_epochs': n_il_epochs, # Number of epochs to train the model after ending IL
            'rl_training_episodes': rl_training_episodes,
            'rl_learning_rate': 0.001,
            'rl_num_batches': 100, # Number of batches to train the model after each RL episode
            'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
            'epsilon_start': espilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': 4_000,
            'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
            'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
            'humans_policy': 'sfm',
            'scenario': 'hybrid_scenario',
            'hybrid_scenario_subset': jnp.array([0,1], np.int32), # Subset of the hybrid scenarios to use for training
            'reward_function': 'socialnav_reward1',
            'custom_episodes': False, # If True, the episodes are loaded from a predefined set
        }
        # Initialize reward function
        if training_hyperparams['reward_function'] == 'socialnav_reward1': 
            reward_function = Reward1(kinematics=training_hyperparams['kinematics'])
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
            'kinematics': training_hyperparams['kinematics'],
        }
        # Initialize environment
        env = SocialNav(**env_params)
        # Initialize robot policy and vnet params
        if training_hyperparams['policy_name'] == "cadrl": 
            policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
            initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((policy.vnet_input_size,)))
        elif training_hyperparams['policy_name'] == "sarl":
            policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
            initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
        else: raise ValueError(f"{training_hyperparams['policy_name']} is not a valid policy name")
        # Initialize replay buffer
        replay_buffer = UniformVNetReplayBuffer(training_hyperparams['buffer_size'], training_hyperparams['batch_size'])
        # Initialize IL optimizer
        if opt == 0:
            optimizer = optax.sgd(learning_rate=training_hyperparams['il_learning_rate'], momentum=0.9)
        elif opt == 1:
            optimizer = optax.adam(learning_rate=training_hyperparams['il_learning_rate'])
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
        loss_during_il[opt,seed] = il_out['losses']

        # Execute tests to evaluate return after IL
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': 5,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': training_hyperparams['scenario'],
            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
            'humans_policy': training_hyperparams['humans_policy'],
            'circle_radius': 7,
            'reward_function': reward_function,
            'kinematics': training_hyperparams['kinematics'],
        }
        test_env = SocialNav(**test_env_params)
        metrics_after_il = test_k_trials(
            n_trials, 
            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
            test_env, 
            policy, 
            il_model_params, 
            reward_function.time_limit)
        returns_after_il[opt,seed] = metrics_after_il['returns']

        # Initialize RL optimizer
        if opt == 0:
            optimizer = optax.sgd(learning_rate=training_hyperparams['rl_learning_rate'], momentum=0.9)
        elif opt == 1:
            optimizer = optax.adam(learning_rate=training_hyperparams['rl_learning_rate'])

        # Initialize custom episodes path
        if training_hyperparams['custom_episodes']:
            rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
        else:
            rl_custom_episodes_path = None

        # Initialize RL rollout params
        rl_rollout_params = {
            'initial_vnet_params': il_model_params,
            'train_episodes': training_hyperparams['rl_training_episodes'],
            'random_seed': training_hyperparams['random_seed'] + training_hyperparams['il_training_episodes'],
            'model': policy.model,
            'optimizer': optimizer,
            'buffer_state': buffer_state,
            'current_buffer_size': current_buffer_size,
            'policy': policy,
            'env': env,
            'replay_buffer': replay_buffer,
            'buffer_size': training_hyperparams['buffer_size'],
            'num_batches': training_hyperparams['rl_num_batches'],
            'epsilon_decay_fn': epsilon_scaling_decay,
            'epsilon_start': training_hyperparams['epsilon_start'],
            'epsilon_end': training_hyperparams['epsilon_end'],
            'decay_rate': training_hyperparams['epsilon_decay'],
            'target_update_interval': training_hyperparams['target_update_interval'],
            'custom_episodes': rl_custom_episodes_path,
        }

        # REINFORCEMENT LEARNING ROLLOUT
        rl_out = deep_vnet_rl_rollout(**rl_rollout_params)

        # Save the training returns
        rl_model_params = rl_out['model_params']
        returns_during_rl[opt,seed] = rl_out['returns']  

        # Execute tests to evaluate return after RL
        metrics_after_rl = test_k_trials(
            n_trials, 
            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
            test_env, 
            policy, 
            rl_model_params, 
            reward_function.time_limit)
        returns_after_rl[opt,seed] = metrics_after_rl['returns']  

# Plot loss curve during IL for each seed for SGD optimizer
figure0, ax0 = plt.subplots(figsize=(10,10))
ax0.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each seed')
for seed in range(n_seeds):
    ax0.plot(
        np.arange(len(loss_during_il[0,seed])), 
        loss_during_il[0,seed],
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure0.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il_SGD.eps"), format='eps')

# Plot loss curve during IL for each seed for ADAM optimizer
figure0, ax0 = plt.subplots(figsize=(10,10))
ax0.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each seed')
for seed in range(n_seeds):
    ax0.plot(
        np.arange(len(loss_during_il[1,seed])), 
        loss_during_il[1,seed],
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure0.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il_ADAM.eps"), format='eps')

# Plot return during RL curve for each seed for SGD optimizer
figure, ax = plt.subplots(figsize=(10,10))
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each seed')
for seed in range(n_seeds):
    ax.plot(
        np.arange(len(returns_during_rl[0,seed])-(window-1))+window, 
        jnp.convolve(returns_during_rl[0,seed], jnp.ones(window,), 'valid') / window,
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure.savefig(os.path.join(os.path.dirname(__file__),"return_curves_during_rl_SGD.eps"), format='eps')

# Plot return during RL curve for each seed for ADAM optimizer
figure, ax = plt.subplots(figsize=(10,10))
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each seed')
for seed in range(n_seeds):
    ax.plot(
        np.arange(len(returns_during_rl[1,seed])-(window-1))+window, 
        jnp.convolve(returns_during_rl[1,seed], jnp.ones(window,), 'valid') / window,
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure.savefig(os.path.join(os.path.dirname(__file__),"return_curves_during_rl_ADAM.eps"), format='eps')

# Plot boxplot of the returns after IL for each seed
figure2, ax2 = plt.subplots(figsize=(10,10))
ax2.set(xlabel='Seed', ylabel='Return', title='Return after IL with SGD and ADAM optimizers for each seed')
ax2.boxplot(np.transpose(returns_after_il[0]), widths=0.4, patch_artist=True, 
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True, 
            zorder=1)
ax2.boxplot(np.transpose(returns_after_il[1]), widths=0.3, patch_artist=True, 
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
            showfliers=False,
            showmeans=True,
            zorder=2)
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="After IL - SGD"),
    Line2D([0], [0], color="lightcoral", lw=4, label="After IL - ADAM")
]
ax2.legend(handles=legend_elements, loc="upper right")
figure2.savefig(os.path.join(os.path.dirname(__file__),"return_curves_after_il_SGD_ADAM.png"), format='png')

# Plot boxplot of the returns after RL for each seed
figure2, ax2 = plt.subplots(figsize=(10,10))
ax2.set(xlabel='Seed', ylabel='Return', title='Return after RL with SGD and ADAM optimizers for each seed')
ax2.boxplot(np.transpose(returns_after_rl[0]), widths=0.4, patch_artist=True, 
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True, 
            zorder=1)
ax2.boxplot(np.transpose(returns_after_rl[1]), widths=0.3, patch_artist=True, 
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
            showfliers=False,
            showmeans=True,
            zorder=2)
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="After RL - SGD"),
    Line2D([0], [0], color="lightcoral", lw=4, label="After RL - ADAM")
]
ax2.legend(handles=legend_elements, loc="upper right")
figure2.savefig(os.path.join(os.path.dirname(__file__),"return_curves_after_rl_SGD_ADAM.png"), format='png')