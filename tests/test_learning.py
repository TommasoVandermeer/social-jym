from jax import random
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.base_vnet_replay_buffer import BaseVNetReplayBuffer
from socialjym.utils.rollouts.vnet_rollouts import vnet_rl_rollout, vnet_il_rollout
from socialjym.utils.aux_functions import linear_decay, plot_state, plot_trajectory, test_k_trials, save_policy_params
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1

n_seeds = 1
n_il_epochs = 50
n_trials = 1000
rl_training_episodes = 30_000
espilon_start = 0.5
epsilon_end = 0.1
kinematics = 'unicycle'
unicycle_box_action_space = True
n_humans_for_tests = [5, 10, 15, 20, 25]

loss_during_il = np.empty((n_seeds,n_il_epochs))
returns_after_il = np.empty((len(n_humans_for_tests),n_seeds,n_trials))
success_rate_after_il = np.empty((len(n_humans_for_tests),n_seeds))
returns_during_rl = np.empty((n_seeds,rl_training_episodes))
returns_after_rl = np.empty((len(n_humans_for_tests),n_seeds,n_trials))
success_rate_after_rl = np.empty((len(n_humans_for_tests),n_seeds))

for seed in range(n_seeds):
    print(f"\n\nSEED {seed}")
    training_hyperparams = {
        'random_seed': seed,
        'kinematics': kinematics,
        'policy_name': 'sarl', # 'cadrl' or 'sarl'
        'n_humans': 5,  # CADRL uses 1, SARL uses 5
        'il_training_episodes': 2_000,
        'il_learning_rate': 0.01,
        'il_num_epochs': n_il_epochs, # Number of epochs to train the model after ending IL
        'rl_training_episodes': rl_training_episodes, # Holonomic kinematics 10.000 episodes, Unicycle kinematics 30.000 episodes
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
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4], np.int32), # Subset of the hybrid scenarios to use for training
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
        policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'], unicycle_box_action_space=unicycle_box_action_space)
        initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((policy.vnet_input_size,)))
    elif training_hyperparams['policy_name'] == "sarl":
        policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'], unicycle_box_action_space=unicycle_box_action_space)
        initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
    else: raise ValueError(f"{training_hyperparams['policy_name']} is not a valid policy name")
    # Initialize replay buffer
    replay_buffer = BaseVNetReplayBuffer(training_hyperparams['buffer_size'], training_hyperparams['batch_size'])
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
    il_out = vnet_il_rollout(**il_rollout_params)

    # Save the IL model parameters, buffer state, and keys
    il_model_params = il_out['model_params']
    buffer_state = il_out['buffer_state']
    current_buffer_size = il_out['current_buffer_size']
    loss_during_il[seed] = il_out['losses']

    # Execute tests to evaluate return after IL
    for test, n_humans in enumerate(n_humans_for_tests):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
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
        returns_after_il[test,seed] = metrics_after_il['returns']
        success_rate_after_il[test,seed] = metrics_after_il['successes'] / n_trials

    # Initialize RL optimizer
    optimizer = optax.sgd(learning_rate=training_hyperparams['rl_learning_rate'], momentum=0.9)

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
        'epsilon_decay_fn': linear_decay,
        'epsilon_start': training_hyperparams['epsilon_start'],
        'epsilon_end': training_hyperparams['epsilon_end'],
        'decay_rate': training_hyperparams['epsilon_decay'],
        'target_update_interval': training_hyperparams['target_update_interval'],
        'custom_episodes': rl_custom_episodes_path,
    }

    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = vnet_rl_rollout(**rl_rollout_params)

    # Save the training returns
    rl_model_params = rl_out['model_params']
    returns_during_rl[seed] = rl_out['returns']  
    with open(os.path.join(os.path.dirname(__file__),f"rl_model_params_{seed}.pkl"), 'wb') as f:
        pickle.dump(rl_model_params, f)

    # Execute tests to evaluate return after RL
    for test, n_humans in enumerate(n_humans_for_tests):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
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
        metrics_after_rl = test_k_trials(
            n_trials, 
            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
            test_env, 
            policy, 
            rl_model_params, 
            reward_function.time_limit)
        returns_after_rl[test,seed] = metrics_after_rl['returns']  
        success_rate_after_rl[test,seed] = metrics_after_rl['successes'] / n_trials

# Save all output data
output_data = {
    'loss_during_il': loss_during_il,
    'returns_after_il': returns_after_il,
    'success_rate_after_il': success_rate_after_il,
    'returns_during_rl': returns_during_rl,
    'returns_after_rl': returns_after_rl,
    'success_rate_after_rl': success_rate_after_rl
}
with open(os.path.join(os.path.dirname(__file__),"output_data_test_learning.pkl"), 'wb') as f:
    pickle.dump(output_data, f)

# Plot loss curve during IL for each seed
figure0, ax0 = plt.subplots(figsize=(10,10))
ax0.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each seed')
for seed in range(n_seeds):
    ax0.plot(
        np.arange(len(loss_during_il[seed])), 
        loss_during_il[seed],
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure0.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il.eps"), format='eps')

# Plot return during RL curve for each seed
figure, ax = plt.subplots(figsize=(10,10))
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each seed')
for seed in range(n_seeds):
    ax.plot(
        np.arange(len(returns_during_rl[seed])-(window-1))+window, 
        jnp.convolve(returns_during_rl[seed], jnp.ones(window,), 'valid') / window,
        color = list(mcolors.TABLEAU_COLORS.values())[seed])
figure.savefig(os.path.join(os.path.dirname(__file__),"return_curves_during_rl.eps"), format='eps')

# Plot boxplot of the returns for each seed
for test, n_humans in enumerate(n_humans_for_tests):
    figure2, ax2 = plt.subplots(figsize=(10,10))
    ax2.set(xlabel='Seed', ylabel='Return', title=f'Return after IL and RL training for each seed - {n_trials} trials - {n_humans} humans')
    ax2.boxplot(np.transpose(returns_after_il[test]), widths=0.4, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True, 
                zorder=1)
    ax2.boxplot(np.transpose(returns_after_rl[test]), widths=0.3, patch_artist=True, 
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
                showfliers=False,
                showmeans=True,
                zorder=2)
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=4, label="After IL"),
        Line2D([0], [0], color="lightcoral", lw=4, label="After RL")
    ]
    ax2.legend(handles=legend_elements, loc="upper right")
    figure2.savefig(os.path.join(os.path.dirname(__file__),f"return_curves_after_il_and_rl_{n_humans}humans.png"), format='png')

# Plot success rate after IL and RL for each seed
figure3, ax3 = plt.subplots(2,1,figsize=(10,10))
ax3[0].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after IL training for each seed - {n_trials} trials', 
    xticks=np.arange(len(n_humans_for_tests)), 
    xticklabels=n_humans_for_tests, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
ax3[1].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after RL training for each seed - {n_trials} trials', 
    xticks=np.arange(len(n_humans_for_tests)), 
    xticklabels=n_humans_for_tests, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
for seed in range(n_seeds):
    ax3[0].plot(success_rate_after_il[:,seed])
    ax3[1].plot(success_rate_after_rl[:,seed])
figure3.savefig(os.path.join(os.path.dirname(__file__),f"success_rate_curves_after_il_and_rl_{n_humans}humans.eps"), format='eps')
    