from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import os
import pickle
import optax
import pandas as pd
import scipy.stats as stats

from socialjym.policies.cadrl import CADRL
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2 as Reward
from socialjym.utils.replay_buffers.base_vnet_replay_buffer import BaseVNetReplayBuffer
from socialjym.utils.aux_functions import linear_decay
from socialjym.utils.rollouts.vnet_gpu_rollouts import vnet_il_rollout, vnet_rl_rollout

out_folder = 'cadrl_seed_sensitivity_results'
base_il_network_name = 'cadrl_il_out'
base_rl_network_name = 'cadrl_rl_out'
### MASTER SEED
n_seeds = 10
### Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 1.0
kinematics = "unicycle"
scenario = "hybrid_scenario"
hybrid_scenario_subset = jnp.array([0,1,2,3,4,6])  # Exclude circular_crossing_with_static_obstacles and corner_traffic
n_humans = 5
n_obstacles = 0
humans_policy = 'hsfm'
### IL Hyperparameters
n_steps = 500_000  # Number of labeled examples to Pre-train DIR-SAFE
policy_learning_rate = 0.005
policy_batch_size = 200
policy_n_epochs = 30 # Just a few to not overfit on DIR-SAFE data (if action space becomes too deterministic there will be no exploration in RL fine-tuning)
### RL Hyperparameters
rl_n_parallel_envs = 50 
n_rl_episodes = 30_000 # Roughly the same as 500 updates * 30_000 steps (at 75 steps per episode)
training_hyperparams = {
    'random_seed': 0,
    'kinematics': 'unicycle',
    'n_humans': n_humans, 
    'n_obstacles': n_obstacles,
    'il_n_parallel_envs': 1_000,
    'il_training_episodes': 10_000,
    'il_learning_rate': 0.01,
    'il_num_epochs': policy_n_epochs, # Number of epochs to train the model after ending IL
    'il_buffer_size': 500_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'rl_training_episodes': n_rl_episodes,
    'rl_parallel_envs': rl_n_parallel_envs,
    'rl_learning_rate': 0.001,
    'rl_num_batches': 100, # Number of batches to train the model after each RL episode
    'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
    'epsilon_start': 0.5,
    'epsilon_end': 0.1,
    'epsilon_decay': int(0.4*n_rl_episodes), # Number of episodes to decay epsilon from start to end
    'rl_buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
    'humans_policy': humans_policy,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': hybrid_scenario_subset,
}

assert training_hyperparams['il_buffer_size'] > training_hyperparams['rl_buffer_size'] 

if not os.path.exists(os.path.join(os.path.dirname(__file__), out_folder)):
    os.makedirs(os.path.join(os.path.dirname(__file__), out_folder))
base_dir = os.path.join(os.path.join(os.path.dirname(__file__), out_folder))


for seed in range(n_seeds):
    random_seed = seed
    training_hyperparams['random_seed'] = random_seed
    il_network_name = base_il_network_name + '_seed' + str(random_seed) + '.pkl'
    rl_network_name = base_rl_network_name + '_seed' + str(random_seed) + '.pkl'
    ### PRE-TRAIN NETWORK (IMITATION LEARNING)
    if not os.path.exists(os.path.join(base_dir, il_network_name)):
        # Initialize reward function
        reward_function = Reward(
            v_max = robot_vmax,
            progress_to_goal_reward = True,
            high_rotation_penalty_reward = True,
            progress_to_goal_weight = 0.03,
            collision_penalty = -.5,
            angular_speed_bound= 1.,
            angular_speed_penalty_weight = 0.0075,
        )
        # Environment parameters
        env_params = {
            'robot_radius': robot_radius,
            'n_humans': training_hyperparams['n_humans'],
            'n_obstacles': training_hyperparams['n_obstacles'],
            'robot_dt': robot_dt,
            'humans_dt': 0.01,
            'robot_visible': False,
            'scenario': training_hyperparams['scenario'],
            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
            'humans_policy': training_hyperparams['humans_policy'],
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        # Initialize environment
        env = SocialNav(**env_params)
        # Initialize robot policy and vnet params
        policy = CADRL(env.reward_function, v_max=robot_vmax, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
        initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
        # Initialize replay buffer
        replay_buffer = BaseVNetReplayBuffer(training_hyperparams['il_buffer_size'], training_hyperparams['batch_size'])
        # Initialize IL optimizer
        optimizer = optax.sgd(learning_rate=training_hyperparams['il_learning_rate'], momentum=0.9)
        # Initialize buffer state
        buffer_state = {
            'vnet_inputs': jnp.empty((training_hyperparams['il_buffer_size'], env.n_humans, policy.vnet_input_size)),
            'targets': jnp.empty((training_hyperparams['il_buffer_size'],1)),
        }
        # Initialize IL rollout params
        il_rollout_params = {
            'n_parallel_envs': training_hyperparams['il_n_parallel_envs'],
            'initial_vnet_params': initial_vnet_params,
            'train_episodes': training_hyperparams['il_training_episodes'],
            'random_seed': training_hyperparams['random_seed'],
            'optimizer': optimizer,
            'buffer_state': buffer_state,
            'current_buffer_size': 0,
            'policy': policy,
            'env': env,
            'replay_buffer': replay_buffer,
            'buffer_size': training_hyperparams['il_buffer_size'],
            'num_epochs': training_hyperparams['il_num_epochs'],
            'batch_size': training_hyperparams['batch_size'],
            'custom_episodes': None
        }
        # IMITATION LEARNING ROLLOUT
        il_out = vnet_il_rollout(**il_rollout_params)
        # Save the IL model parameters, buffer state, and keys
        il_model_params = il_out['model_params']
        ### Plot losses during IL
        figure, ax = plt.subplots(1,1,figsize=(10,10))
        ax.set(
            xlabel='Epoch', 
            ylabel='Loss', 
            title='Critic Loss during IL training'
        )
        ax.plot(
            jnp.arange(len(il_out['losses'])), 
            il_out['losses'],
        )
        figure.savefig(os.path.join(base_dir,il_network_name.replace('.pkl', '.eps')), format='eps')
        plt.close(figure)
        ## Save IL rollout output
        with open(os.path.join(base_dir, il_network_name), 'wb') as f:
            pickle.dump(il_out, f)
    else:
        # Load trained parameters
        with open(os.path.join(base_dir, il_network_name), 'rb') as f:
            il_out = pickle.load(f)
    il_model_params = il_out['model_params']
    buffer_state = il_out['buffer_state']
    buffer_state = tree_map(lambda x: x[:training_hyperparams['rl_buffer_size']], buffer_state)
    current_buffer_size = training_hyperparams['rl_buffer_size']

    ### RL SARL NETWORK
    if not os.path.exists(os.path.join(base_dir, rl_network_name)):
        # Initialize reward function
        reward_function = Reward(
            v_max = robot_vmax,
            progress_to_goal_reward = True,
            high_rotation_penalty_reward = True,
            progress_to_goal_weight = 0.03,
            collision_penalty = -.5,
            angular_speed_bound= 1.,
            angular_speed_penalty_weight = 0.0075,
        )
        # Environment parameters
        env_params = {
            'robot_radius': robot_radius,
            'n_humans': training_hyperparams['n_humans'],
            'n_obstacles': training_hyperparams['n_obstacles'],
            'robot_dt': robot_dt,
            'humans_dt': 0.01,
            'robot_visible': False,
            'scenario': training_hyperparams['scenario'],
            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
            'humans_policy': training_hyperparams['humans_policy'],
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        # Initialize environment
        env = SocialNav(**env_params)
        # Initialize replay buffer
        replay_buffer = BaseVNetReplayBuffer(training_hyperparams['rl_buffer_size'], training_hyperparams['batch_size'])
        
        # Initialize policy
        policy = CADRL(env.reward_function, v_max=robot_vmax, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
        # Initialize RL optimizer
        optimizer = optax.sgd(learning_rate=training_hyperparams['rl_learning_rate'], momentum=0.9)
        # Initialize RL rollout params
        rl_rollout_params = {
            'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
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
            'buffer_size': training_hyperparams['rl_buffer_size'],
            'num_batches': training_hyperparams['rl_num_batches'],
            'epsilon_decay_fn': linear_decay,
            'epsilon_start': training_hyperparams['epsilon_start'],
            'epsilon_end': training_hyperparams['epsilon_end'],
            'decay_rate': training_hyperparams['epsilon_decay'],
            'target_update_interval': training_hyperparams['target_update_interval'],
            'custom_episodes': None,
            'debugging': True,
        }
        # REINFORCEMENT LEARNING ROLLOUT
        rl_out = vnet_rl_rollout(**rl_rollout_params)
        # Save the training returns and losses
        rl_model_params = rl_out['model_params']
        losses = rl_out['losses']
        returns = rl_out['returns'] 
        # Save RL rollout output
        with open(os.path.join(base_dir,rl_network_name), 'wb') as f:
            pickle.dump(rl_out, f)
        # Load RL rollout output
        with open(os.path.join(base_dir,rl_network_name), 'rb') as f:
            rl_out = pickle.load(f)
        # Save the training returns
        window = 20 
        ## Plot RL training stats
        from matplotlib import rc
        font = {'weight' : 'regular',
                'size'   : 18}
        rc('font', **font)
        figure, ax = plt.subplots(1,2,figsize=(15,15))
        figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95, right=0.95, left=0.1, wspace=0.35)
        # Plot returns during RL
        ax[0].grid()
        ax[0].set(
            xlabel='Training Update', 
            ylabel=f'Return ({window} upd. window)', 
            title='Return'
        )
        ax[0].plot(
            jnp.arange(len(returns)-(window-1))+window, 
            jnp.convolve(returns, jnp.ones(window,), 'valid') / window,
        )
        # Plot loss during RL
        ax[1].grid()
        ax[1].set(
            xlabel='Training Update', 
            ylabel=f'Loss ({window} upd. window)', 
            title='Loss'
        )
        ax[1].plot(
            jnp.arange(len(losses)-(window-1))+window, 
            jnp.convolve(losses, jnp.ones(window,), 'valid') / window,
        )
        figure.savefig(os.path.join(base_dir,rl_network_name.replace('.pkl', '.eps')), format='eps')
    else:
        print(f"Seed {seed} of CADRL training already done and saved... Remove '{rl_network_name}' to retrain.")

### Plots returns
from matplotlib import rc, rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "regular",
    "font.size": 23,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
if not os.path.exists(os.path.join(base_dir, "cadrl_mean_returns.pdf")):
    window_size = int(0.04 * n_rl_episodes)
    def plot():
        returns_mt = jnp.full((n_seeds, n_rl_episodes), jnp.nan)
        for seed in range(n_seeds):
            network_name = base_rl_network_name + '_seed' + str(seed) + '.pkl'
            if os.path.exists(os.path.join(base_dir, network_name)):
                print(f"Loading metrics SEED{seed}")
                with open(os.path.join(base_dir, network_name), 'rb') as f:
                    metrics = pickle.load(f)#['aux_data']
                    processed_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, list):
                            processed_metrics[key] = jnp.array(value)
                        if isinstance(value, dict):
                            processed_metrics[key] = tree_map(lambda x: jnp.array(x), value)
                        if isinstance(value, jnp.ndarray):
                            processed_metrics[key] = value
                    returns_mt = returns_mt.at[seed].set(processed_metrics['returns'])
        smoothed_returns = jnp.array([
            pd.Series(seed_data).rolling(window=window_size, min_periods=1).mean() 
            for seed_data in returns_mt
        ])
        mean_returns = jnp.nanmean(smoothed_returns, axis=0)
        std_returns = jnp.nanstd(smoothed_returns, axis=0)
        # x = jnp.arange(n_rl_episodes)
        # plt.figure(figsize=(10, 6))
        # color = 'pink'
        # plt.plot(x, mean_returns, color=color, linewidth=3.)
        # for seed in range(n_seeds): plt.plot(x, smoothed_returns[seed], color=color, linewidth=1.)
        # plt.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.2)
        # plt.xlabel('Training Updates')
        # plt.ylabel('Return')
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.show()
        return mean_returns, std_returns
    ## UNBOUNDED
    mean_dir, std_dir = plot()
    ci_dir = stats.t.ppf(0.975, 9) * std_dir / jnp.sqrt(n_seeds)
    ## Summary plot
    x = jnp.arange(n_rl_episodes)
    figure, ax = plt.subplots(1,1, figsize=(10, 6))
    figure.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.13)
    ax.plot(x, mean_dir, color='pink', linewidth=1.5, label='CADRL')
    ax.fill_between(x, mean_dir - ci_dir, mean_dir + ci_dir, color='pink', alpha=0.2)
    ax.set_xlabel('Training Updates')
    ax.set_ylabel('Return')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],loc='lower right', bbox_to_anchor=(0.999, 0.01))
    ax.grid(True, linestyle='--', alpha=0.7)
    figure.savefig(os.path.join(base_dir, "cadrl_mean_returns.pdf"), format='pdf')