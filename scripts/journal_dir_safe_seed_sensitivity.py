from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pickle
import optax
import pandas as pd
import scipy.stats as stats
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

from socialjym.policies.dir_safe import DIRSAFE
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2 as Reward
from socialjym.utils.rollouts.ppo_gpu_rollouts import ppo_rl_rollout
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer
from socialjym.utils.rollouts.act_cri_rollouts import actor_critic_il_rollout

out_folder = 'dir_safe_seed_sensitivity_results'
base_il_network_name = 'dir_safe_il_out'
base_rl_network_name = 'dir_safe_rl_out'
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
n_obstacles = 3
humans_policy = 'hsfm'
### IL Hyperparameters
n_steps = 500_000  # Number of labeled examples to Pre-train DIR-SAFE
policy_learning_rate = 0.005
policy_batch_size = 200
policy_n_epochs = 30 # Just a few to not overfit on DIR-SAFE data (if action space becomes too deterministic there will be no exploration in RL fine-tuning)
### RL Hyperparameters
rl_n_parallel_envs = 1000 
rl_training_updates = 500
training_hyperparams = {
    'random_seed': 0,
    'n_humans': 5, 
    'n_obstacles': 3,
    'il_buffer_size': n_steps, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'il_training_episodes': 2_000,
    'il_actor_learning_rate': policy_learning_rate,
    'il_critic_learning_rate': 0.01,
    'il_num_epochs': policy_n_epochs, # Number of epochs to train the model after ending IL
    'il_batch_size': policy_batch_size, # Number of experiences to sample from the replay buffer for each model update
    'rl_training_updates': rl_training_updates,
    'rl_parallel_envs': rl_n_parallel_envs,
    'rl_actor_learning_rate': 1e-4, # 1e-5
    'rl_critic_learning_rate': 1e-3, # 1e-4
    'rl_buffer_capacity': 30_000, # Number of experiences to sample from the replay buffer for each model update
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 6, # 10
    'rl_num_batches': 30, # 30
    'rl_beta_entropy': 5e-4, # 5e-4
    'lambda_gae': 0.95, # 0.95
    'humans_policy': 'hsfm',
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': hybrid_scenario_subset, # Subset of the hybrid scenarios to use for training
    'reward_function': 'socialnav_reward2',
    'gradient_norm_scale': 1., # Scale the gradient norm by this value
}

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
        _, _, obs, info, _ = env.reset(random.PRNGKey(training_hyperparams['random_seed']))
        # Initialize robot policy and vnet params
        policy = DIRSAFE(
            env.reward_function, 
            v_max=robot_vmax, 
            dt=env_params['robot_dt'], 
        )
        initial_actor_params, initial_critic_params = policy.init_nns(
            training_hyperparams['random_seed'],
            obs,
            info,
        )
        # Initialize replay buffer
        replay_buffer = BaseACBuffer(training_hyperparams['il_buffer_size'], training_hyperparams['il_batch_size'])
        # Initialize IL optimizer
        actor_optimizer = optax.sgd(learning_rate=training_hyperparams['il_actor_learning_rate'], momentum=0.9)
        critic_optimizer = optax.sgd(learning_rate=training_hyperparams['il_critic_learning_rate'], momentum=0.9)
        # Initialize buffer state
        buffer_state = {
            'inputs': jnp.empty((training_hyperparams['il_buffer_size'], env.n_humans, policy.vnet_input_size)),
            'critic_targets': jnp.empty((training_hyperparams['il_buffer_size'],)),
            'sample_actions': jnp.empty((training_hyperparams['il_buffer_size'], 2)),
        }
        # Initialize IL rollout params
        il_rollout_params = {
            'initial_actor_params': initial_actor_params,
            'initial_critic_params': initial_critic_params,
            'train_episodes': training_hyperparams['il_training_episodes'],
            'random_seed': training_hyperparams['random_seed'],
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'buffer_state': buffer_state,
            'current_buffer_size': 0,
            'policy': policy,
            'env': env,
            'replay_buffer': replay_buffer,
            'buffer_capacity': training_hyperparams['il_buffer_size'],
            'num_epochs': training_hyperparams['il_num_epochs'],
            'batch_size': training_hyperparams['il_batch_size'],
        }
        # IMITATION LEARNING ROLLOUT
        il_out = actor_critic_il_rollout(**il_rollout_params)
        ### Plot losses during IL
        figure, ax = plt.subplots(2,1,figsize=(10,10))
        ax[0].set(
            xlabel='Epoch', 
            ylabel='Loss', 
            title='Actor Loss during IL training'
        )
        ax[0].plot(
            jnp.arange(len(il_out['actor_losses'])), 
            il_out['actor_losses'],
        )
        ax[1].set(
            xlabel='Epoch', 
            ylabel='Loss', 
            title='Critic Loss during IL training'
        )
        ax[1].plot(
            jnp.arange(len(il_out['critic_losses'])), 
            il_out['critic_losses'],
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
    il_actor_params = il_out['actor_params']
    il_critic_params = il_out['critic_params'] 

    ### RL UNBOUNDED POLICY NETWORK
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
        # Initialize policy
        policy = DIRSAFE(
            env.reward_function, 
            v_max=robot_vmax, 
            dt=env_params['robot_dt'], 
        )
        # Initialize RL optimizer
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    init_value=training_hyperparams['rl_actor_learning_rate'], 
                    end_value=0., 
                    transition_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
                    transition_begin=0
                ), 
                eps=1e-7, 
                b1=0.9,
            ),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
            optax.sgd(
                learning_rate=training_hyperparams['rl_critic_learning_rate'], 
                momentum=0.9
            ),
        )
        # Initialize RL replay buffer
        replay_buffer = PPOBuffer(training_hyperparams['rl_buffer_capacity'], int(training_hyperparams['rl_buffer_capacity']/training_hyperparams['rl_num_batches']))
        # Initialize RL buffer state
        buffer_state = {
            'inputs': jnp.empty((training_hyperparams['rl_buffer_capacity'], env.n_humans, policy.vnet_input_size)),
            'critic_targets': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
            'sample_actions': jnp.empty((training_hyperparams['rl_buffer_capacity'], 2)),
            'old_values': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
            'old_neglogpdfs': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
        }
        # Initialize RL rollout params
        rl_rollout_params = {
            'initial_actor_params': il_actor_params,
            'initial_critic_params': il_critic_params,
            'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
            'train_updates': training_hyperparams['rl_training_updates'],
            'random_seed': training_hyperparams['random_seed'] + training_hyperparams['il_training_episodes'],
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'buffer_state': buffer_state,
            'buffer_capacity': training_hyperparams['rl_buffer_capacity'],
            'policy': policy,
            'env': env,
            'replay_buffer': replay_buffer,
            'clip_range': training_hyperparams['rl_clip_frac'],
            'n_epochs': training_hyperparams['rl_num_epochs'],
            'beta_entropy': training_hyperparams['rl_beta_entropy'],
            'lambda_gae': training_hyperparams['lambda_gae'],
            'debugging': True,
            'debugging_interval': 10,
        }
        # REINFORCEMENT LEARNING ROLLOUT
        rl_out = ppo_rl_rollout(**rl_rollout_params)
        # Save RL rollout output
        with open(os.path.join(base_dir,rl_network_name), 'wb') as f:
            pickle.dump(rl_out, f)
        # Load RL rollout output
        with open(os.path.join(base_dir,rl_network_name), 'rb') as f:
            rl_out = pickle.load(f)
            print(f"Total episodes simulated: {jnp.sum(rl_out['aux_data']['episodes'])}")
        # Save the training returns
        rl_actor_params = rl_out['actor_params']
        returns_during_rl = rl_out['aux_data']['returns']  
        actor_losses = rl_out['aux_data']['actor_losses']
        critic_losses = rl_out['aux_data']['critic_losses']
        entropy_losses = rl_out['aux_data']['entropy_losses']
        success_during_rl = rl_out['aux_data']['successes']
        failure_during_rl = rl_out['aux_data']['failures']
        timeout_during_rl = rl_out['aux_data']['timeouts']
        episodes_during_rl = rl_out['aux_data']['episodes']
        stds_during_rl = rl_out['aux_data']['stds']
        episode_count = jnp.sum(episodes_during_rl)
        window = 20 
        ## Plot RL training stats
        from matplotlib import rc
        font = {'weight' : 'regular',
                'size'   : 18}
        rc('font', **font)
        figure, ax = plt.subplots(4,2,figsize=(15,15))
        figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95, right=0.95, left=0.1, wspace=0.35)
        # Plot returns during RL
        ax[0,0].grid()
        ax[0,0].set(
            xlabel='Training Update', 
            ylabel=f'Return ({window} upd. window)', 
            title='Return'
        )
        ax[0,0].plot(
            jnp.arange(len(returns_during_rl)-(window-1))+window, 
            jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
        )
        # Plot success, failure, and timeout rates during RL
        success_rate_during_rl = success_during_rl / rl_out['aux_data']['episodes']
        failure_rate_during_rl = failure_during_rl / rl_out['aux_data']['episodes']
        timeout_rate_during_rl = timeout_during_rl / rl_out['aux_data']['episodes']
        ax[0,1].grid()
        ax[0,1].set(
            xlabel='Training Update', 
            ylabel=f'Rate ({window} upd. window)', 
            title='Success, Failure, and Timeout rates',
            ylim=(-0.1,1.1)
        )
        ax[0,1].plot(
            jnp.arange(len(success_rate_during_rl)-(window-1))+window, 
            jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
            label='Success rate',
            color='g',
        )
        ax[0,1].plot(
            jnp.arange(len(failure_rate_during_rl)-(window-1))+window, 
            jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
            label='Failure rate',
            color='r',
        )
        ax[0,1].plot(
            jnp.arange(len(timeout_rate_during_rl)-(window-1))+window, 
            jnp.convolve(timeout_rate_during_rl, jnp.ones(window,), 'valid') / window,
            label='Timeout rate',
            color='yellow',
        )
        ax[0,1].legend()
        # Plot actor loss during RL
        ax[1,0].grid()
        ax[1,0].set(
            xlabel='Training Update', 
            ylabel=f'Loss ({window} upd. window)', 
            title='Actor Loss'
        )
        ax[1,0].plot(
            jnp.arange(len(actor_losses)-(window-1))+window, 
            jnp.convolve(actor_losses, jnp.ones(window,), 'valid') / window,
        )
        # Plot critic loss during RL
        ax[1,1].grid()
        ax[1,1].set(
            xlabel='Training Update', 
            ylabel=f'Loss ({window} upd. window)', 
            title='Critic Loss'
        )
        ax[1,1].plot(
            jnp.arange(len(critic_losses)-(window-1))+window, 
            jnp.convolve(critic_losses, jnp.ones(window,), 'valid') / window,
        )
        # Plot entropy loss during RL
        entropy_window = window // 10
        ax[2,0].grid()
        ax[2,0].set(
            xlabel='Training Update', 
            ylabel=f'Loss ({entropy_window} upd. window)', 
            title='Entropy Loss'
        )
        ax[2,0].plot(
            jnp.arange(len(entropy_losses)-(entropy_window-1))+entropy_window, 
            jnp.convolve(entropy_losses, jnp.ones(entropy_window,), 'valid') / entropy_window,
        )
        # Plot episodes during RL
        ax[2,1].grid()
        ax[2,1].set(
            xlabel='Training Update', 
            ylabel=f'Episodes', 
            title='Simulated episodes'
        )
        ax[2,1].plot(
            jnp.arange(len(episodes_during_rl)), 
            jnp.cumsum(episodes_during_rl),
        )
        # Plot stds[0] during RL
        ax[3,0].grid()
        ax[3,0].set(
            xlabel='Training Update',
            ylabel='Standard deviation',
            title='First action std',
            ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
        )
        ax[3,0].plot(
            jnp.arange(len(stds_during_rl)),
            stds_during_rl[:,0],
        )
        # Plot stds[1] during RL
        ax[3,1].grid()
        ax[3,1].set(
            xlabel='Training Update',
            ylabel='Standard deviation',
            title='Second action std',
            ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
        )
        ax[3,1].plot(
            jnp.arange(len(stds_during_rl)),
            stds_during_rl[:,1],
        )
        figure.savefig(os.path.join(base_dir,rl_network_name.replace('.pkl', '.eps')), format='eps')
    else:
        print(f"Seed {seed} of DIR-SAFE training already done and saved... Remove '{rl_network_name}' to retrain.")