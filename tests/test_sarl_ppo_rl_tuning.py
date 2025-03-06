import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import os
import optax
import matplotlib.pyplot as plt
import pickle
from decimal import Decimal

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import test_k_trials, animate_trajectory, linear_decay
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.rollouts.ppo_rollouts import ppo_rl_rollout
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer
from socialjym.policies.sarl_ppo import SARLPPO

### Hyperparameters sets
hyperparamters_seed = 0
n_trials = 20
training_updates = 2_000 # For each set of hyperparameters
num_batches = 30
hp_lims = {
    'actor_learning_rates': [3e-4, 5e-6],
    'critic_learning_rates': [3e-3, 3e-4],
    'n_parallel_envs_multiplier': [1, 10],
    'buffer_capacities_multiplier': [30, 60],
    'betas_entropy': [1e-3, 5e-7],
}

### Trial loop
np.random.seed(hyperparamters_seed)
for trial in range(n_trials):
    ### Sample from hyperparameters space
    actor_lr = np.random.uniform(*hp_lims['actor_learning_rates'])
    critic_lr = np.random.uniform(*hp_lims['critic_learning_rates'])
    n_parallel_envs = num_batches * np.random.randint(*hp_lims['n_parallel_envs_multiplier'])
    buffer_capacity = n_parallel_envs * np.random.randint(*hp_lims['buffer_capacities_multiplier'])
    beta_entropy = np.random.uniform(*hp_lims['betas_entropy'])
    print(f"\n#################################### \nTRIAL {trial+1}/{n_trials}\nActor LR: {actor_lr}\nCritic LR: {critic_lr}\nParallel envs: {n_parallel_envs}\nBuffer capacity: {buffer_capacity}\nBeta entropy: {beta_entropy}")
    ### Hyperparameters initialization
    training_hyperparams = {
        'random_seed': 0,
        'kinematics': 'unicycle',
        'policy_name': 'sarl-ppo',
        'n_humans': 5, 
        'rl_training_updates': training_updates,
        'rl_parallel_envs': n_parallel_envs,
        'rl_actor_learning_rate': actor_lr, 
        'rl_critic_learning_rate': critic_lr,
        'rl_buffer_capacity': buffer_capacity,
        'rl_clip_frac': 0.2, 
        'rl_num_epochs': 10, 
        'rl_num_batches': num_batches, 
        'rl_beta_entropy': beta_entropy, 
        'lambda_gae': 0.95, 
        'humans_policy': 'hsfm',
        'scenario': 'hybrid_scenario',
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5], np.int32),
        'reward_function': 'socialnav_reward2',
        'custom_episodes': False,
        'gradient_norm_scale': 0.5,
    }
    # Initialize reward function
    if training_hyperparams['reward_function'] == 'socialnav_reward1': 
        reward_function = Reward1(
            kinematics=training_hyperparams['kinematics']
        )
    elif training_hyperparams['reward_function'] == 'socialnav_reward2':
        reward_function = Reward2(
            progress_to_goal_reward = True,
            progress_to_goal_weight = 0.03,
        )
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
    policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
    # Load IL rollout output
    with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'rb') as f:
        il_out = pickle.load(f)
    # Save the IL model parameters, buffer state, and keys
    il_actor_params = il_out['actor_params']
    il_critic_params = il_out['critic_params']
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
    # Initialize custom episodes path
    if training_hyperparams['custom_episodes']:
        rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
    else:
        rl_custom_episodes_path = None
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
        'random_seed': training_hyperparams['random_seed'],
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
        'debugging': False,
        'debugging_interval': 1,
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = ppo_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'wb') as f:
        pickle.dump(rl_out, f)
    # Load RL rollout output
    with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'rb') as f:
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
    episode_count = jnp.sum(episodes_during_rl)
    window = 500 if training_updates > 1000 else 50
    ## Plot RL training stats
    from matplotlib import rc
    font = {'weight' : 'regular',
            'size'   : 18}
    rc('font', **font)
    figure, ax = plt.subplots(3,2,figsize=(15,15))
    figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.90, right=0.95, left=0.1, wspace=0.35)
    figure.suptitle(f"Trial {trial+1}/{n_trials} - ALR: {'%.2E' % Decimal(actor_lr)}, CLR: {'%.2E' % Decimal(critic_lr)}, PEnv: {n_parallel_envs}, BCap: {buffer_capacity}, BEnt: {'%.2E' % Decimal(beta_entropy)}", fontsize=15)
    # Plot returns during RL
    ax[0,0].grid()
    ax[0,0].set(
        xlabel='Training Update', 
        ylabel=f'Return ({window} upd. window)', 
        title='Return (not episodic)'
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
        np.arange(len(success_rate_during_rl)-(window-1))+window, 
        jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Success rate',
        color='g',
    )
    ax[0,1].plot(
        np.arange(len(failure_rate_during_rl)-(window-1))+window, 
        jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Failure rate',
        color='r',
    )
    ax[0,1].plot(
        np.arange(len(timeout_rate_during_rl)-(window-1))+window, 
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
        np.arange(len(actor_losses)-(window-1))+window, 
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
    ax[2,0].grid()
    ax[2,0].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Entropy Loss'
    )
    ax[2,0].plot(
        jnp.arange(len(entropy_losses)-(window-1))+window, 
        jnp.convolve(entropy_losses, jnp.ones(window,), 'valid') / window,
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
    figure.savefig(os.path.join(os.path.dirname(__file__),f"{trial}_rl_training_plots.eps"), format='eps')