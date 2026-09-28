import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import os
import optax
import matplotlib.pyplot as plt
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import test_k_trials, animate_trajectory, linear_decay
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.rollouts.act_cri_rollouts import actor_critic_il_rollout
from socialjym.utils.rollouts.ppo_rollouts import ppo_rl_rollout
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer
from socialjym.policies.sarl_ppo import SARLPPO

### Hyperparameters
n_humans_for_tests = [5, 10, 15]
n_trials = 1000
n_parallel_envs = 50 
training_updates = 10_000
rl_debugging_interval = 5
training_hyperparams = {
    'random_seed': 0,
    'kinematics': 'unicycle', # 'unicycle' or 'holonomic'
    'policy_name': 'sarl-ppo',
    'n_humans': 2, 
    'circle_radius': 2,
    'il_buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'il_training_episodes': 2_000,
    'il_actor_learning_rate': 0.001,
    'il_critic_learning_rate': 0.01,
    'il_num_epochs': 50, # Number of epochs to train the model after ending IL
    'il_batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
    'rl_training_updates': training_updates,
    'rl_parallel_envs': n_parallel_envs,
    'rl_actor_learning_rate': 1e-5, 
    'rl_critic_learning_rate': 1e-4, # 1e-3
    'rl_buffer_capacity': 3_000, # Number of experiences to sample from the replay buffer for each model update
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 10, # 10
    'rl_num_batches': 30, # 30
    'rl_beta_entropy': 2e-5, 
    'lambda_gae': 0.95, # 0.95
    'humans_policy': 'hsfm',
    'scenario': 'circular_crossing',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5], np.int32), # Subset of the hybrid scenarios to use for training
    'reward_function': 'socialnav_reward2',
    'custom_episodes': False, # If True, the episodes are loaded from a predefined set
    'gradient_norm_scale': 0.5, # Scale the gradient norm by this value
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
    'circle_radius': training_hyperparams['circle_radius'],
    'reward_function': reward_function,
    'kinematics': training_hyperparams['kinematics'],
}
# Initialize environment
env = SocialNav(**env_params)
# Initialize robot policy and vnet params
policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
initial_actor_params = policy.actor.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
initial_critic_params = policy.critic.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
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
# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    il_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/il_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    il_custom_episodes_path = None
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
    'custom_episodes': il_custom_episodes_path
}

# IMITATION LEARNING ROLLOUT
il_out = actor_critic_il_rollout(**il_rollout_params)

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
        'circle_radius': training_hyperparams['circle_radius'],
        'reward_function': reward_function,
        'kinematics': training_hyperparams['kinematics'],
    }
    test_env = SocialNav(**test_env_params)
    metrics_after_il = test_k_trials(
        n_trials, 
        training_hyperparams['il_training_episodes'], 
        test_env, 
        policy, 
        il_out['actor_params'], 
        reward_function.time_limit)

# Plot losses during IL
figure, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Actor Loss during IL training'
)
ax[0].plot(
    np.arange(len(il_out['actor_losses'])), 
    il_out['actor_losses'],
)
ax[1].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Critic Loss during IL training'
)
ax[1].plot(
    np.arange(len(il_out['critic_losses'])), 
    il_out['critic_losses'],
)
figure.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il.eps"), format='eps')
plt.close(figure)

# Save IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'wb') as f:
    pickle.dump(il_out, f)

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
    'debugging_interval': rl_debugging_interval,
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
episode_count = jnp.sum(rl_out['aux_data']['episodes'])
window = 500 if training_updates > 1000 else 50

# Plot returns during RL
figure, ax = plt.subplots(1,1,figsize=(10,5))
ax.set(
    xlabel='Training Episode', 
    ylabel=f'Return moving average over {window} episodes window', 
    title='Return during RL'
)
ax.plot(
    np.arange(len(returns_during_rl)-(window-1))+window, 
    jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"returns_during_rl.eps"), format='eps')
# Plot actor loss during RL
figure, ax = plt.subplots(1,1,figsize=(10,5))
ax.set(
    xlabel='Training Episode', 
    ylabel=f'Loss moving average over {window} episodes window', 
    title='Actor Loss during RL'
)
ax.plot(
    np.arange(len(actor_losses)-(window-1))+window, 
    jnp.convolve(actor_losses, jnp.ones(window,), 'valid') / window,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"actor_losses.eps"), format='eps')
# Plot critic loss during RL
figure, ax = plt.subplots(1,1,figsize=(10,5))
ax.set(
    xlabel='Training Episode', 
    ylabel=f'Loss moving average over {window} episodes window', 
    title='Critic Loss during RL'
)
ax.plot(
    np.arange(len(critic_losses)-(window-1))+window, 
    jnp.convolve(critic_losses, jnp.ones(window,), 'valid') / window,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"critic_losses.eps"), format='eps')
# Plot entropy loss during RL
figure, ax = plt.subplots(1,1,figsize=(10,5))
ax.set(
    xlabel='Training Episode', 
    ylabel=f'Loss moving average over {window} episodes window', 
    title='Entropy Loss during RL'
)
ax.plot(
    np.arange(len(entropy_losses)-(window-1))+window, 
    jnp.convolve(entropy_losses, jnp.ones(window,), 'valid') / window,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"entropy_losses.eps"), format='eps')
# Plot success, failure, and timeout rates during RL
success_rate_during_rl = success_during_rl / rl_out['aux_data']['episodes']
failure_rate_during_rl = failure_during_rl / rl_out['aux_data']['episodes']
timeout_rate_during_rl = timeout_during_rl / rl_out['aux_data']['episodes']
figure, ax = plt.subplots(1,1,figsize=(10,5))
ax.set(
    xlabel='Training Episode', 
    ylabel='Rate moving average over {window} episodes window', 
    title='Success, Failure, and Timeout rates during RL'
)
ax.plot(
    np.arange(len(success_rate_during_rl)-(window-1))+window, 
    jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Success rate',
    color='g',
)
ax.plot(
    np.arange(len(failure_rate_during_rl)-(window-1))+window, 
    jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Failure rate',
    color='r',
)
ax.plot(
    np.arange(len(timeout_rate_during_rl)-(window-1))+window, 
    jnp.convolve(timeout_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Timeout rate',
    color='yellow',
)
ax.legend()
figure.savefig(os.path.join(os.path.dirname(__file__),"outcome_rates_during_rl.eps"), format='eps')

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
        'circle_radius': training_hyperparams['circle_radius'],
        'reward_function': reward_function,
        'kinematics': training_hyperparams['kinematics'],
    }
    test_env = SocialNav(**test_env_params)
    metrics_after_rl = test_k_trials(
        n_trials, 
        training_hyperparams['il_training_episodes'] + episode_count, 
        test_env, 
        policy, 
        rl_actor_params, 
        reward_function.time_limit)