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
from socialjym.utils.rollouts.a2c_rollouts import a2c_il_rollout, a2c_rl_rollout
from socialjym.utils.replay_buffers.base_a2c_buffer import BaseA2CBuffer
from socialjym.policies.sarl_a2c import SARLA2C

### Hyperparameters
n_humans_for_tests = [5, 10, 15]
n_trials = 1000
training_updates = 30_000
training_hyperparams = {
    'random_seed': 0,
    'kinematics': 'unicycle', # 'unicycle' or 'holonomic'
    'policy_name': 'sarl-a2c', # 'cadrl' or 'sarl'
    'n_humans': 5,  # CADRL uses 1, SARL uses 5
    'il_training_episodes': 2_000,
    'il_actor_learning_rate': 0.001,
    'il_critic_learning_rate': 0.01,
    'il_num_epochs': 50, # Number of epochs to train the model after ending IL
    'il_batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
    'rl_training_updates': training_updates,
    'rl_actor_learning_rate': 0.000015, # 0.00002
    'rl_critic_learning_rate': 0.00015, # 0.0002
    'rl_batch_size': 2_000, # Number of experiences to sample from the replay buffer for each model update
    'rl_sigma_start': 0.2,
    'rl_sigma_end': 0.02, # 0.02
    'rl_sigma_decay': int(0.4 * training_updates), # Training updates to reach the minimum sigma
    'rl_sigma_decay_fn': linear_decay,
    'rl_beta_entropy': 0.0002, # 0.0002
    'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'humans_policy': 'hsfm',
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5], np.int32), # Subset of the hybrid scenarios to use for training
    'reward_function': 'socialnav_reward2',
    'custom_episodes': False, # If True, the episodes are loaded from a predefined set
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
policy = SARLA2C(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
initial_actor_params = policy.actor.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
initial_critic_params = policy.critic.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
# Initialize replay buffer
replay_buffer = BaseA2CBuffer(training_hyperparams['buffer_size'], training_hyperparams['il_batch_size'])
# Initialize IL optimizer
actor_optimizer = optax.sgd(learning_rate=training_hyperparams['il_actor_learning_rate'], momentum=0.9)
critic_optimizer = optax.sgd(learning_rate=training_hyperparams['il_critic_learning_rate'], momentum=0.9)
# Initialize buffer state
buffer_state = {
    'inputs': jnp.empty((training_hyperparams['buffer_size'], env.n_humans, policy.vnet_input_size)),
    'critic_targets': jnp.empty((training_hyperparams['buffer_size'],1)),
    'sample_actions': jnp.empty((training_hyperparams['buffer_size'], 2)),
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
    'buffer_capacity': training_hyperparams['buffer_size'],
    'num_epochs': training_hyperparams['il_num_epochs'],
    'batch_size': training_hyperparams['il_batch_size'],
    'custom_episodes': il_custom_episodes_path
}

# IMITATION LEARNING ROLLOUT
il_out = a2c_il_rollout(**il_rollout_params)

# Save IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'wb') as f:
    pickle.dump(il_out, f)

# Load IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'rb') as f:
    il_out = pickle.load(f)

# Save the IL model parameters, buffer state, and keys
il_actor_params = il_out['actor_params']
il_critic_params = il_out['critic_params']
critic_loss_during_il = il_out['critic_losses']
actor_loss_during_il = il_out['actor_losses']

# Plot losses during IL
figure, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Actor Loss during IL training'
)
ax[0].plot(
    np.arange(len(actor_loss_during_il)), 
    actor_loss_during_il,
)
ax[1].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Critic Loss during IL training'
)
ax[1].plot(
    np.arange(len(critic_loss_during_il)), 
    critic_loss_during_il,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il.eps"), format='eps')
plt.close(figure)

# # Watch some trials after IL
# test_env_params = {
#     'robot_radius': 0.3,
#     'n_humans': 5,
#     'robot_dt': 0.25,
#     'humans_dt': 0.01,
#     'robot_visible': True,
#     'scenario': training_hyperparams['scenario'],
#     'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
#     'humans_policy': training_hyperparams['humans_policy'],
#     'circle_radius': 7,
#     'reward_function': reward_function,
#     'kinematics': training_hyperparams['kinematics'],
# }
# test_env = SocialNav(**test_env_params)
# i = 0
# while(input("Do you want to watch some trials after IL? (y/n): ") == 'y'):
#     policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + i)
#     state, reset_key, obs, info, outcome = env.reset(reset_key)
#     all_states = np.array([state])
#     while outcome["nothing"]:
#         action, policy_key, _, sampled_action, distrs = policy.act(policy_key, obs, info, il_actor_params, sigma=0.)
#         state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True) 
#         all_states = np.vstack((all_states, [state]))
#     animate_trajectory(
#         all_states, 
#         info['humans_parameters'][:,0], 
#         env.robot_radius, 
#         env_params['humans_policy'],
#         info['robot_goal'],
#         info['current_scenario'],
#         robot_dt=env_params['robot_dt'],
#         kinematics=env_params['kinematics'])
#     i += 1

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
        training_hyperparams['il_training_episodes'], 
        test_env, 
        policy, 
        il_actor_params, 
        reward_function.time_limit)
    
# Initialize RL optimizer
actor_optimizer = optax.adam(
    learning_rate=optax.schedules.linear_schedule(
        init_value=training_hyperparams['rl_actor_learning_rate'], 
        end_value=0.0, 
        transition_steps=training_hyperparams['rl_training_updates'],
        transition_begin=0
    ), 
    eps=1e-7, 
    b1=0.9
)
# critic_optimizer = optax.sgd(learning_rate=training_hyperparams['rl_critic_learning_rate'], momentum=0.9)
critic_optimizer = optax.sgd(learning_rate=training_hyperparams['rl_critic_learning_rate'], momentum=0.9)

# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    rl_custom_episodes_path = None

# Initialize RL replay buffer
replay_buffer = BaseA2CBuffer(training_hyperparams['buffer_size'], training_hyperparams['rl_batch_size'])
# Initialize RL buffer state
buffer_state = {
    'inputs': jnp.empty((training_hyperparams['buffer_size'], env.n_humans, policy.vnet_input_size)),
    'critic_targets': jnp.empty((training_hyperparams['buffer_size'], 1)),
    'sample_actions': jnp.empty((training_hyperparams['buffer_size'], 2)),
}

# Initialize RL rollout params
rl_rollout_params = {
    'initial_actor_params': il_actor_params,
    'initial_critic_params': il_critic_params,
    'train_updates': training_hyperparams['rl_training_updates'],
    'random_seed': training_hyperparams['random_seed'] + training_hyperparams['il_training_episodes'],
    'actor_optimizer': actor_optimizer,
    'critic_optimizer': critic_optimizer,
    'buffer_state': buffer_state,
    'buffer_capacity': training_hyperparams['buffer_size'],
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'sigma_start': training_hyperparams['rl_sigma_start'],
    'sigma_end': training_hyperparams['rl_sigma_end'],
    'sigma_decay': training_hyperparams['rl_sigma_decay'],
    'sigma_decay_fn': training_hyperparams['rl_sigma_decay_fn'],
    'beta_entropy': training_hyperparams['rl_beta_entropy'],
    'debugging': True,
}

# REINFORCEMENT LEARNING ROLLOUT
rl_out = a2c_rl_rollout(**rl_rollout_params)
print(f"Total episodes simulated: {rl_out['episode_count']}")

# Save RL rollout output
with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'wb') as f:
    pickle.dump(rl_out, f)

# Load RL rollout output
with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'rb') as f:
    rl_out = pickle.load(f)

# Save the training returns
rl_actor_params = rl_out['actor_params']
returns_during_rl = rl_out['returns']  
episode_count = rl_out['episode_count']

# Plot returns during RL
figure, ax = plt.subplots(1,1,figsize=(10,5))
window = 1000
ax.set(
    xlabel='Training Episode', 
    ylabel=f'Return moving average over {window} episodes window', 
    title='Return during RL'
)
ax.plot(
    np.arange(len(returns_during_rl[:episode_count])-(window-1))+window, 
    jnp.convolve(returns_during_rl[:episode_count], jnp.ones(window,), 'valid') / window,
)
figure.savefig(os.path.join(os.path.dirname(__file__),"returns_during_rl.eps"), format='eps')

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
        training_hyperparams['il_training_episodes'] + episode_count, 
        test_env, 
        policy, 
        rl_actor_params, 
        reward_function.time_limit)