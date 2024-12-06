from jax import random
from jax.tree_util import tree_map
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import pickle
from datetime import date

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rollouts import deep_vnet_rl_rollout, deep_vnet_il_rollout
from socialjym.utils.aux_functions import epsilon_scaling_decay, plot_state, plot_trajectory, test_k_trials, save_policy_params, decimal_to_binary
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

random_seed = 0
n_il_epochs = 50
n_rl_episodes = 30_000
n_test_trials = 1000
test_n_humans = [5,15,25]
humans_policy = 'hsfm'
# Reward terms params
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.07 # High rotation penalty weight
w_bound = 2. # Rotation bound

# Initialize arrays to store training metrics
loss_during_il = jnp.empty((2**len(reward_terms),n_il_epochs))
returns_after_il = jnp.empty((2**len(reward_terms),len(test_n_humans),n_test_trials))
success_rate_after_il = jnp.empty((2**len(reward_terms),len(test_n_humans)))
returns_during_rl = jnp.empty((2**len(reward_terms),n_rl_episodes))
returns_after_rl = jnp.empty((2**len(reward_terms),len(test_n_humans),n_test_trials))
success_rate_after_rl = jnp.empty((2**len(reward_terms),len(test_n_humans)))

# Initialize dictionaries to store testing metrics
all_metrics_after_il = {
    "successes": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "collisions": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "timeouts": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "returns": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "times_to_goal": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "min_distance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "space_compliance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "episodic_spl": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "path_length": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials))
}
all_metrics_after_rl = {
    "successes": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "collisions": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "timeouts": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "returns": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "times_to_goal": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "min_distance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "space_compliance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "episodic_spl": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "path_length": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials))
}

### TRAINING LOOP FOR EACH REWARD FUNCTION
for reward_type_decimal in range(2**(len(reward_terms))):
    print(f"\n#### REWARD {reward_type_decimal} ####\n")
    binary_reward = decimal_to_binary(reward_type_decimal, len(reward_terms))
    ### INITIALIZE REWARD FUNCTION
    reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = binary_reward[0],
        time_penalty_reward = binary_reward[1],
        high_rotation_penalty_reward = binary_reward[2],
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
        time_penalty=wt,
        angular_speed_bound=w_bound,
        angular_speed_penalty_weight=wr
    )
    # Initialize training parameters
    training_hyperparams = {
        'random_seed': random_seed,
        'kinematics': 'unicycle',
        'policy_name': 'sarl', #
        'n_humans': 5,  # CADRL uses 1, SARL uses 5
        'il_training_episodes': 2_000,
        'il_learning_rate': 0.01,
        'il_num_epochs': n_il_epochs, # Number of epochs to train the model after ending IL
        'rl_training_episodes': n_rl_episodes,
        'rl_learning_rate': 0.001,
        'rl_num_batches': 100, # Number of batches to train the model after each RL episode
        'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
        'epsilon_start': 0.5,
        'epsilon_end': 0.1,
        'epsilon_decay': 4_000,
        'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
        'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
        'humans_policy': humans_policy,
        'scenario': 'circular_crossing',
        'hybrid_scenario_subset': jnp.array([0,1,2,3], dtype=jnp.int32),
        'reward_function': reward_function.type,
        'custom_episodes': False, # If True, the episodes are loaded from a predefined set
    }
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
    policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'])
    initial_vnet_params = policy.model.init(training_hyperparams['random_seed'], jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
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
    loss_during_il = loss_during_il.at[reward_type_decimal].set(il_out['losses'])
    # Execute tests to evaluate return after IL
    for test, n_humans in enumerate(test_n_humans):
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
            n_test_trials, 
            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
            test_env, 
            policy, 
            il_model_params, 
            reward_function.time_limit)
        returns_after_il = returns_after_il.at[reward_type_decimal,test].set(metrics_after_il['returns'])
        success_rate_after_il = success_rate_after_il.at[reward_type_decimal,test].set(metrics_after_il['successes'] / n_test_trials)
        all_metrics_after_il = tree_map(lambda x, y: x.at[reward_type_decimal,test].set(y), all_metrics_after_il, metrics_after_il)
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
    returns_during_rl = returns_during_rl.at[reward_type_decimal].set(rl_out['returns']) 
    # Save the policy parameters
    save_policy_params(
        training_hyperparams['policy_name'], 
        rl_model_params, 
        env.get_parameters(), 
        reward_function.get_parameters(), 
        training_hyperparams, 
        os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/"),
        filename=f"sarl_{humans_policy}_unicycle_reward_{reward_type_decimal}_{training_hyperparams['scenario']}_{date.today().strftime('%d_%m_%Y')}"
    )
    # Execute tests to evaluate return after RL
    for test, n_humans in enumerate(test_n_humans):
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
            n_test_trials, 
            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
            test_env, 
            policy, 
            rl_model_params, 
            reward_function.time_limit)
        returns_after_rl = returns_after_rl.at[reward_type_decimal,test].set(metrics_after_rl['returns'])
        success_rate_after_rl = success_rate_after_rl.at[reward_type_decimal,test].set(metrics_after_rl['successes'] / n_test_trials)
        all_metrics_after_rl = tree_map(lambda x, y: x.at[reward_type_decimal,test].set(y), all_metrics_after_rl, metrics_after_rl)

# Save all output data
training_data = {
    'loss_during_il': loss_during_il,
    'returns_after_il': returns_after_il,
    'success_rate_after_il': success_rate_after_il,
    'returns_during_rl': returns_during_rl,
    'returns_after_rl': returns_after_rl,
    'success_rate_after_rl': success_rate_after_rl
}
with open(os.path.join(os.path.dirname(__file__),"training_data_ablation_study.pkl"), 'wb') as f:
    pickle.dump(training_data, f)
with open(os.path.join(os.path.dirname(__file__),"metrics_after_il_ablation_study.pkl"), 'wb') as f:
    pickle.dump(all_metrics_after_il, f)
with open(os.path.join(os.path.dirname(__file__),"metrics_after_rl_ablation_study.pkl"), 'wb') as f:
    pickle.dump(all_metrics_after_rl, f)