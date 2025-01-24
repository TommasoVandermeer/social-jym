from jax.tree_util import tree_map
import jax.numpy as jnp
import optax
import os
import pickle
from datetime import date

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rollouts import deep_vnet_rl_rollout, deep_vnet_il_rollout
from socialjym.utils.aux_functions import epsilon_scaling_decay, test_k_trials, save_policy_params, decimal_to_binary
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

####################################################################################################
### HOW TO REPLICATE RESULTS IN THE SPRINGER BOOK CHAPTER
###
###
####################################################################################################

### PARAMETERS
random_seed = 0
n_il_epochs = 50
n_rl_episodes = 30_000
n_test_trials = 1000
# Training settings
train_n_humans = 5
train_scenarios = ['delayed_circular_crossing'] # ['circular_crossing', 'parallel_traffic', 'perpendicular_traffic', 'robot_crowding', 'delayed_circular_crossing', 'hybrid_scenario']
train_envs = ['hsfm'] # ['sfm', 'hsfm']
train_hybrid_scenario_subset = jnp.array([1,2,3,4], dtype=jnp.int32) # Pick the scenarios to include in the hybrid scenario
# Testing settings
test_n_humans = [5,15,25]
test_scenarios = ['delayed_circular_crossing'] # ['circular_crossing', 'parallel_traffic', 'perpendicular_traffic', 'robot_crowding', 'delayed_circular_crossing']
test_envs = ['sfm', 'hsfm']
# Reward terms parameters
only_base_and_full_rewards = False
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.035 # High rotation penalty weight
w_bound = 1. # Rotation bound

### INITIALIZATION
# Initialize arrays to store training and testing metrics
empty_trials_outcomes_array = jnp.zeros((len(train_scenarios),len(train_envs),2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans)))
empty_trials_metrics_array = jnp.zeros((len(train_scenarios),len(train_envs),2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans),n_test_trials))
training_data = {
    'loss_during_il': jnp.zeros((len(train_scenarios),len(train_envs),2**len(reward_terms),n_il_epochs)),
    'loss_during_rl': jnp.zeros((len(train_scenarios),len(train_envs),2**len(reward_terms),n_rl_episodes)),
    'returns_during_rl': jnp.zeros((len(train_scenarios),len(train_envs),2**len(reward_terms),n_rl_episodes)),
    'returns_after_il': empty_trials_metrics_array,
    'success_rate_after_il': empty_trials_outcomes_array,
    'returns_after_rl': empty_trials_metrics_array,
    'success_rate_after_rl': empty_trials_outcomes_array
}
all_metrics_after_il = {
    "successes": empty_trials_outcomes_array, 
    "collisions": empty_trials_outcomes_array, 
    "timeouts": empty_trials_outcomes_array, 
    "returns": empty_trials_metrics_array,
    "times_to_goal": empty_trials_metrics_array,
    "average_speed": empty_trials_metrics_array,
    "average_acceleration": empty_trials_metrics_array,
    "average_jerk": empty_trials_metrics_array,
    "average_angular_speed": empty_trials_metrics_array,
    "average_angular_acceleration": empty_trials_metrics_array,
    "average_angular_jerk": empty_trials_metrics_array,
    "min_distance": empty_trials_metrics_array,
    "space_compliance": empty_trials_metrics_array,
    "episodic_spl": empty_trials_metrics_array,
    "path_length": empty_trials_metrics_array
}
all_metrics_after_rl = all_metrics_after_il.copy()

### TRAINING LOOP
for ts_idx, train_scenario in enumerate(train_scenarios):
    for te_idx, humans_policy in enumerate(train_envs):
        for reward_type_decimal in range(2**(len(reward_terms))):
            if (only_base_and_full_rewards) and (reward_type_decimal > 0 and reward_type_decimal < 2**(len(reward_terms))-1):
                continue
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
                'n_humans': train_n_humans,  # CADRL uses 1, SARL uses 5
                'il_training_episodes': 2_000,
                'il_learning_rate': 0.01,
                'il_num_epochs': n_il_epochs, # Number of epochs to train the model after ending IL
                'rl_training_episodes': n_rl_episodes,
                'rl_learning_rate': 0.001,
                'rl_num_batches': 100, # Number of batches to train the model after each RL episode
                'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
                'epsilon_start': 0.5,
                'epsilon_end': 0.1,
                'epsilon_decay': int(0.4*n_rl_episodes), # Number of episodes to decay epsilon from start to end
                'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
                'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
                'humans_policy': humans_policy,
                'scenario': train_scenario,
                'hybrid_scenario_subset': train_hybrid_scenario_subset,
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
            training_data['loss_during_il'] = training_data['loss_during_il'].at[ts_idx,te_idx,reward_type_decimal].set(il_out['losses'])
            # Save the IL policy parameters
            if not os.path.exists(os.path.join(os.path.dirname(__file__),'policies')):
                os.makedirs(os.path.join(os.path.dirname(__file__),'policies'))
            policies_dir = os.path.join(os.path.dirname(__file__),'policies')
            save_policy_params(
                training_hyperparams['policy_name'], 
                il_model_params, 
                env.get_parameters(), 
                reward_function.get_parameters(), 
                training_hyperparams, 
                policies_dir,
                filename=f"sarl_after_IL_{humans_policy}_unicycle_reward_{reward_type_decimal}_{training_hyperparams['scenario']}_{date.today().strftime('%d_%m_%Y')}"
            )
            # Execute tests to evaluate return after IL
            for s_idx, scenario in enumerate(test_scenarios):
                for e_idx, test_human_policy in enumerate(test_envs):
                    print(f"## Test - scenario {scenario} - humans policy {test_human_policy} ##")
                    for h_idx, n_humans in enumerate(test_n_humans):
                        test_env_params = {
                            'robot_radius': 0.3,
                            'n_humans': n_humans,
                            'robot_dt': 0.25,
                            'humans_dt': 0.01,
                            'robot_visible': True,
                            'scenario': scenario,
                            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
                            'humans_policy': test_human_policy,
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
                        training_data['returns_after_il'] = training_data['returns_after_il'].at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_il['returns'])
                        training_data['success_rate_after_il'] = training_data['success_rate_after_il'].at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_il['successes'] / n_test_trials)
                        all_metrics_after_il = tree_map(lambda x, y: x.at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(y), all_metrics_after_il, metrics_after_il)
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
            # Save the training returns and losses
            rl_model_params = rl_out['model_params']
            training_data['loss_during_rl'] = training_data['loss_during_rl'].at[ts_idx,te_idx,reward_type_decimal].set(rl_out['losses'])
            training_data['returns_during_rl'] = training_data['returns_during_rl'].at[ts_idx,te_idx,reward_type_decimal].set(rl_out['returns']) 
            # Save the RL policy parameters
            save_policy_params(
                training_hyperparams['policy_name'], 
                rl_model_params, 
                env.get_parameters(), 
                reward_function.get_parameters(), 
                training_hyperparams, 
                policies_dir,
                filename=f"sarl_after_RL_{humans_policy}_unicycle_reward_{reward_type_decimal}_{training_hyperparams['scenario']}_{date.today().strftime('%d_%m_%Y')}"
            )
            # Execute tests to evaluate return after RL
            for s_idx, scenario in enumerate(test_scenarios):
                for e_idx, test_human_policy in enumerate(test_envs):
                    print(f"## Test - scenario {scenario} - humans policy {test_human_policy} ##")
                    for h_idx, n_humans in enumerate(test_n_humans):
                        test_env_params = {
                            'robot_radius': 0.3,
                            'n_humans': n_humans,
                            'robot_dt': 0.25,
                            'humans_dt': 0.01,
                            'robot_visible': True,
                            'scenario': scenario,
                            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
                            'humans_policy': test_human_policy,
                            'circle_radius': 7,
                            'reward_function': reward_function,
                            'kinematics': training_hyperparams['kinematics'],
                        }
                        test_env = SocialNav(**test_env_params)
                        metrics_after_rl = test_k_trials(
                            n_test_trials, 
                            training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'] + n_test_trials, 
                            test_env, 
                            policy, 
                            rl_model_params, 
                            reward_function.time_limit)
                        training_data['returns_after_rl'] = training_data['returns_after_rl'].at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_rl['returns'])
                        training_data['success_rate_after_rl'] = training_data['success_rate_after_rl'].at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_rl['successes'] / n_test_trials)
                        all_metrics_after_rl = tree_map(lambda x, y: x.at[ts_idx,te_idx,reward_type_decimal,s_idx,e_idx,h_idx].set(y), all_metrics_after_rl, metrics_after_rl)

### SAVE ALL DATA
if not os.path.exists(os.path.join(os.path.dirname(__file__),'results')):
    os.makedirs(os.path.join(os.path.dirname(__file__),'results'))
results_dir = os.path.join(os.path.dirname(__file__),'results')
with open(os.path.join(results_dir,"training_data_ablation_study.pkl"), 'wb') as f:
    pickle.dump(training_data, f)
with open(os.path.join(results_dir,"metrics_after_il_ablation_study.pkl"), 'wb') as f:
    pickle.dump(all_metrics_after_il, f)
with open(os.path.join(results_dir,"metrics_after_rl_ablation_study.pkl"), 'wb') as f:
    pickle.dump(all_metrics_after_rl, f)