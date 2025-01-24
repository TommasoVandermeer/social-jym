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
import math
import pandas as pd

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.deep_vnet_rollouts import deep_vnet_rl_rollout, deep_vnet_il_rollout
from socialjym.utils.aux_functions import epsilon_scaling_decay, plot_state, plot_trajectory, test_k_trials, save_policy_params, decimal_to_binary
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

only_base_and_full_rewards = False
random_seed = 0
n_il_epochs = 50
n_rl_episodes = 30_000
n_test_trials = 1000
# Train and test environments
test_n_humans = [5,15,25]
humans_policy = 'hsfm'
train_scenario = 'delayed_circular_crossing'
train_hybrid_scenario_subset = jnp.array([1,2,3,4], dtype=jnp.int32) # Exclude normal circular crossing
test_scenarios = ['delayed_circular_crossing'] # ['parallel_traffic', 'perpendicular_traffic', 'robot_crowding', 'delayed_circular_crossing']
scenarios_labels = ['DCC'] # ["PaT", "PeT", "RC", "DCC"]
plot_one_test_scenario_only = None # If not None, plot only this test scenario
test_envs = ['sfm', 'hsfm']
# Reward terms parameters
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.035 # High rotation penalty weight
w_bound = 1. # Rotation bound

# Initialize arrays to store training metrics
loss_during_il = jnp.zeros((2**len(reward_terms),n_il_epochs))
loss_during_rl = jnp.zeros((2**len(reward_terms),n_rl_episodes))
returns_during_rl = jnp.zeros((2**len(reward_terms),n_rl_episodes))
returns_after_il = jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans),n_test_trials))
success_rate_after_il = jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans)))
returns_after_rl = jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans),n_test_trials))
success_rate_after_rl = jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans)))

# Initialize dictionaries to store testing metrics
all_metrics_after_il = {
    "successes": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "collisions": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "timeouts": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "returns": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "times_to_goal": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_speed": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_acceleration": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_jerk": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_speed": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_acceleration": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_jerk": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "min_distance": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "space_compliance": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "episodic_spl": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "path_length": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials))
}
all_metrics_after_rl = {
    "successes": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "collisions": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "timeouts": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans))), 
    "returns": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "times_to_goal": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_speed": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_acceleration": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_jerk": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_speed": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_acceleration": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "average_angular_jerk": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "min_distance": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "space_compliance": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "episodic_spl": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials)),
    "path_length": jnp.zeros((2**len(reward_terms),len(test_scenarios),len(test_envs),len(test_n_humans), n_test_trials))
}

### TRAINING LOOP FOR EACH REWARD FUNCTION
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
    loss_during_il = loss_during_il.at[reward_type_decimal].set(il_out['losses'])
    # Save the IL policy parameters
    save_policy_params(
        training_hyperparams['policy_name'], 
        il_model_params, 
        env.get_parameters(), 
        reward_function.get_parameters(), 
        training_hyperparams, 
        os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/"),
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
                returns_after_il = returns_after_il.at[reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_il['returns'])
                success_rate_after_il = success_rate_after_il.at[reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_il['successes'] / n_test_trials)
                all_metrics_after_il = tree_map(lambda x, y: x.at[reward_type_decimal,s_idx,e_idx,h_idx].set(y), all_metrics_after_il, metrics_after_il)
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
    loss_during_rl = loss_during_rl.at[reward_type_decimal].set(rl_out['losses'])
    returns_during_rl = returns_during_rl.at[reward_type_decimal].set(rl_out['returns']) 
    # Save the RL policy parameters
    save_policy_params(
        training_hyperparams['policy_name'], 
        rl_model_params, 
        env.get_parameters(), 
        reward_function.get_parameters(), 
        training_hyperparams, 
        os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/"),
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
                returns_after_rl = returns_after_rl.at[reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_rl['returns'])
                success_rate_after_rl = success_rate_after_rl.at[reward_type_decimal,s_idx,e_idx,h_idx].set(metrics_after_rl['successes'] / n_test_trials)
                all_metrics_after_rl = tree_map(lambda x, y: x.at[reward_type_decimal,s_idx,e_idx,h_idx].set(y), all_metrics_after_rl, metrics_after_rl)

# Save all output data
training_data = {
    'loss_during_il': loss_during_il,
    'loss_during_rl': loss_during_rl,
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

# Load all output data
# Load results
with open(os.path.join(os.path.dirname(__file__), "metrics_after_il_ablation_study.pkl"), "rb") as f:
    all_metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "metrics_after_rl_ablation_study.pkl"), "rb") as f:
    all_metrics_after_rl = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "training_data_ablation_study.pkl"), "rb") as f:
    training_data = pickle.load(f)

#### PLOTS ####
## TRAINING DATA ##

# Create figure folder
if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"))
figure_folder = os.path.join(os.path.dirname(__file__), "figures")

# Compute scenario idx
if plot_one_test_scenario_only is not None:
    test_scen_idx = test_scenarios.index(plot_one_test_scenario_only)

# Plot loss curve during IL for each reward
figure, ax = plt.subplots(figsize=(10,10))
ax.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each reward')
ax.grid()
for loss in range(len(training_data["loss_during_il"])):
    if (only_base_and_full_rewards) and (loss > 0 and loss < 2**(len(reward_terms))-1):
        continue
    ax.plot(
        np.arange(len(training_data["loss_during_il"][loss])), 
        training_data["loss_during_il"][loss],
        color = list(mcolors.TABLEAU_COLORS.values())[loss])
if (only_base_and_full_rewards):
    ax.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    ax.legend(["Reward {}".format(i) for i in range(len(training_data["loss_during_il"]))])
figure.savefig(os.path.join(figure_folder,"loss_curves_during_il_ablation_study.eps"), format='eps')

# Plot returns during RL for each reward
figure, ax = plt.subplots(figsize=(10,10))
figure.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85)
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each reward')
ax.grid()
for reward in range(len(training_data["returns_during_rl"])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    ax.plot(
        np.arange(len(training_data["returns_during_rl"][reward])-(window-1))+window, 
        np.convolve(training_data["returns_during_rl"][reward], np.ones(window,), 'valid') / window,
        color = list(mcolors.TABLEAU_COLORS.values())[reward])
if (only_base_and_full_rewards):
    figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    figure.legend(["Reward {}".format(i) for i in range(len(training_data["returns_during_rl"]))], loc="center right")
figure.savefig(os.path.join(figure_folder,"return_curves_during_rl_ablation_study.eps"), format='eps')

# Plot return after IL and RL for each reward
figure, ax = plt.subplots(int(len(training_data['returns_after_il'])/2), 2, figsize=(10,10))
figure.suptitle(f"Return after IL and RL training for each test.\nAll test scenarios - All test environments - {n_test_trials} trials")
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="After IL"),
    Line2D([0], [0], color="lightcoral", lw=4, label="After RL")
]
figure.legend(handles=legend_elements, loc="center right")
figure.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.05, top=0.90, right=0.87)
for reward in range(len(training_data['returns_after_il'])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    i = reward // 2
    j = reward % 2
    ax[i,j].set(
        xlabel='N° humans', 
        ylabel='Return', 
        title=f'REWARD {reward}',
        ylim=[-0.5,0.5])
    ax[i,j].grid()
    # Aggregate data by test scenario
    unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)*len(test_envs)))
    for h_idx in range(len(test_n_humans)):
        unclean_data = unclean_data.at[h_idx].set(training_data['returns_after_il'][reward,:,:,h_idx,:].flatten())
    # Clean data from NaNs
    data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
    data = data.dropna()
    ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                tick_labels=test_n_humans,
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True, 
                zorder=1)
    # Aggregate data by test scenario
    unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)*len(test_envs)))
    for h_idx in range(len(test_n_humans)):
        unclean_data = unclean_data.at[h_idx].set(training_data['returns_after_rl'][reward,:,:,h_idx,:].flatten())
    # Clean data from NaNs
    data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
    data = data.dropna()
    ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                tick_labels=test_n_humans,
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
                showfliers=False,
                showmeans=True,
                zorder=2)
figure.savefig(os.path.join(figure_folder,f"return_boxplots_after_il_and_rl_ablation_study.pdf"), format='pdf')

# Plot success rate after IL and RL for each reward
figure, ax = plt.subplots(2,1,figsize=(10,10))
figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.90, right=0.85)
ax[0].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after IL training for each test - {n_test_trials} trials', 
    xticks=np.arange(len(test_n_humans)), 
    xticklabels=test_n_humans, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
ax[0].grid()
ax[1].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after RL training for each test - {n_test_trials} trials', 
    xticks=np.arange(len(test_n_humans)), 
    xticklabels=test_n_humans, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
ax[1].grid()
for reward in range(len(training_data['success_rate_after_il'])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    if plot_one_test_scenario_only is not None:
        ax[0].plot(np.mean(training_data['success_rate_after_il'][reward][test_scen_idx], axis=0))
        ax[1].plot(np.mean(training_data['success_rate_after_rl'][reward][test_scen_idx], axis=0))
    else:
        ax[0].plot(np.mean(training_data['success_rate_after_il'][reward], axis=(0,1)))
        ax[1].plot(np.mean(training_data['success_rate_after_rl'][reward], axis=(0,1)))
if (only_base_and_full_rewards):
    figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    figure.legend(["Reward {}".format(i) for i in range(len(training_data["success_rate_after_il"]))], loc="center right")
figure.savefig(os.path.join(figure_folder,f"success_rate_curves_after_il_and_rl_ablation_study.eps"), format='eps')

## TESTING DATA ##
# Plot boxplot of time to goal, path length, angular_speed, space_compliance after RL for base reward and reward with all contributions
# TODO: Rewrite this plot code, it sucks
for e_idx, test_human_policy in enumerate(test_envs):
    figure, ax = plt.subplots(4,len(test_n_humans),figsize=(10,10))
    figure.suptitle(f"Tested on {test_human_policy}")
    figure.subplots_adjust(hspace=0.7, wspace=0.5, top=0.9, bottom=0.05, left=0.07, right=0.9)
    for i, n_humans in enumerate(test_n_humans):
        ax[0, i].grid()
        ax[0, i].set_title(f"{n_humans} humans")
        ax[0, i].set_xlabel("Scenario")
        ax[0, i].set_ylabel("Time to goal (s)")
        ax[1, i].grid()
        ax[1, i].set_title(f"{n_humans} humans")
        ax[1, i].set_xlabel("Scenario")
        ax[1, i].set_ylabel("Path length (m)")
        ax[2, i].grid()
        ax[2, i].set_title(f"{n_humans} humans")
        ax[2, i].set_xlabel("Scenario")
        ax[2, i].set_ylabel("Angular speed (r/s)")
        ax[3, i].grid()
        ax[3, i].set_title(f"{n_humans} humans")
        ax[3, i].set_xlabel("Scenario")
        ax[3, i].set_ylabel("Space compliance")
        for j, scenario in enumerate(test_scenarios):
            # Base reward
            time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][0,j,e_idx,i])
            time_data = time_data.dropna()
            ax[0, i].boxplot(
                time_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True)
            length_data = pd.DataFrame(all_metrics_after_rl["path_length"][0,j,e_idx,i])
            length_data = length_data.dropna()
            ax[1, i].boxplot(
                length_data, widths=0.4, patch_artist=True,
                positions = [j], 
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True)
            angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][0,j,e_idx,i])
            angular_speed_data = angular_speed_data.dropna()
            ax[2, i].boxplot(
                angular_speed_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True)
            space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][0,j,e_idx,i])
            space_compliance_data = space_compliance_data.dropna()
            ax[3, i].boxplot(
                space_compliance_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True)
            # Full reward
            time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][-1,j,e_idx,i])
            time_data = time_data.dropna()
            ax[0, i].boxplot(
                time_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
                showfliers=False,
                showmeans=True)
            length_data = pd.DataFrame(all_metrics_after_rl["path_length"][-1,j,e_idx,i])
            length_data = length_data.dropna()
            ax[1, i].boxplot(
                length_data, widths=0.4, patch_artist=True,
                positions = [j], 
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
                showfliers=False,
                showmeans=True)
            angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][-1,j,e_idx,i])
            angular_speed_data = angular_speed_data.dropna()
            ax[2, i].boxplot(
                angular_speed_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
                showfliers=False,
                showmeans=True)
            space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][-1,j,e_idx,i])
            space_compliance_data = space_compliance_data.dropna()
            ax[3, i].boxplot(
                space_compliance_data, widths=0.4, patch_artist=True, 
                positions = [j],
                tick_labels = [scenarios_labels[j]],
                boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                whiskerprops=dict(color="coral", alpha=0.4),
                capprops=dict(color="coral", alpha=0.4),
                medianprops=dict(color="coral", alpha=0.4),
                meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
                showfliers=False,
                showmeans=True)
        ## Plot boxplot aggregating all scenarios
        # Base reward
        time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][0,:,e_idx,i].flatten())
        time_data = time_data.dropna()
        ax[0, i].boxplot(
            time_data, widths=0.4, patch_artist=True, 
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        length_data = pd.DataFrame(all_metrics_after_rl["path_length"][0,:,e_idx,i].flatten())
        length_data = length_data.dropna()
        ax[1, i].boxplot(
            length_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
            showfliers=False,
            showmeans=True)
        angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][0,:,e_idx,i].flatten())
        angular_speed_data = angular_speed_data.dropna()
        ax[2, i].boxplot(
            angular_speed_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
            showfliers=False,
            showmeans=True)
        space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][0,:,e_idx,i].flatten())
        space_compliance_data = space_compliance_data.dropna()
        ax[3, i].boxplot(
            space_compliance_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
            showfliers=False,
            showmeans=True)
        # Full reward
        time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][-1,:,e_idx,i].flatten())
        time_data = time_data.dropna()
        ax[0, i].boxplot(
            time_data, widths=0.4, patch_artist=True, 
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
        length_data = pd.DataFrame(all_metrics_after_rl["path_length"][-1,:,e_idx,i].flatten())
        length_data = length_data.dropna()
        ax[1, i].boxplot(
            length_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
            showfliers=False,
            showmeans=True)
        angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][-1,:,e_idx,i].flatten())
        angular_speed_data = angular_speed_data.dropna()
        ax[2, i].boxplot(
            angular_speed_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
            showfliers=False,
            showmeans=True)
        space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][-1,:,e_idx,i].flatten())
        space_compliance_data = space_compliance_data.dropna()
        ax[3, i].boxplot(
            space_compliance_data, widths=0.4, patch_artist=True,
            positions = [len(test_scenarios)],
            tick_labels = ["All"],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
            showfliers=False,
            showmeans=True)
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=4, label="Reward 0"),
        Line2D([0], [0], color="lightcoral", lw=4, label="Reward {}".format(2**len(reward_terms)-1))
    ]
    figure.legend(handles=legend_elements, loc="center right")
    # Save figure
    figure.savefig(os.path.join(figure_folder,f"some_metrics_boxplots_after_rl_full_and_base_all_scenarios_ablation_study_{test_human_policy}.pdf"), format='pdf')


# Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
for e_idx, test_human_policy in enumerate(test_envs):
    figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
    figure.suptitle(f"Metrics after RL training for each test for base and full rewards\nTest scenario {plot_one_test_scenario_only if plot_one_test_scenario_only is not None else 'All'} - Humans policy {test_human_policy.upper()} - {n_test_trials} trials")
    figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=4, label="Reward 0"),
        Line2D([0], [0], color="lightcoral", lw=4, label="Reward {}".format(2**len(reward_terms)-1))
    ]
    figure.legend(handles=legend_elements, loc="center right")
    idx = 0
    for key, values in all_metrics_after_rl.items():
        if key == "successes" or key == "collisions" or key == "timeouts":
            continue
        else:
            i = idx // 3
            j = idx % 3
            ax[i,j].set(
                xlabel='N° humans', 
                ylabel=key)
            ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
            ax[i,j].grid()
            # Base reward
            if plot_one_test_scenario_only is not None:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,test_scen_idx,e_idx,h_idx,:].flatten())
            else:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,:,e_idx,h_idx,:].flatten())
            data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', edgecolor='lightblue', alpha=0.7),
                tick_labels=test_n_humans,
                whiskerprops=dict(color='blue', alpha=0.7),
                capprops=dict(color='blue', alpha=0.7),
                medianprops=dict(color='blue', alpha=0.7),
                meanprops=dict(markerfacecolor='blue', markeredgecolor='blue'), 
                showfliers=False,
                showmeans=True, 
                zorder=1)
            # Full reward
            if plot_one_test_scenario_only is not None:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[-1,test_scen_idx,e_idx,h_idx,:].flatten())
            else:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[-1,:,e_idx,h_idx,:].flatten())
            data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
                    boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                    tick_labels=test_n_humans,
                    whiskerprops=dict(color="coral", alpha=0.4),
                    capprops=dict(color="coral", alpha=0.4),
                    medianprops=dict(color="coral", alpha=0.4),
                    meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
                    showfliers=False,
                    showmeans=True,
                    zorder=2)
            idx += 1
    figure.savefig(os.path.join(figure_folder,f"metrics_boxplots_after_rl_full_and_base_ablation_study_{test_human_policy}.pdf"), format='pdf')

# Plot curves of each metric (aggregated by test scenario) after RL for each reward
for e_idx, test_human_policy in enumerate(test_envs):
    figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
    figure.suptitle(f"Metrics after RL training for each test\nTest scenario {plot_one_test_scenario_only if plot_one_test_scenario_only is not None else 'All'} - Humans policy {test_human_policy.upper()} - {n_test_trials} trials")
    figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
    idx = 0
    for key, values in all_metrics_after_rl.items():
        if key == "successes" or key == "collisions" or key == "timeouts":
            continue
        else:
            i = idx // 3
            j = idx % 3
            ax[i,j].set(
                xlabel='N° humans', 
                ylabel=key)
            ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
            ax[i,j].grid()
            for reward in range(len(values)):
                if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
                    continue
                if plot_one_test_scenario_only is not None:
                    ax[i,j].plot(test_n_humans, np.nanmean(values[reward,test_scen_idx,e_idx], axis=1), color=list(mcolors.TABLEAU_COLORS.values())[reward])
                else:
                    ax[i,j].plot(test_n_humans, np.nanmean(values[reward,:,e_idx], axis=(0,2)), color=list(mcolors.TABLEAU_COLORS.values())[reward])
            idx += 1
    if (only_base_and_full_rewards):
        figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
    else:
        figure.legend(["Reward {}".format(i) for i in range(len(all_metrics_after_rl["times_to_goal"]))], loc="center right")
    figure.savefig(os.path.join(figure_folder,f"metrics_after_rl_ablation_study_{test_human_policy}.eps"), format='eps')

# Plot boxplots side to side for each metric (aggregated by test scenario) after RL for tests with all rewards (one figure for each n_humans test)
for e_idx, test_human_policy in enumerate(test_envs):
    for test, n_humans in enumerate(test_n_humans):
        figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
        figure.suptitle(f"Metrics after RL training for each test for base and full rewards\nHumans policy {test_human_policy} - {n_humans} humans - {n_test_trials} trials")
        figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
        idx = 0
        for key, values in all_metrics_after_rl.items():
            if key == "successes" or key == "collisions" or key == "timeouts":
                continue
            else:
                i = idx // 3
                j = idx % 3
                ax[i,j].set(
                    xlabel='Reward', 
                    ylabel=key)
                ax[i,j].grid()
                # All rewards
                if only_base_and_full_rewards:
                    if plot_one_test_scenario_only is not None:
                        unclean_data = jnp.zeros((2,n_test_trials))
                        for r, reward in enumerate([0, 2**len(reward_terms)-1]):
                            unclean_data = unclean_data.at[r].set(values[reward,test_scen_idx,e_idx,test].flatten())
                    else:
                        unclean_data = jnp.zeros((2,n_test_trials*len(test_scenarios)))
                        for r, reward in enumerate([0, 2**len(reward_terms)-1]):
                            unclean_data = unclean_data.at[r].set(values[reward,:,e_idx,test].flatten())
                    data = pd.DataFrame(np.transpose(unclean_data), columns=["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]])
                else:
                    if plot_one_test_scenario_only is not None:
                        unclean_data = jnp.zeros((len(values),n_test_trials))
                        for reward in range(len(values)):
                            unclean_data = unclean_data.at[reward].set(values[reward,test_scen_idx,e_idx,test].flatten())
                    else:
                        unclean_data = jnp.zeros((len(values),n_test_trials*len(test_scenarios)))
                        for reward in range(len(values)):
                            unclean_data = unclean_data.at[reward].set(values[reward,:,e_idx,test].flatten())
                    data = pd.DataFrame(np.transpose(unclean_data), columns=["Reward {}".format(i) for i in range(len(values))])
                data = data.dropna()
                bplots = ax[i,j].boxplot(
                    data, 
                    widths=0.4, 
                    patch_artist=True,
                    tick_labels=np.arange(len(values)) if not only_base_and_full_rewards else [0, 2**len(reward_terms)-1],
                    showfliers=False,
                    showmeans=True, 
                    zorder=1)
                for patch, color in zip(bplots['boxes'], list(mcolors.TABLEAU_COLORS.values())):
                    patch.set_facecolor(color)
                idx += 1
        legend_elements = []
        for i in range(len(values)):
            if only_base_and_full_rewards and (i > 0 and i < 2**(len(reward_terms))-1):
                continue
            legend_elements.append(Line2D([0], [0], color=list(mcolors.TABLEAU_COLORS.values())[i], lw=4, label=f"Reward {i}"))
        figure.legend(handles=legend_elements, loc="center right")
        figure.savefig(os.path.join(figure_folder,f"metrics_boxplots_after_rl_{n_humans}humans_{test_human_policy}_ablation_study.eps"), format='eps')

# Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
metrics_labels = {
    "times_to_goal": "Time to goal ($s$)",
    "average_speed": "Average speed ($m/s$)",
    "average_acceleration": "Average acceleration ($m/s^2$)",
    "average_jerk": "Average jerk ($m/s^3$)",
    "average_angular_speed": "Average angular speed ($r/s$)",
    "average_angular_acceleration": "Average angular acceleration ($r/s^2$)",
    "average_angular_jerk": "Average angular jerk ($r/s^3$)",
    "space_compliance": "Space compliance",
    "path_length": "Path length ($m$)",
}
for e_idx, test_human_policy in enumerate(test_envs):
    figure, ax = plt.subplots(math.ceil(len(metrics_labels)/3), 3, figsize=(10,10))
    figure.suptitle(f"Metrics after RL training for each test for base and full rewards\nTest scenario {plot_one_test_scenario_only if plot_one_test_scenario_only is not None else 'All'} - Humans policy {test_human_policy.upper()} - {n_test_trials} trials")
    figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=4, label="Reward 0"),
        Line2D([0], [0], color="lightcoral", lw=4, label="Reward {}".format(2**len(reward_terms)-1))
    ]
    figure.legend(handles=legend_elements, loc="center right")
    idx = 0
    for key, values in all_metrics_after_rl.items():
        if key in ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]:
            continue
        else:
            i = idx // 3
            j = idx % 3
            ax[i,j].set(
                xlabel='N° humans',
                title=metrics_labels[key],)
            ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
            ax[i,j].grid()
            # Base reward
            if plot_one_test_scenario_only is not None:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,test_scen_idx,e_idx,h_idx,:].flatten())
            else:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,:,e_idx,h_idx,:].flatten())
            data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', edgecolor='lightblue', alpha=0.7),
                tick_labels=test_n_humans,
                whiskerprops=dict(color='blue', alpha=0.7),
                capprops=dict(color='blue', alpha=0.7),
                medianprops=dict(color='blue', alpha=0.7),
                meanprops=dict(markerfacecolor='blue', markeredgecolor='blue'), 
                showfliers=False,
                showmeans=True, 
                zorder=1)
            # Full reward
            if plot_one_test_scenario_only is not None:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[-1,test_scen_idx,e_idx,h_idx,:].flatten())
            else:
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[-1,:,e_idx,h_idx,:].flatten())
            data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
                    boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                    tick_labels=test_n_humans,
                    whiskerprops=dict(color="coral", alpha=0.4),
                    capprops=dict(color="coral", alpha=0.4),
                    medianprops=dict(color="coral", alpha=0.4),
                    meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
                    showfliers=False,
                    showmeans=True,
                    zorder=2)
            idx += 1
    figure.savefig(os.path.join(figure_folder,f"1_metrics_boxplots_after_rl_full_and_base_ablation_study_{test_human_policy}.pdf"), format='pdf')
