from jax.tree_util import tree_map
import jax.numpy as jnp
import optax
import os
import pickle
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import pandas as pd
import math

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.sarl import SARL
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.vnet_rollouts import vnet_rl_rollout, vnet_il_rollout
from socialjym.utils.aux_functions import linear_decay, test_k_trials, save_policy_params, decimal_to_binary
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
train_envs = ['sfm', 'hsfm'] # ['sfm', 'hsfm']
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

############################################################################################  INITIALIZATION

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
            il_out = vnet_il_rollout(**il_rollout_params)
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
                'epsilon_decay_fn': linear_decay,
                'epsilon_start': training_hyperparams['epsilon_start'],
                'epsilon_end': training_hyperparams['epsilon_end'],
                'decay_rate': training_hyperparams['epsilon_decay'],
                'target_update_interval': training_hyperparams['target_update_interval'],
                'custom_episodes': rl_custom_episodes_path,
            }
            # REINFORCEMENT LEARNING ROLLOUT
            rl_out = vnet_rl_rollout(**rl_rollout_params)
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

############################################################################################ PLOTTING

### LOAD ALL DATA
with open(os.path.join(os.path.dirname(__file__),'results', "metrics_after_il_ablation_study.pkl"), "rb") as f:
    all_metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__),'results', "metrics_after_rl_ablation_study.pkl"), "rb") as f:
    all_metrics_after_rl = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__),'results', "training_data_ablation_study.pkl"), "rb") as f:
    training_data = pickle.load(f)

### INITIALIZATION
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "timeouts": {"label": "Timeout Rate", "episodic": False}, 
    "returns": {"label": "Discounted return ($\gamma = 0.9$)", "episodic": True},
    "times_to_goal": {"label": "Time to goal ($s$)", "episodic": True},
    "average_speed": {"label": "Lin. speed ($m/s$)", "episodic": True},
    "average_acceleration": {"label": "Lin. accel. ($m/s^2$)", "episodic": True},
    "average_jerk": {"label": "Lin. jerk ($m/s^3$)", "episodic": True},
    "average_angular_speed": {"label": "Ang. speed ($rad/s$)", "episodic": True},
    "average_angular_acceleration": {"label": "Ang. accel. ($rad/s^2$)", "episodic": True},
    "average_angular_jerk": {"label": "Ang. jerk ($rad/s^3$)", "episodic": True},
    "min_distance": {"label": "Minimum distance to humans ($m$)", "episodic": True},
    "space_compliance": {"label": "Space compliance", "episodic": True},
    "episodic_spl": {"label": "Episodic SPL", "episodic": True},
    "path_length": {"label": "Path length ($m$)", "episodic": True},
}
scenarios = {
    "circular_crossing": {"label": "CC"},
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "robot_crowding": {"label": "RC"},
    "delayed_circular_crossing": {"label": "DCC"},
    "hybrid_scenario": {"label": "HS"},
}
envs = {
    "sfm": {"label": "SFM"},
    "hsfm": {"label": "HSFM"},
}
rewards = {
    0: {"label": "R0"},
    1: {"label": "R1"},
    2: {"label": "R2"},
    3: {"label": "R3"},
    4: {"label": "R4"},
    5: {"label": "R5"},
    6: {"label": "R6"},
    7: {"label": "R7"},
}

### PLOTS
# Create figure folder
if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"))
figure_folder = os.path.join(os.path.dirname(__file__), "figures")

# Matplotlib font
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 15}
rc('font', **font)

# Plot action space
policy = SARL(Reward2(), dt=0.25, kinematics="unicycle")
figure, ax = plt.subplots(figsize=(10,5))
figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
# ax.scatter(policy.action_space[:,0], policy.action_space[:,1], zorder=3, label="Sampled actions", color='blue', s=60)
actions_space_bound = Polygon(
    jnp.array([[policy.v_max,0.],[0.,policy.v_max*2/policy.wheels_distance],[0.,-policy.v_max*2/policy.wheels_distance]]), 
    closed=True, 
    fill=True, 
    edgecolor='black',
    facecolor='lightgrey',
    linewidth=2,
    zorder=3,
    label="Feasible action space"
)
ax.grid(zorder=1)
ax.add_patch(actions_space_bound)
xoffset=0.1
yoffset = 0.3
ax.set(xlim=[0-xoffset,policy.v_max+xoffset], ylim=[-policy.v_max*2/policy.wheels_distance-yoffset,policy.v_max*2/policy.wheels_distance+yoffset])
ax.set_xlabel("$v_r$ $(m/s)$")
ax.set_ylabel("$\omega_r$ $(rad/s)$")
ax.legend()
figure.savefig(os.path.join(figure_folder,f"0.pdf"), format='pdf')

# Plot barplot of TtG and Ang.Accel. after RL for base reward trained and tested in different environments (HSFM and SFM)
metrics_to_plot = ["times_to_goal", "average_angular_acceleration"]
colors = ["lightskyblue", "blue", "lightcoral", "red"]
figure, ax = plt.subplots(1, 2, figsize=(10,5))
figure.subplots_adjust(hspace=0.7, wspace=0.2, bottom=0.13, top=0.91, left=0.05, right=0.77)
for m_idx, metric in enumerate(metrics_to_plot):        
    bar_width = 0.2
    p0 = jnp.arange(len(test_n_humans))
    ax[m_idx].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[m_idx].set_xticks(p0 + (3 / 2) * bar_width, labels=test_n_humans)
    idx = 0
    for te_idx, train_env in enumerate(train_envs):
        for e_idx, test_env in enumerate(test_envs):
            p = p0 + idx * bar_width
            bars = jnp.nanmean(all_metrics_after_rl[metric][0,te_idx,0,0,e_idx, :, :], axis=1)
            color = colors[int(2 * te_idx + e_idx)]
            ax[m_idx].bar(
                p, 
                bars, 
                width=bar_width, 
                color=color, 
                label=f"{envs[train_env]['label']}-{envs[test_env]['label']}", 
                edgecolor='white', 
                zorder=3
            )
            idx += 1
    if m_idx == 0:
        figure.legend(loc="center right", title="Train-test\nenvironments", fontsize=15)
    ax[m_idx].grid(zorder=0)
figure.savefig(os.path.join(figure_folder,f"1.pdf"), format='pdf')

# Plot boxplot of each metric for base reward  in all train envs
metrics_to_plot = ["times_to_goal", "path_length", "space_compliance", "average_speed", "average_acceleration", "average_jerk", "average_angular_speed", "average_angular_acceleration", "average_angular_jerk"]
figure, ax = plt.subplots(3, 3, figsize=(10,10))
figure.subplots_adjust(hspace=0.7, wspace=0.3, bottom=0.08, top=0.93, left=0.07, right=0.83)
colors = {
    "sfm": "blue",
    "hsfm": "red",
}
legend_elements = [
    Line2D([0], [0], color=colors[train_envs[0]], lw=4, label=train_envs[0].upper()),
    Line2D([0], [0], color=colors[train_envs[1]], lw=4, label=train_envs[1].upper())
]
figure.legend(handles=legend_elements, loc="center right", title="Training\nenv.")
for m_idx, metric in enumerate(metrics_to_plot):
        values = all_metrics_after_rl[metric]
        i = m_idx // 3
        j = m_idx % 3
        ax[i,j].set(
            xlabel='N° humans',
            title=metrics[metric]['label'],)
        p0 = jnp.arange(len(test_n_humans))
        box_width = 0.3
        ax[i,j].grid(zorder=0)
        # Train env
        for te_idx, train_env in enumerate(train_envs):
            unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)*len(test_envs)))
            for h_idx in range(len(test_n_humans)):
                unclean_data = unclean_data.at[h_idx].set(values[0,te_idx,0,0,:,h_idx,:].flatten())
            data = pd.DataFrame(jnp.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(
                data, 
                widths=box_width,
                positions=p0 + te_idx * box_width, 
                patch_artist=True, 
                boxprops=dict(facecolor=colors[train_env], edgecolor=colors[train_env]),
                meanprops=dict(markerfacecolor="white", markeredgecolor="white"), 
                medianprops=dict(color='black'),
                showfliers=False,
                showmeans=True, 
                zorder=3,
            )
        ax[i,j].set_xticks(p0 + box_width * 0.5, labels=test_n_humans)
figure.savefig(os.path.join(figure_folder,f"2.pdf"), format='pdf')

# Plot barplot of TtG and Ang.Accel. after RL for base reward trained in different environments (HSFM and SFM) with different reward (R0 and R7)
metrics_to_plot = ["times_to_goal", "average_angular_acceleration"]
colors = ["blue", "green", "red", "grey"]
figure, ax = plt.subplots(1, 2, figsize=(10,5))
figure.subplots_adjust(hspace=0.7, wspace=0.2, bottom=0.13, top=0.91, left=0.05, right=0.77)
for m_idx, metric in enumerate(metrics_to_plot):        
    bar_width = 0.2
    p0 = jnp.arange(len(test_n_humans))
    ax[m_idx].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[m_idx].set_xticks(p0 + (3 / 2) * bar_width, labels=test_n_humans)
    idx = 0
    for te_idx, train_env in enumerate(train_envs):
        for r_idx, reward in enumerate([0,7]):
            p = p0 + idx * bar_width
            bars = jnp.nanmean(all_metrics_after_rl[metric][0,te_idx,reward,0,:, :, :], axis=(0,2))
            color = colors[int(2 * te_idx + r_idx)]
            ax[m_idx].bar(
                p, 
                bars, 
                width=bar_width, 
                color=color, 
                label=f"{envs[train_env]['label']}-{rewards[reward]['label']}", 
                edgecolor='white', 
                zorder=3
            )
            idx += 1
    if m_idx == 0:
        figure.legend(loc="center right", title="Train env.\n- Reward", fontsize=15)
    ax[m_idx].grid(zorder=0)
figure.savefig(os.path.join(figure_folder,f"3.pdf"), format='pdf')

# Plot boxplot of each metric for base and full rewards  in all train envs
metrics_to_plot = ["times_to_goal", "path_length", "space_compliance", "average_speed", "average_acceleration", "average_jerk", "average_angular_speed", "average_angular_acceleration", "average_angular_jerk"]
colors = {
        "sfm": {
            "R0": "blue",
            "R7": "green",
        },
        "hsfm": {
            "R0": "red",
            "R7": "grey",
        },
    }
for te_idx, train_env in enumerate(train_envs):
    figure, ax = plt.subplots(3, 3, figsize=(10,10))
    figure.subplots_adjust(hspace=0.7, wspace=0.3, bottom=0.08, top=0.93, left=0.07, right=0.83)
    legend_elements = [
        Line2D([0], [0], color=colors[train_env]["R0"], lw=4, label="R0"),
        Line2D([0], [0], color=colors[train_env]["R7"], lw=4, label="R7")
    ]
    figure.legend(handles=legend_elements, loc="center right", title=f"Train env.\n{train_env.upper()}\n\nReward")
    for m_idx, metric in enumerate(metrics_to_plot):
            values = all_metrics_after_rl[metric]
            i = m_idx // 3
            j = m_idx % 3
            ax[i,j].set(
                xlabel='N° humans',
                title=metrics[metric]['label'],)
            p0 = jnp.arange(len(test_n_humans))
            box_width = 0.3
            ax[i,j].grid(zorder=0)
            # Train env
            for r_idx, reward in enumerate([0,7]):
                unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)*len(test_envs)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,te_idx,reward,0,:,h_idx,:].flatten())
                data = pd.DataFrame(jnp.transpose(unclean_data), columns=test_n_humans)
                data = data.dropna()
                ax[i,j].boxplot(
                    data, 
                    widths=box_width,
                    positions=p0 + r_idx * box_width, 
                    patch_artist=True, 
                    boxprops=dict(facecolor=colors[train_env]["R"+str(reward)], edgecolor=colors[train_env]["R"+str(reward)]),
                    meanprops=dict(markerfacecolor="white", markeredgecolor="white"), 
                    medianprops=dict(color='black'),
                    showfliers=False,
                    showmeans=True, 
                    zorder=3,
                )
            ax[i,j].set_xticks(p0 + box_width * 0.5, labels=test_n_humans)
    add = "_not_included" if not(bool(te_idx)) else ""
    figure.savefig(os.path.join(figure_folder,f"{4}{add}.pdf"), format='pdf')

# Plot boxplot of each metric for all rewards in all train envs
colors = [list(mcolors.TABLEAU_COLORS.values())[:len(rewards)] for _ in range(len(train_envs))]
colors[0][0] = "blue"
colors[0][7] = "green"
colors[1][0] = "red"
colors[1][7] = "grey"
metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
for te_idx, train_env in enumerate(train_envs):
    legend_elements = [Line2D([0], [0], color=colors[te_idx][r], lw=4, label=rewards[r]["label"]) for r in range(len(rewards))]
    figure, ax = plt.subplots(2, 2, figsize=(10,10))
    figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.93, left=0.07, right=0.95)
    # figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.93, left=0.07, right=0.87)
    # figure.legend(handles=legend_elements, loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nN° Hum.\n{test_n_humans[1]}\n\nReward", fontsize=15)
    for m_idx, metric in enumerate(metrics_to_plot):
        values = all_metrics_after_rl[metric]
        i = m_idx // 2
        j = m_idx % 2
        ax[i,j].set(
            xlabel='Reward function',
            title=metrics[metric]['label'],)
        box_width = 0.4
        ax[i,j].grid(zorder=0)
        for r_idx in range(len(rewards)):
            unclean_data = values[0,te_idx,r_idx,0,:,1,:].flatten()
            data = pd.DataFrame(jnp.transpose(unclean_data), columns=[r_idx])
            data = data.dropna()
            ax[i,j].boxplot(
                data, 
                widths=box_width,
                positions=[r_idx],
                boxprops=dict(facecolor=colors[te_idx][r_idx], edgecolor=colors[te_idx][r_idx]),
                meanprops=dict(markerfacecolor="white", markeredgecolor="white", markersize=5), 
                medianprops=dict(color='black'),
                patch_artist=True, 
                showfliers=False,
                showmeans=True, 
                zorder=3,
            )
        ax[i,j].set_xticks(jnp.arange(len(rewards)), labels=[rewards[r]["label"] for r in range(len(rewards))])
    add = "_not_included" if not(bool(te_idx)) else ""
    figure.savefig(os.path.join(figure_folder,f"{5}{add}.pdf"), format='pdf')


# # Plot boxplot of each metric for all rewards in all train envs
# colors = list(mcolors.TABLEAU_COLORS.values())[:len(rewards)]
# metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
# legend_elements = [Line2D([0], [0], color=colors[r], lw=4, label=rewards[r]["label"]) for r in range(len(rewards))]
# for te_idx, train_env in enumerate(train_envs):
#     figure, ax = plt.subplots(2, 2, figsize=(10,10))
#     figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.93, left=0.06, right=0.87)
#     figure.legend(handles=legend_elements, loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nReward", fontsize=15)
#     for m_idx, metric in enumerate(metrics_to_plot):
#         values = all_metrics_after_rl[metric]
#         i = m_idx // 2
#         j = m_idx % 2
#         ax[i,j].set(
#             xlabel='N° humans',
#             title=metrics[metric]['label'],)
#         box_width = 0.22
#         ax[i,j].grid(zorder=0)
#         for h_idx in range(len(test_n_humans)):
#             for r_idx in range(len(rewards)):
#                 unclean_data = values[0,te_idx,r_idx,0,:,h_idx,:].flatten()
#                 data = pd.DataFrame(jnp.transpose(unclean_data), columns=[r_idx])
#                 data = data.dropna()
#                 ax[i,j].boxplot(
#                     data, 
#                     widths=box_width,
#                     positions=[2 * h_idx + r_idx * box_width],
#                     boxprops=dict(facecolor=colors[r_idx], edgecolor=colors[r_idx]),
#                     meanprops=dict(markerfacecolor="white", markeredgecolor="white", markersize=5), 
#                     medianprops=dict(color='black', linewidth=0),
#                     patch_artist=True, 
#                     showfliers=False,
#                     showmeans=True, 
#                     whis=0,
#                     showcaps=False,
#                     zorder=3,
#                 )
#         ax[i,j].set_xticks(2 * jnp.arange(len(test_n_humans)) + (7 / 2) * box_width, labels=test_n_humans)
#     figure.savefig(os.path.join(figure_folder,f"{6 + te_idx}.pdf"), format='pdf')

# # Plot barplots of each metric for all rewards in all train envs
# colors = list(mcolors.TABLEAU_COLORS.values())[:len(rewards)]
# metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
# for te_idx, train_env in enumerate(train_envs):
#     figure, ax = plt.subplots(2, 2, figsize=(10,10))
#     figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.95, left=0.06, right=0.87)
#     for m_idx, metric in enumerate(metrics_to_plot):
#         values = all_metrics_after_rl[metric]
#         i = m_idx // 2
#         j = m_idx % 2
#         ax[i,j].set(
#             xlabel='N° humans',
#             title=metrics[metric]['label'],)
#         box_width = 0.22
#         ax[i,j].grid(zorder=0)
#         for h_idx in range(len(test_n_humans)):
#             for r_idx in range(len(rewards)):
#                 bar = jnp.nanmean(values[0,te_idx,r_idx,0,:,h_idx,:], axis=(0,1))
#                 ax[i,j].bar(
#                     2 * h_idx + r_idx * box_width, 
#                     bar,
#                     width=box_width,
#                     color=colors[r_idx], 
#                     label=f"R{r_idx}", 
#                     edgecolor='white', 
#                     zorder=3,
#                 )
#             if (m_idx == 0) and (h_idx == 0):
#                 figure.legend(loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nReward", fontsize=15)
#         ax[i,j].set_xticks(2 * jnp.arange(len(test_n_humans)) + (7 / 2) * box_width, labels=test_n_humans)
#     figure.savefig(os.path.join(figure_folder,f"barplots_rewards_benchmark_{train_env}.pdf"), format='pdf')

# # Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
# exclude_metrics = ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]
# for r_idx in range(2**len(reward_terms)):
#     for e_idx, test_env in enumerate(test_envs):
#         figure, ax = plt.subplots(math.ceil((len(metrics)-len(exclude_metrics))/3), 3, figsize=(10,10))
#         figure.subplots_adjust(hspace=0.7, wspace=0.5, bottom=0.08, top=0.93, left=0.07, right=0.83)
#         figure.suptitle(f"Metrics after RL - Test env: {envs[test_env]['label']} - Reward:{rewards[r_idx]['label']}", fontsize=10)
#         legend_elements = [
#             Line2D([0], [0], color="lightblue", lw=4, label=train_envs[0].upper()),
#             Line2D([0], [0], color="lightcoral", lw=4, label=train_envs[1].upper())
#         ]
#         figure.legend(handles=legend_elements, loc="center right", title="Train.\nEnv.")
#         idx = 0
#         for key, values in all_metrics_after_rl.items():
#             if key in exclude_metrics:
#                 continue
#             else:
#                 i = idx // 3
#                 j = idx % 3
#                 ax[i,j].set(
#                     xlabel='N° humans',
#                     title=metrics[key]['label'],)
#                 ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
#                 ax[i,j].grid()
#                 # First train env
#                 unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
#                 for h_idx in range(len(test_n_humans)):
#                     unclean_data = unclean_data.at[h_idx].set(values[0,0,r_idx,0,e_idx,h_idx,:].flatten())
#                 data = pd.DataFrame(jnp.transpose(unclean_data), columns=test_n_humans)
#                 data = data.dropna()
#                 ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
#                     boxprops=dict(facecolor='lightblue', edgecolor='lightblue', alpha=0.7),
#                     tick_labels=test_n_humans,
#                     whiskerprops=dict(color='blue', alpha=0.7),
#                     capprops=dict(color='blue', alpha=0.7),
#                     medianprops=dict(color='blue', alpha=0.7),
#                     meanprops=dict(markerfacecolor='blue', markeredgecolor='blue'), 
#                     showfliers=False,
#                     showmeans=True, 
#                     zorder=1)
#                 # Second train env
#                 unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
#                 for h_idx in range(len(test_n_humans)):
#                     unclean_data = unclean_data.at[h_idx].set(values[0,1,r_idx,0,e_idx,h_idx,:].flatten())
#                 data = pd.DataFrame(jnp.transpose(unclean_data), columns=test_n_humans)
#                 data = data.dropna()
#                 ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
#                         boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
#                         tick_labels=test_n_humans,
#                         whiskerprops=dict(color="coral", alpha=0.4),
#                         capprops=dict(color="coral", alpha=0.4),
#                         medianprops=dict(color="coral", alpha=0.4),
#                         meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
#                         showfliers=False,
#                         showmeans=True,
#                     zorder=2)
#             idx += 1
#         figure.savefig(os.path.join(figure_folder,f"metrics_boxplots_after_rl_train_env_benchmark_{test_env}_R{r_idx}.pdf"), format='pdf')