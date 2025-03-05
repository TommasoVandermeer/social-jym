from jax.tree_util import tree_map
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import math

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import load_crowdnav_policy, test_k_trials

### Hyperparameters
test_environments = ["hsfm"] #["sfm", "hsfm"]
test_scenarios = ["perpendicular_traffic","robot_crowding","crowd_navigation"]
random_seed = 0 
test_n_humans = [5, 10, 15, 20, 25]
n_test_trials = 100
kinematics = 'holonomic'
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
reward_function = Reward1(**reward_params)
train_envs = ["ORCA", "SFM", "HSFM"]
train_scenarios = ["CC", "PT", "HS"]
vnet_params_dirs = [
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_cc/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_cc/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_cc/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_pat/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_pat/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_pat/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_hs/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_hs/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_hs/rl_model.pth"),
]
policy_labels = [
    "SARL-CC-ORCA",
    "SARL-CC-SFM",
    "SARL-CC-HSFM",
    "SARL-PT-ORCA",
    "SARL-PT-SFM",
    "SARL-PT-HSFM",
    "SARL-HS-ORCA",
    "SARL-HS-SFM",
    "SARL-HS-HSFM",
]

# Initialize metrics
empty_trials_outcomes_array = jnp.zeros((len(train_envs),len(train_scenarios),len(test_environments),len(test_scenarios),len(test_n_humans)))
empty_trials_metrics_array = jnp.zeros((len(train_envs),len(train_scenarios),len(test_environments),len(test_scenarios),len(test_n_humans),n_test_trials))
all_metrics = {
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

# ### Test loop
# for i, vnet_params_dir in enumerate(vnet_params_dirs):
#     train_env_idx = train_envs.index(policy_labels[i].split("-")[2])
#     train_scenario_idx = train_scenarios.index(policy_labels[i].split("-")[1])
#     for j, test_env in enumerate(test_environments):
#         for k, test_scenario in enumerate(test_scenarios):
#             for h, n_humans in enumerate(test_n_humans):
#                 print(f"\nTesting {vnet_params_dir.split('/')[-2]} on {test_env} in {test_scenario}")
#                 ### Initialize and reset environment
#                 env_params = {
#                     'robot_radius': 0.3,
#                     'n_humans': n_humans,
#                     'robot_dt': 0.25,
#                     'humans_dt': 0.01,
#                     'robot_visible': True,
#                     'scenario': test_scenario,
#                     'humans_policy': test_env,
#                     'reward_function': reward_function,
#                     'kinematics': kinematics,
#                 }
#                 env = SocialNav(**env_params)
#                 ### Initialize robot policy
#                 policy = SARL(
#                     env.reward_function, 
#                     dt = env_params['robot_dt'], 
#                     kinematics = kinematics, 
#                     noise = False)
#                 vnet_params = load_crowdnav_policy(
#                     "sarl",
#                     vnet_params_dir)
#                 ### Execute test
#                 metrics = test_k_trials(
#                     n_test_trials,
#                     random_seed,
#                     env,
#                     policy,
#                     vnet_params,
#                     reward_function.time_limit)
#                 ### Save results
#                 all_metrics = tree_map(lambda x, y: x.at[train_env_idx,train_scenario_idx,j,k,h].set(y), all_metrics, metrics)

# ### Save results
# with open(os.path.join(os.path.dirname(__file__),f"metrics_tests_new_evaluation_scenarios.pkl"), 'wb') as f:
#     pickle.dump(all_metrics, f)

### Load results
with open(os.path.join(os.path.dirname(__file__),f"metrics_tests_new_evaluation_scenarios.pkl"), 'rb') as f:
    all_metrics = pickle.load(f)

### Plot results
# Matplotlib font
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 23}
rc('font', **font)
# Auxiliary plot data
scenarios_data = {
    "perpendicular_traffic": {"label": "PeT"},
    "robot_crowding": {"label": "RC"},
    "crowd_navigation": {"label": "CN"},
    "circular_crossing": {"label": "CC"},
    "parallel_traffic": {"label": "PT"},
    "hybrid_scenario": {"label": "HS"},
}
exclude_metrics = ["collisions", "timeouts", "average_speed", "returns", "average_angular_speed", "average_angular_acceleration", "average_angular_jerk"]
metrics_data = {
    "successes": {"row_position": 0, "col_position": 0, "label": "Success rate", "ylim": [0.4,1.1], "yticks": [i/10 for i in range(4,11)]}, 
    "times_to_goal": {"row_position": 0, "col_position": 1, "label": "Time to goal ($s$)"},
    "path_length": {"row_position": 1, "col_position": 0, "label": "Path length ($m$)"}, 
    "episodic_spl": {"row_position": 1, "col_position": 1, "label": "SPL", "ylim": [0,1], "yticks": [i/10 for i in range(11)]},
    "space_compliance": {"row_position": 2, "col_position": 0, "label": "Space compliance", "ylim": [0,1]},
    "average_acceleration": {"row_position": 2, "col_position": 1, "label": "Acceleration ($m/s^2$)"},
    "average_jerk": {"row_position": 3, "col_position": 0, "label": "Jerk ($m/s^3$)"},
    "min_distance": {"row_position": 3, "col_position": 1, "label": "Min. dist. to humans ($m$)"},
}

# Plot metrics curves for each training scenario, averaged over everything else
figure, ax = plt.subplots(math.ceil((len(all_metrics)-len(exclude_metrics))/2), 2, figsize=(18,18))
figure.subplots_adjust(right=0.78, top=0.985, bottom=0.05, left=0.07, hspace=0.3, wspace=0.3)
for key, values in all_metrics.items():
    if key in exclude_metrics:
        continue
    else:
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set(
            xlabel='Number of humans',
            ylabel=metrics_data[key]["label"])
        if "ylim" in metrics_data[key]:
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_ylim(metrics_data[key]["ylim"])
        if "yticks" in metrics_data[key]:
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_yticks(metrics_data[key]["yticks"])
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_xticks(jnp.arange(len(test_n_humans)), labels=test_n_humans)
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].grid()
        for ts_idx, train_scenario in enumerate(train_scenarios):
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
                jnp.arange(len(test_n_humans)), 
                jnp.nanmean(values[:,ts_idx,:,:], axis=(0,1,2,4)) if key != "successes" else jnp.nanmean(values[:,ts_idx,:,:], axis=(0,1,2)) / n_test_trials,
                color=list(mcolors.TABLEAU_COLORS.values())[ts_idx+6],
                linewidth=2,
            )
handles, labels = ax[0,0].get_legend_handles_labels()
figure.legend(train_scenarios, loc="center right", title=f"SARL policies tested\nin new scenarios.\nTrain scenario:", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_tests_in_new_scenarios.eps"), format='eps')

# Plot success rate and time to goal for each training scenario, averaged over everything else
figure, ax = plt.subplots(1, 2, figsize=(18,6))
figure.subplots_adjust(right=0.78, top=0.985, bottom=0.13, left=0.07, hspace=0.3, wspace=0.3)
col = 0
for key, values in all_metrics.items():
    if key not in ["successes", "times_to_goal"]:
        continue
    else:
        ax[col].set(
            xlabel='Number of humans',
            ylabel=metrics_data[key]["label"])
        if "ylim" in metrics_data[key]:
            ax[col].set_ylim(metrics_data[key]["ylim"])
        if "yticks" in metrics_data[key]:
            ax[col].set_yticks(metrics_data[key]["yticks"])
        ax[col].set_xticks(jnp.arange(len(test_n_humans)), labels=test_n_humans)
        ax[col].grid()
        for ts_idx, train_scenario in enumerate(train_scenarios):
            ax[col].plot(
                jnp.arange(len(test_n_humans)), 
                jnp.nanmean(values[:,ts_idx,:,:], axis=(0,1,2,4)) if key != "successes" else jnp.nanmean(values[:,ts_idx,:,:], axis=(0,1,2)) / n_test_trials,
                color=list(mcolors.TABLEAU_COLORS.values())[ts_idx+6],
                linewidth=2,
            )
        col += 1
handles, labels = ax[0].get_legend_handles_labels()
figure.legend(train_scenarios, loc="center right", title=f"SARL policies tested\nin new scenarios.\nTrain scenario:", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
figure.savefig(os.path.join(os.path.dirname(__file__),f"sr_ttg_tests_in_new_scenarios.eps"), format='eps')