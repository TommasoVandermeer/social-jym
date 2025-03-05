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
test_scenarios = ["circular_crossing","circular_crossing_with_static_obstacles"]
random_seed = 0 
n_static_humans = 3
test_n_humans = [5, 10, 15, 20, 25]
test_n_humans = [i+n_static_humans for i in test_n_humans]
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
vnet_params_dirs = [
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_hs/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_ccso/rl_model.pth"),
]
policy_labels = ["SARL-HS-HSFM","SARL-CCSO-HSFM"]
test_scenarios_labels = ["CC","CCSO"]
empty_trials_outcomes_array = jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(test_n_humans)))
empty_trials_metrics_array = jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(test_n_humans),n_test_trials))
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
#                     'ccso_n_static_humans': n_static_humans,
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
#                 all_metrics = tree_map(lambda x, y: x.at[i,j,k,h].set(y), all_metrics, metrics)

# ### Save results
# with open(os.path.join(os.path.dirname(__file__),f"metrics_tests_with_static_humans.pkl"), 'wb') as f:
#     pickle.dump(all_metrics, f)

### Load results
with open(os.path.join(os.path.dirname(__file__),f"metrics_tests_with_static_humans.pkl"), 'rb') as f:
    all_metrics = pickle.load(f)

### Plot results
# Matplotlib font
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 23}
rc('font', **font)
# Plot curves of each metric (aggregated by test scenario) after RL for each reward
exclude_metrics = ["collisions", "timeouts", "path_length", "returns", "average_angular_speed", "average_angular_acceleration", "average_angular_jerk"]
metrics_data = {
    "successes": {"row_position": 0, "col_position": 0, "label": "Success rate", "ylim": [0.4,1.1], "yticks": [i/10 for i in range(4,11)]}, 
    "times_to_goal": {"row_position": 0, "col_position": 1, "label": "Time to goal ($s$)"},
    "average_speed": {"row_position": 1, "col_position": 0, "label": "Speed ($m/s$)"}, 
    "episodic_spl": {"row_position": 1, "col_position": 1, "label": "SPL", "ylim": [0,1], "yticks": [i/10 for i in range(11)]},
    "space_compliance": {"row_position": 2, "col_position": 0, "label": "Space compliance", "ylim": [0,1]},
    "average_acceleration": {"row_position": 2, "col_position": 1, "label": "Acceleration ($m/s^2$)"},
    "average_jerk": {"row_position": 3, "col_position": 0, "label": "Jerk ($m/s^3$)"},
    "min_distance": {"row_position": 3, "col_position": 1, "label": "Min. dist. to humans ($m$)"},
}

# Plot usual metrics for each (policy-test-scenario) couple against the number of humans
for e_idx, test_env in enumerate(test_environments):
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
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_xticks(jnp.arange(len(test_n_humans)), labels=[i-n_static_humans for i in test_n_humans])
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].grid()
            for policy in range(len(values)):
                if policy_labels[policy] == "SARL-CCSO-HSFM":
                    continue
                for s_idx, test_scenario in enumerate(test_scenarios):
                    ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
                        jnp.arange(len(test_n_humans)), 
                        jnp.nanmean(values[policy,e_idx,s_idx], axis=1) if key != "successes" else values[policy,e_idx,s_idx] / n_test_trials,
                        color=list(mcolors.TABLEAU_COLORS.values())[policy+s_idx*len(values)],
                        linewidth=2,
                        label=f"({policy_labels[policy].split('-')[1]},{test_scenarios_labels[s_idx]})",
                    )
    handles, labels = ax[0,0].get_legend_handles_labels()
    figure.legend(labels, loc="center right", title=f"SARL policies trained\nand tested on {test_env.upper()}.\n(Train, test) scenarios", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
    figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_static_obstacles_tests_on_{test_env}.eps"), format='eps')