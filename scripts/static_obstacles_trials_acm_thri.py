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
test_scenarios = ["circular_crossing_with_static_obstacles"]
random_seed = 0 
n_static_humans = 3
test_n_humans = [5, 10, 15]
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
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_ccso/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_ccso/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_ccso/rl_model.pth"),
]
policy_labels = ["SARL-CCSO-ORCA","SARL-CCSO-SFM","SARL-CCSO-HSFM"]
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

### Test loop
for i, vnet_params_dir in enumerate(vnet_params_dirs):
    for j, test_env in enumerate(test_environments):
        for k, test_scenario in enumerate(test_scenarios):
            for h, n_humans in enumerate(test_n_humans):
                print(f"\nTesting {vnet_params_dir.split('/')[-2]} on {test_env} in {test_scenario}")
                ### Initialize and reset environment
                env_params = {
                    'robot_radius': 0.3,
                    'n_humans': n_humans,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,
                    'robot_visible': True,
                    'scenario': test_scenario,
                    'humans_policy': test_env,
                    'reward_function': reward_function,
                    'kinematics': kinematics,
                    'ccso_n_static_humans': n_static_humans,
                }
                env = SocialNav(**env_params)
                ### Initialize robot policy
                policy = SARL(
                    env.reward_function, 
                    dt = env_params['robot_dt'], 
                    kinematics = kinematics, 
                    noise = False)
                vnet_params = load_crowdnav_policy(
                    "sarl",
                    vnet_params_dir)
                ### Execute test
                metrics = test_k_trials(
                    n_test_trials,
                    random_seed,
                    env,
                    policy,
                    vnet_params,
                    reward_function.time_limit)
                ### Save results
                all_metrics = tree_map(lambda x, y: x.at[i,j,k,h].set(y), all_metrics, metrics)

### Save results
with open(os.path.join(os.path.dirname(__file__),f"metrics_tests_with_static_humans.pkl"), 'wb') as f:
    pickle.dump(all_metrics, f)

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

# Plot success rate, collision rate and timout rate for each test scenario
for e_idx, test_env in enumerate(test_environments):
    for h_idx, n_humans in enumerate(test_n_humans):
        figure, ax = plt.subplots(1,1,figsize=(10,10))
        figure.subplots_adjust(hspace=0.5, bottom=0.1, top=0.87, right=0.8)
        ax.set(
            xlabel='Training environment', 
            ylabel='Success rate', 
            xticks=jnp.arange(len(policy_labels)), 
            xticklabels=[i.split("-")[-1] for i in policy_labels],
            yticks=[i for i in range(0,110,10)]
        )
        ax.set_title(f'Outcomes of trials in CCSO ({n_humans} humans)\n SARL-HS policies - {n_test_trials} trials - Test env {test_env}', pad=30)
        ax.grid(zorder=0)
        outcomes = ["successes","collisions","timeouts"]
        outcome_colors = ["green","red","yellow"]
        bottoms = jnp.zeros((len(policy_labels),))
        for m_idx, metric in enumerate(outcomes):
            ax.bar(
                jnp.arange(len(policy_labels)),
                all_metrics[metric][:,e_idx,0,h_idx],
                color = outcome_colors[m_idx],
                edgecolor = "white",
                width = 0.5,
                bottom = bottoms,
                zorder=3
            )
            bottoms = bottoms.at[:].set(bottoms + all_metrics[metric][:,e_idx,0,h_idx])
        ax.legend(outcomes, loc='center right', title="Outcome", bbox_to_anchor=(1.3, 0.5), fontsize=15, title_fontsize=15)
        figure.savefig(os.path.join(os.path.dirname(__file__),f"outcomes_trials_with_static_obstacles_{test_env}_{n_humans}.eps"), format='eps')

# Plot usual metrics for each policy against the number of humans
for e_idx, test_env in enumerate(test_environments):
    figure, ax = plt.subplots(math.ceil((len(all_metrics)-len(exclude_metrics))/2), 2, figsize=(18,18))
    figure.subplots_adjust(right=0.75, top=0.985, bottom=0.05, left=0.07, hspace=0.3, wspace=0.3)
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
                ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
                    jnp.arange(len(test_n_humans)), 
                    jnp.nanmean(values[policy,e_idx], axis=(0,2)) if key != "successes" else jnp.nanmean(values[policy,e_idx], axis=(0)) / n_test_trials,
                    color=list(mcolors.TABLEAU_COLORS.values())[policy+3],
                    linewidth=2,)
    figure.legend(policy_labels, loc="center right", title=f"Policy tested\non {test_env.upper()}")
    figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_after_rl_noisy_tests_on_{test_env}.eps"), format='eps')