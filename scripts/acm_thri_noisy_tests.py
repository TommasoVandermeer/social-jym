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
from socialjym.utils.aux_functions import load_crowdnav_policy, test_k_trials, initialize_metrics_dict

### Hyperparameters
noise_sigma_percentage_levels = [0., 0.1, 0.2, 0.3]
test_environments = ["sfm", "hsfm"]
test_scenarios = ["circular_crossing", "parallel_traffic"]
random_seed = 0 
n_humans = 15
n_test_trials = 500
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
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_hs/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_hs/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_hs/rl_model.pth"),
]
policy_labels = ["SARL-HS-ORCA","SARL-HS-SFM","SARL-HS-HSFM"]
metrics_dims = (len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels))
all_metrics = initialize_metrics_dict(n_test_trials, metrics_dims)

### Test loop
for i, vnet_params_dir in enumerate(vnet_params_dirs):
    for j, test_env in enumerate(test_environments):
        for k, test_scenario in enumerate(test_scenarios):
            for w, noise_sigma_percentage in enumerate(noise_sigma_percentage_levels):
                print(f"\nTesting {vnet_params_dir.split('/')[-2]} on {test_env} in {test_scenario} with noise {noise_sigma_percentage*100}%")
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
                }
                env = SocialNav(**env_params)
                ### Initialize robot policy
                policy = SARL(
                    env.reward_function, 
                    dt = env_params['robot_dt'], 
                    kinematics = kinematics, 
                    noise = True, 
                    noise_sigma_percentage = noise_sigma_percentage)
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
                all_metrics = tree_map(lambda x, y: x.at[i,j,k,w].set(y), all_metrics, metrics)

### Save results
with open(os.path.join(os.path.dirname(__file__),"metrics_noisy_tests.pkl"), 'wb') as f:
    pickle.dump(all_metrics, f)

### Load results
with open(os.path.join(os.path.dirname(__file__),"metrics_noisy_tests.pkl"), 'rb') as f:
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
for e_idx, test_env in enumerate(test_environments):
    figure, ax = plt.subplots(math.ceil((len(all_metrics)-len(exclude_metrics))/2), 2, figsize=(18,18))
    figure.subplots_adjust(right=0.78, top=0.985, bottom=0.05, left=0.07, hspace=0.3, wspace=0.3)
    for key, values in all_metrics.items():
        if key in exclude_metrics:
            continue
        else:
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set(
                xlabel='Noise $\sigma$ (%)',
                ylabel=metrics_data[key]["label"])
            if "ylim" in metrics_data[key]:
                ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_ylim(metrics_data[key]["ylim"])
            if "yticks" in metrics_data[key]:
                ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_yticks(metrics_data[key]["yticks"])
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_xticks(noise_sigma_percentage_levels, labels=[n*100 for n in noise_sigma_percentage_levels])
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].grid()
            for policy in range(len(values)):
                ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
                    noise_sigma_percentage_levels, 
                    jnp.nanmean(values[policy,e_idx], axis=(0,2)) if key != "successes" else jnp.nanmean(values[policy,e_idx], axis=(0)) / n_test_trials,
                    color=list(mcolors.TABLEAU_COLORS.values())[policy+3],
                    linewidth=2,)
    figure.legend(policy_labels, loc="center right", title=f"Policy tested\non {test_env.upper()}")
    figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_after_rl_noisy_tests_on_{test_env}.eps"), format='eps')