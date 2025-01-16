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
noise_sigma_percentage_levels = [0., 0.15, 0.2, 0.3]
test_environments = ["sfm", "hsfm"]
test_scenarios = ["circular_crossing", "parallel_traffic"]
random_seed = 0 
n_humans = 15
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
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_orca_hybrid_scenario/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_sfm_hybrid_scenario/rl_model.pth"),
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_hybrid_scenario/rl_model.pth"),
]
policy_labels = ["SARL-HS-ORCA","SARL-HS-SFM","SARL-HS-HSFM"]

all_metrics = {
    "successes": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels))), 
    "collisions": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels))), 
    "timeouts": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels))), 
    "returns": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "times_to_goal": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_speed": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_acceleration": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_jerk": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_angular_speed": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_angular_acceleration": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "average_angular_jerk": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "min_distance": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "space_compliance": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "episodic_spl": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials)),
    "path_length": jnp.zeros((len(vnet_params_dirs),len(test_environments),len(test_scenarios),len(noise_sigma_percentage_levels), n_test_trials))
}

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
# Plot curves of each metric (aggregated by test scenario) after RL for each reward
figure, ax = plt.subplots(math.ceil((len(all_metrics)-4)/3), 3, figsize=(10,10))
figure.suptitle(f"Metrics after RL training for each test with noise - All test scenarios - All test environments - {n_test_trials} trials")
figure.subplots_adjust(hspace=0.6, wspace=0.5, bottom=0.05, top=0.90, left=0.08, right=0.82)
idx = 0
for key, values in all_metrics.items():
    if key == "successes" or key == "collisions" or key == "timeouts":
        continue
    else:
        i = idx // 3
        j = idx % 3
        ax[i,j].set(
            xlabel='Noise sigma (%)', 
            ylabel=key)
        ax[i,j].set_xticks(noise_sigma_percentage_levels, labels=[n*100 for n in noise_sigma_percentage_levels])
        ax[i,j].grid()
        for policy in range(len(values)):
            ax[i,j].plot(noise_sigma_percentage_levels, jnp.nanmean(values[policy], axis=(0,1,3)), color=list(mcolors.TABLEAU_COLORS.values())[policy])
        idx += 1
figure.legend(policy_labels, loc="center right", title="Policy")
figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_after_rl_noisy_tests.eps"), format='eps')