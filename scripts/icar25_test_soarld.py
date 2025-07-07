import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.soappo import SOAPPO
from socialjym.utils.aux_functions import test_k_trials

### Hyperparameters
random_seed = 0 
tests_n_humans = [5, 10, 15]
tests_scenarios = ["parallel_traffic", "perpendicular_traffic", "corner_traffic"]
n_obstacles = 5
n_trials = 1000
reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        v_max = 1.,
        progress_to_goal_reward = True,
        progress_to_goal_weight = 0.03,
        high_rotation_penalty_reward=True,
        angular_speed_bound=1.,
        angular_speed_penalty_weight=0.0075,
    )

### Initialize robot policy
policy = SOAPPO(reward_function, v_max=1., dt=0.25)
with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']

### Initialize output data structure
empty_trials_outcomes_array = jnp.zeros((len(tests_scenarios),len(tests_n_humans)))
empty_trials_metrics_array = jnp.zeros((len(tests_scenarios),len(tests_n_humans),n_trials))
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
    "path_length": empty_trials_metrics_array,
    "scenario": jnp.zeros((len(tests_scenarios),len(tests_n_humans),n_trials), dtype=jnp.int32),
}

# ### Test policy
# for i, scenario in enumerate(tests_scenarios):
#     for j, n_humans in enumerate(tests_n_humans):
#         print(f"\n## Testing scenario: {scenario}, n_humans: {n_humans} ###")
#         test_env_params = {
#             'robot_radius': 0.3,
#             'n_humans': n_humans,
#             'n_obstacles': n_obstacles,
#             'robot_dt': 0.25,
#             'humans_dt': 0.01,
#             'robot_visible': True,
#             'scenario': scenario,
#             'humans_policy': 'hsfm',
#             'reward_function': reward_function,
#             'kinematics': 'unicycle',
#         }
#         test_env = SocialNav(**test_env_params)
#         metrics = test_k_trials(
#             n_trials, 
#             random_seed, 
#             test_env, 
#             policy, 
#             actor_params, 
#             reward_function.time_limit
#         )
#         ### Save results
#         all_metrics = tree_map(lambda x, y: x.at[i,j].set(y), all_metrics, metrics)

# ### Save results
# with open(os.path.join(os.path.dirname(__file__),"soarld_tests.pkl"), 'wb') as f:
#     pickle.dump(all_metrics, f)

### Load results
with open(os.path.join(os.path.dirname(__file__),"soarld_tests.pkl"), 'rb') as f:
    all_metrics = pickle.load(f)

### Plot results
# Matplotlib font
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 23}
rc('font', **font)
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "timeouts": {"label": "Timeout Rate", "episodic": False}, 
    "returns": {"label": "Return ($\gamma = 0.9$)", "episodic": True},
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
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "corner_traffic": {"label": "CoT"},
}
# Plot metrics for each test scenario against number of humans
metrics_to_plot = ["successes","collisions","timeouts","times_to_goal", "path_length", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
colors = ["red", "green", "blue"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.85)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],)
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(tests_n_humans)
    ax[i,j].set_xticklabels(tests_n_humans)
    for s, scenario in enumerate(tests_scenarios):
        if metric in ['successes', 'collisions', 'timeouts']:
            y_data = all_metrics[metric][s, :] / n_trials
            ax[i, j].set_ylim(0, 1)
        else:
            y_data = jnp.nanmean(all_metrics[metric][s, :, :], axis=1)
        ax[i, j].plot(tests_n_humans, y_data, label=scenarios[scenario]["label"], color=colors[s])
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right')
figure.savefig(os.path.join(os.path.dirname(__file__), "icar25_soarld_test_results.eps"), format='eps')