import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import optax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import math

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import test_k_trials
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl import SARL
from socialjym.policies.sarl_ppo import SARLPPO

### Hyperparameters
sarl_ppo_distribution = "gaussian"
robot_vmax = 1
n_humans_for_tests = [5, 10, 15, 20, 25]
n_trials = 1000
reward_function = Reward2(
    target_reached_reward=True,
    collision_penalty_reward=True,
    discomfort_penalty_reward=True,
    v_max=robot_vmax,
    # progress_to_goal_reward = True,
    # progress_to_goal_weight = 0.03,
    # high_rotation_penalty_reward=True,
    # angular_speed_bound=1.,
    # angular_speed_penalty_weight=0.035,
)

# Initialize and load policies
sarl_ppo = SARLPPO(reward_function, v_max=robot_vmax, dt=0.25, kinematics='unicycle', distribution=sarl_ppo_distribution)
sarl = SARL(reward_function, v_max=robot_vmax, dt=0.25, kinematics='unicycle', unicycle_box_action_space=True)
with open(os.path.join(os.path.dirname(__file__), 'sarl_params.pkl'), 'rb') as f:
    sarl_params = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), 'sarl_ppo_params.pkl'), 'rb') as f:
    sarl_ppo_params = pickle.load(f)['actor_params']
policies = [sarl, sarl_ppo]
policies_params = [sarl_params, sarl_ppo_params]

# Initialize output dictionary
outcomes_arrays = jnp.zeros((len(policies), len(n_humans_for_tests),))
other_metrics_arrays = jnp.zeros((len(policies), len(n_humans_for_tests), n_trials))
metrics_sarl_vs_sarl_ppo = {
    "successes": outcomes_arrays, 
    "collisions": outcomes_arrays, 
    "timeouts": outcomes_arrays, 
    "returns": other_metrics_arrays,
    "times_to_goal": other_metrics_arrays,
    "average_speed": other_metrics_arrays,
    "average_acceleration": other_metrics_arrays,
    "average_jerk": other_metrics_arrays,
    "average_angular_speed": other_metrics_arrays,
    "average_angular_acceleration": other_metrics_arrays,
    "average_angular_jerk": other_metrics_arrays,
    "min_distance": other_metrics_arrays,
    "space_compliance": other_metrics_arrays,
    "episodic_spl": other_metrics_arrays,
    "path_length": other_metrics_arrays,
    "scenario": jnp.zeros((len(policies), len(n_humans_for_tests), n_trials), dtype=jnp.int32),
}

# Test loop
for p in range(len(policies)):
    print(f"\n##############\n{'SARL' if p==0 else 'SARL-PPO'}")
    for test, n_humans in enumerate(n_humans_for_tests):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': 'hybrid_scenario',
            'hybrid_scenario_subset': jnp.array([0,1], jnp.int32), #jnp.array([0,1,2,3,4,5], jnp.int32),
            'humans_policy': 'hsfm',
            'circle_radius': 7,
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        test_env = SocialNav(**test_env_params)
        trial_out = test_k_trials(
            n_trials, 
            20_000, 
            test_env, 
            policies[p], 
            policies_params[p], 
            reward_function.time_limit
        )
        # Store trail metrics
        metrics_sarl_vs_sarl_ppo = tree_map(lambda x, y: x.at[p,test].set(y), metrics_sarl_vs_sarl_ppo, trial_out)

# Save metrics
with open(os.path.join(os.path.dirname(__file__),"metrics_sarl_vs_sarl_ppo.pkl"), 'wb') as f:
    pickle.dump(metrics_sarl_vs_sarl_ppo, f)
# Load metrics files
with open(os.path.join(os.path.dirname(__file__),"metrics_sarl_vs_sarl_ppo.pkl"), 'rb') as f:
    metrics_sarl_vs_sarl_ppo = pickle.load(f)

### Plotting
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 18}
rc('font', **font)
metrics_data = {
    "successes": {"row_position": 0, "col_position": 0, "label": "Success rate", "ylim": [0.,1.1], "yticks": [i/10 for i in range(0,11)]}, 
    "times_to_goal": {"row_position": 0, "col_position": 1, "label": "Time to goal ($s$)"},
    "average_angular_speed": {"row_position": 0, "col_position": 2, "label": "Angular speed ($rad/s$)"},
    "average_speed": {"row_position": 1, "col_position": 0, "label": "Speed ($m/s$)"}, 
    "episodic_spl": {"row_position": 1, "col_position": 1, "label": "SPL", "ylim": [0,1], "yticks": [i/10 for i in range(11)]},
    "average_angular_acceleration": {"row_position": 1, "col_position": 2, "label": "Angular acceleration ($rad/s^2$)"},
    "space_compliance": {"row_position": 2, "col_position": 0, "label": "Space compliance", "ylim": [0,1]},
    "average_acceleration": {"row_position": 2, "col_position": 1, "label": "Acceleration ($m/s^2$)"},
    "average_angular_jerk": {"row_position": 2, "col_position": 2, "label": "Angular jerk ($rad/s^3$)"},
    "average_jerk": {"row_position": 3, "col_position": 0, "label": "Jerk ($m/s^3$)"},
    "min_distance": {"row_position": 3, "col_position": 1, "label": "Min. dist. to humans ($m$)"},
    "returns": {"row_position": 3, "col_position": 2, "label": "Return"},
}
figure, ax = plt.subplots(math.ceil(len(metrics_data)/3), 3, figsize=(18,18))
figure.subplots_adjust(right=0.82, top=0.985, bottom=0.05, left=0.09, hspace=0.3, wspace=0.3)
for key in metrics_data:
    ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set(
        xlabel='Number of humans',
        ylabel=metrics_data[key]["label"])
    if "ylim" in metrics_data[key]:
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_ylim(metrics_data[key]["ylim"])
    if "yticks" in metrics_data[key]:
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_yticks(metrics_data[key]["yticks"])
    ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].set_xticks(jnp.arange(len(n_humans_for_tests)), labels=[i for i in n_humans_for_tests])
    ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].grid()
    for p, policy in enumerate(policies):
        ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
            jnp.arange(len(n_humans_for_tests)), 
            jnp.nanmean(metrics_sarl_vs_sarl_ppo[key][p], axis=1) if key != "successes" else metrics_sarl_vs_sarl_ppo[key][p] / n_trials,
            color=list(mcolors.TABLEAU_COLORS.values())[p],
            linewidth=2,
            label=f"{'SARL' if p == 0 else 'SARL-PPO'}",
        )
handles, labels = ax[0,0].get_legend_handles_labels()
figure.legend(labels, loc="center right", title=f"Policy:", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
figure.savefig(os.path.join(os.path.dirname(__file__),f"sarl_ppo_vs_sarl.eps"), format='eps')