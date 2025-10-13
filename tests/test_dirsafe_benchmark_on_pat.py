import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from jax.tree_util import tree_map
import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.base_policy import BasePolicy
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.sarl import SARL
from socialjym.policies.sarl_star import SARLStar
from socialjym.utils.aux_functions import \
    initialize_metrics_dict, \
    load_socialjym_policy, \
    test_k_trials_dwa, \
    test_k_trials, \
    test_k_trials_sfm, \
    test_k_trials_hsfm

### Hyperparameters
n_trials = 100
random_seed = 0
n_humans = [2,4,6]
n_obstacles = [1,3,5]
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
robot_vmax = 1.0
robot_wheels_distance = 0.7

### Initialize DIRSAFE policy
dirsafe = DIRSAFE(reward_function, v_max=robot_vmax, dt=0.25, wheels_distance=robot_wheels_distance)
with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
    dirsafe_params = pickle.load(f)['actor_params']
### Initialize SARL policy
sarl = SARL(reward_function, v_max=robot_vmax, dt=0.25, kinematics='unicycle', wheels_distance=robot_wheels_distance)
sarl_params = load_socialjym_policy(os.path.join(os.path.dirname(__file__), 'best_sarl.pkl'))
### Initialize SARL* policy
dummy_env = SocialNav(
    robot_radius=0.3,
    n_humans=5,
    robot_dt=0.25,
    humans_dt=0.01,
    scenario='hybrid_scenario',
    reward_function=reward_function,
    kinematics='unicycle',
)
sarl_star = SARLStar(
    reward_function,  
    dummy_env.get_grid_size(),
    use_planner=False, 
    v_max=robot_vmax, 
    dt=0.25, 
    kinematics='unicycle', 
    wheels_distance=robot_wheels_distance
)

### Initialize output data structure
policies = ['DIR-SAFE', 'DWA', 'SFM', 'HSFM', 'SARL*']
n_policies = len(policies)
metrics_dims = (n_policies,len(n_humans),len(n_obstacles))
all_metrics = initialize_metrics_dict(n_trials, dims=metrics_dims)

### Execute tests
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_on_pat_results.pkl')):
    for i, nh in enumerate(n_humans):
        for j, no in enumerate(n_obstacles):
            ## Initialize environment
            test_env_params = {
                'robot_radius': 0.3,
                'n_humans': nh,
                'n_obstacles': no,
                'robot_dt': 0.25,
                'humans_dt': 0.01,
                'robot_visible': True,
                'scenario': 'parallel_traffic',
                'humans_policy': 'hsfm',
                'reward_function': reward_function,
                'kinematics': 'unicycle',
                'ccso_n_static_humans': 0,
            }
            test_env = SocialNav(**test_env_params)
            ## DIR-SAFE tests
            print("\nDIR-SAFE Tests")
            metrics_dir_safe = test_k_trials(n_trials, random_seed, test_env, dirsafe, dirsafe_params, time_limit=50.)
            ## DWA Tests
            print("\nDWA Tests")
            metrics_dwa = test_k_trials_dwa(n_trials, random_seed, test_env, time_limit=50, robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
            ## SFM Tests
            print("\nSFM Tests")
            metrics_sfm = test_k_trials_sfm(n_trials, random_seed, test_env, time_limit=50., robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
            ## HSFM Tests
            print("\nHSFM Tests")
            metrics_hsfm = test_k_trials_hsfm(n_trials, random_seed, test_env, time_limit=50., robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
            ## SARL Tests
            print("\nSARL* Tests")
            metrics_sarl_star = test_k_trials(n_trials, random_seed, test_env, sarl_star, sarl_params, time_limit=50.)
            ### Store results
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_dir_safe)
            all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_dwa)
            all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_sfm)
            all_metrics = tree_map(lambda x, y: x.at[3,i,j].set(y), all_metrics, metrics_hsfm)
            all_metrics = tree_map(lambda x, y: x.at[4,i,j].set(y), all_metrics, metrics_sarl_star)
    ### Save results
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_on_pat_results.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
### Load results
else:
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_on_pat_results.pkl'), 'rb') as f:
        all_metrics = pickle.load(f)

### Plot results
# Matplotlib font
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 18
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
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
    "min_distance": {"label": "Min. dist. ($m$)", "episodic": True},
    "space_compliance": {"label": "Space compliance", "episodic": True},
    "episodic_spl": {"label": "Episodic SPL", "episodic": True},
    "path_length": {"label": "Path length ($m$)", "episodic": True},
    "feasible_actions_rate": {"label": "Action Feasibility Rate", "episodic": True},
}
metrics_to_plot = [
    "successes",
    "collisions",
    "timeouts",
    "feasible_actions_rate",
    "times_to_goal", 
    "path_length", 
    "average_speed",
    "average_acceleration", 
    "average_jerk", 
    "average_angular_speed", 
    "average_angular_acceleration",
    "average_angular_jerk",
    "min_distance",
    "episodic_spl", 
    "space_compliance",
    "returns"
]
colors = list(mcolors.TABLEAU_COLORS.values())

## Plot results against number of humans (averaged over number of obstacles)
nrows, ncols = 4, 4
figure, ax = plt.subplots(nrows, ncols, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // ncols
    j = m % ncols
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(n_humans)))
    ax[i,j].set_xticklabels(n_humans)
    for p, policy in enumerate(policies):
        if metric in ['successes', 'collisions', 'timeouts','feasible_actions_rate','space_compliance']:
            ax[i, j].set_ylim(-0.05, 1.05)
        if metric in ['successes', 'collisions', 'timeouts']:
            y_data = jnp.nanmean(all_metrics[metric][p, :, :], axis=1) / n_trials
        else:
            y_data = jnp.nanmean(all_metrics[metric][p, :, :, :], axis=(1,2))
        ax[i, j].plot(jnp.arange(len(n_humans)), y_data, label=policies[p], color=colors[p], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy')
figure.savefig(os.path.join(os.path.dirname(__file__), "dir_safe_benchmark_on_pat_1.eps"), format='eps')

## Plot results against number of obstacles (averaged over number of humans)
nrows, ncols = 4, 4
figure, ax = plt.subplots(nrows, ncols, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // ncols
    j = m % ncols
    ax[i,j].set(
        xlabel='N° obstacles',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(n_obstacles)))
    ax[i,j].set_xticklabels(n_obstacles)
    for p, policy in enumerate(policies):
        if metric in ['successes', 'collisions', 'timeouts','feasible_actions_rate','space_compliance']:
            ax[i, j].set_ylim(-0.05, 1.05)
        if metric in ['successes', 'collisions', 'timeouts']:
            y_data = jnp.nanmean(all_metrics[metric][p, :, :], axis=0) / n_trials
        else:
            y_data = jnp.nanmean(all_metrics[metric][p, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(n_obstacles)), y_data, label=policies[p], color=colors[p], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy')
figure.savefig(os.path.join(os.path.dirname(__file__), "dir_safe_benchmark_on_pat_2.eps"), format='eps')
