import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import random
import os
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import initialize_metrics_dict

# Hyperparameters
random_seed = 0
n_trials = 200
# Tests
tests_n_humans = [1, 3, 5, 10]
tests_n_obstacles = [1, 3, 5]
# Policy parameters
lidar_max_dist = 10.0
with open(os.path.join(os.path.dirname(__file__), 'jessi_rl_out.pkl'), 'rb') as f:
    network_params, _, _ = pickle.load(f)
# Lidar configurations
lidar_configurations = [ # (num_rays, angular_range, n_stack) #
    (100, jnp.pi * 2, 5), # Training conditions (used for main tests)
    (50, jnp.pi * 2, 5), # Reduced resolution
    (100, jnp.pi, 5), # Reduced angular range
    (100, jnp.pi * 2, 3), # Reduced n_stack
    (200, jnp.pi * 2, 3), # Augmented resolution
    (100, jnp.pi * 2, 8), # Augmented n_stack
    # Keep this as last tests
    (100, (jnp.pi / 180) * 70, 5), # Heavily reduced angular range (70°, LOOMO-like)
]
n_stack_for_action_space_bounding = [1, 3, 5] # For final tests with heavily reduced angular range
# Plots utils
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "collisions_with_human": {"label": "Coll. w/ Hum. Rate", "episodic": False},
    "collisions_with_obstacle": {"label": "Coll. w/ Obs. Rate", "episodic": False},
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
    "corner_traffic": {"label": "CT"},
    "circular_crossing": {"label": "CC"},
    "circular_crossing_with_static_obstacles": {"label": "CCSO"},
    "delayed_circular_crossing": {"label": "DCC"},
    "robot_crowding": {"label": "RC"},
    "crowd_navigation": {"label": "CN"},
}

### TEST JESSI WITH DIFFERENT ENVIRONMENT CONFIGURATIONS ON SEEN VS UNSEEN SCENARIOS ###
if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_preliminary_tests.pkl")):
    metrics_dims = (2,len(tests_n_obstacles),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    policy = JESSI(
        lidar_num_rays=lidar_configurations[0][0],
        lidar_angular_range=lidar_configurations[0][1],
        lidar_max_dist=lidar_max_dist,
        n_stack=lidar_configurations[0][2],
        n_stack_for_action_space_bounding=1,
    )
    for i, n_obstacle in enumerate(tests_n_obstacles):
        for j, n_human in enumerate(tests_n_humans):
            seen_env_params = {
                'n_stack': lidar_configurations[0][2],
                'lidar_num_rays': lidar_configurations[0][0],
                'lidar_angular_range': lidar_configurations[0][1],
                'lidar_max_dist': lidar_max_dist,
                'n_humans': n_human,
                'n_obstacles': n_obstacle,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'hybrid_scenario', 
                'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                'ccso_n_static_humans': 0,
                'reward_function': Reward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            ct_env_params = seen_env_params.copy()
            ct_env_params['scenario'] = 'corner_traffic'
            ccso_env_params = seen_env_params.copy()
            ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
            ccso_env_params['ccso_n_static_humans'] = n_obstacle
            ccso_env_params['n_humans'] = n_human + n_obstacle
            # Initialize the environments
            seen_env = LaserNav(**seen_env_params)
            ct_env = LaserNav(**ct_env_params) # Unseen scenario
            ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
            # Test the trained JESSI policy
            metrics_seen_scenarios = policy.evaluate(
                n_trials,
                random_seed,
                seen_env,
                network_params,
            )
            metrics_ct = policy.evaluate(
                n_trials//2,
                random_seed,
                ct_env,
                network_params,
            )
            metrics_ccso = policy.evaluate(
                n_trials//2,
                random_seed,
                ccso_env,
                network_params,
            )
            metrics_unseen_scenarios = tree_map(
                lambda x, y: x + y if len(x.shape)==0 else jnp.append(x, y), 
                metrics_ct, 
                metrics_ccso
            )
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
            all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_unseen_scenarios)
    with open(os.path.join(os.path.dirname(__file__),"jessi_preliminary_tests.pkl"), 'wb') as f:
        pickle.dump(all_metrics, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_preliminary_tests.pkl"), 'rb') as f:
        all_metrics = pickle.load(f)          
## PLOTS
# Plot metrics for each test scenario against number of humans
metrics_to_plot = ["successes","collisions","timeouts","times_to_goal", "path_length", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
colors = ["green", "red"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for s in range(len(all_metrics[metric])):
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_metrics[metric][s, :, :], axis=0) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_metrics[metric][s, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(tests_n_humans)), y_data, label="SEEN" if s == 0 else "UNSEEN", color=colors[s], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Scenarios')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_preliminary_tests_1.eps"), format='eps')
# Plot metrics for each test scenario against number of obstacles
metrics_to_plot = ["successes","collisions","timeouts","times_to_goal", "path_length", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
colors = ["green", "red"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° obstacles',
        title=metrics[metric]['label'],)
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_obstacles)))
    ax[i,j].set_xticklabels(tests_n_obstacles)
    for s in range(len(all_metrics[metric])):
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_metrics[metric][s, :, :], axis=1) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_metrics[metric][s, :, :, :], axis=(1,2))
        ax[i, j].plot(jnp.arange(len(tests_n_obstacles)), y_data, label="SEEN" if s == 0 else "UNSEEN", color=colors[s], linewidth=2)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Scenarios')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_preliminary_tests_2.eps"), format='eps')


### TEST JESSI WITH DIFFERENT LIDAR CONFIGURATIONS (ALSO INFERENCE TIME) ON INCREASING N_HUMANS (TRAINING CONDITIONS) ###
if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_lidar_configuration_tests.pkl")):
    metrics_dims = (len(lidar_configurations),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    inference_times = jnp.zeros((len(lidar_configurations), len(tests_n_humans)))
    for i, lidar_config in enumerate(lidar_configurations):
        for j, n_human in enumerate(tests_n_humans):
            policy = JESSI(
                lidar_num_rays=lidar_config[0],
                lidar_angular_range=lidar_config[1],
                lidar_max_dist=lidar_max_dist,
                n_stack=lidar_config[2],
                n_stack_for_action_space_bounding=1,
            )
            env_params = {
                'n_stack':lidar_config[2],
                'lidar_num_rays':lidar_config[0],
                'lidar_angular_range':lidar_config[1],
                'lidar_max_dist': lidar_max_dist,
                'n_humans': n_human,
                'n_obstacles': 3,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'hybrid_scenario', 
                'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                'ccso_n_static_humans': 0,
                'reward_function': Reward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            env = LaserNav(**env_params)
            # Measure inference time
            _, _, obs, info, _ = env.reset(random.PRNGKey(random_seed))
            start_time = time.time()
            for i in range(100): policy.act(random.PRNGKey(random_seed+i), obs, info, network_params)
            end_time = time.time()
            inference_times = inference_times.at[i,j].set((end_time - start_time)/100)
            # Test performance
            metrics = policy.evaluate(
                n_trials,
                random_seed,
                env,
                network_params,
            )
            all_metrics = tree_map(lambda x, y: x.at[i,j].set(y), all_metrics, metrics)
    inference_times = jnp.mean(inference_times, axis=1)
    all_metrics['inference_times'] = inference_times
    with open(os.path.join(os.path.dirname(__file__),"jessi_lidar_configuration_tests.pkl"), 'wb') as f:
        pickle.dump(all_metrics, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_lidar_configuration_tests.pkl"), 'rb') as f:
        all_metrics = pickle.load(f)   
## PLOTS
# Plot metrics for each test scenario against number of humans
metrics_to_plot = ["successes","collisions","timeouts","collisions_with_obstacle","collisions_with_human","times_to_goal", "path_length", "average_speed", "average_angular_speed","episodic_spl", "space_compliance","returns"]
colors = ["green", "red", "blue", "orange", "purple", "brown", "pink"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for l in range(len(all_metrics[metric])):
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = all_metrics[metric][l, :] / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_metrics[metric][l, :, :], axis=(1))
        ax[i, j].plot(
            jnp.arange(len(tests_n_humans)), 
            y_data, 
            label=f"({lidar_configurations[l][0]}, {jnp.rad2deg(lidar_configurations[l][1]):.2f}°, {lidar_configurations[l][2]}) - {all_metrics['inference_times'][l]*1000:.2f}ms", 
            color=colors[l], 
            linewidth=2.5
        )
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='(rays, range, stacks) - time[ms]')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_lidar_config_tests.eps"), format='eps')


### TEST HEAVILY REDUCED LIDAR ANGULAR RANGE WITH DIFFERENT N_STACK FOR ACTION SPACE BOUNDING (AND ITS INFERENCE TIME) ###
if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_lidar_reduced_range_tests.pkl")):
    metrics_dims = (len(n_stack_for_action_space_bounding),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    inference_times = jnp.zeros((len(n_stack_for_action_space_bounding),len(tests_n_humans)))
    for i, nstack_asb in enumerate(n_stack_for_action_space_bounding):
        for j, n_human in enumerate(tests_n_humans):
            policy = JESSI(
                lidar_num_rays=lidar_configurations[-1][0],
                lidar_angular_range=lidar_configurations[-1][1],
                lidar_max_dist=lidar_max_dist,
                n_stack=lidar_configurations[-1][2],
                n_stack_for_action_space_bounding=nstack_asb,
            )
            env_params = {
                'n_stack':lidar_configurations[-1][2],
                'lidar_num_rays':lidar_configurations[-1][0],
                'lidar_angular_range':lidar_configurations[-1][1],
                'lidar_max_dist': lidar_max_dist,
                'n_humans': n_human,
                'n_obstacles': 3,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'hybrid_scenario', 
                'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                'ccso_n_static_humans': 0,
                'reward_function': Reward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            env = LaserNav(**env_params)
            # Measure inference time
            _, _, obs, info, _ = env.reset(random.PRNGKey(random_seed))
            start_time = time.time()
            for i in range(100): policy.act(random.PRNGKey(random_seed+i), obs, info, network_params)
            end_time = time.time()
            inference_times = inference_times.at[i,j].set((end_time - start_time)/100)
            # Test performance
            metrics = policy.evaluate(
                n_trials,
                random_seed,
                env,
                network_params,
            )
            all_metrics = tree_map(lambda x, y: x.at[i,j].set(y), all_metrics, metrics)
    inference_times = jnp.mean(inference_times, axis=1)
    all_metrics['inference_times'] = inference_times
    with open(os.path.join(os.path.dirname(__file__),"jessi_lidar_reduced_range_tests.pkl"), 'wb') as f:
        pickle.dump(all_metrics, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_lidar_reduced_range_tests.pkl"), 'rb') as f:
        all_metrics = pickle.load(f)           
## PLOTS
# Plot metrics for each test scenario against number of humans
metrics_to_plot = ["successes","collisions","timeouts","collisions_with_obstacle","collisions_with_human","times_to_goal", "path_length", "average_speed", "average_angular_speed","episodic_spl", "space_compliance","returns"]
colors = ["green", "red", "blue", "orange", "purple", "brown", "pink"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for l in range(len(all_metrics[metric])):
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = all_metrics[metric][l] / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_metrics[metric][l, :, :], axis=(1))
        ax[i, j].plot(
            jnp.arange(len(tests_n_humans)), 
            y_data, 
            label=f"{n_stack_for_action_space_bounding[l]} - {all_metrics['inference_times'][l]*1000:.2f}ms", 
            color=colors[l], 
            linewidth=2.5
        )
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title=f'Bounding stacks - time[ms]\n(rays: {lidar_configurations[-1][0]}, range: {jnp.rad2deg(lidar_configurations[-1][1]):.2f}° , stacks: {lidar_configurations[-1][2]})')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_lidar_reduced_range_tests.eps"), format='eps')


### TEST PERCEPTION ACCURACY WITH TRAINING CONDITIONS BEFORE AND AFTER RL (ON SEEN AND UNSEEN SCENARIOS) ###
# Accuracy in terms of Probabilistic Coverage for HCGs with scores > 1. For position and velocity.