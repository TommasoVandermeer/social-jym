import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
from tabulate import tabulate
import scipy.stats as stats

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2 as Reward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.utils.aux_functions import initialize_metrics_dict
from socialjym.policies.sarl import SARL
from socialjym.policies.sarl_star import SARLStar
from socialjym.policies.jessi import JESSI

# Hyperparameters
random_seed = 1_000_000 # Make sure test episodes are not the same as the training ones
n_trials = 100
n_training_seeds = 10
out_folder = 'sarl_seed_sensitivity_results'
base_rl_network_name = 'sarl_rl_out'
jessi_network_name = 'pre_perception_network.pkl'
robot_dt = 0.25
robot_vmax = 1.
# Tests
tests_n_humans = [1, 3, 5, 10]
tests_n_obstacles = [1, 3, 5]

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
    "precision": {"label": "Precision (%)"},
    "recall": {"label": "Recall (%) "},
    "ADE": {"label": "Displacement Error ($m$)"},
    "AVE": {"label": "Velocity Error ($m/s$)"},
    "mahalanobis_pos": {"label": "Mahalanobis Dist. Pos."},
    "mahalanobis_vel": {"label": "Mahalanobis Dist. Vel."},
}
scenarios = {
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "corner_traffic": {"label": "CoT"},
    "circular_crossing": {"label": "CiC"},
    "circular_crossing_with_static_obstacles": {"label": "CCSO"},
    "delayed_circular_crossing": {"label": "DCC"},
    "robot_crowding": {"label": "RoC"},
    "crowd_navigation": {"label": "CrN"},
    "door_crossing": {"label": "DoC"},
    "crowd_chasing": {"label": "CrC"},
}
policies = {
    "sarl_star": {"label": "SARL*", "short": "SARL*", "only_ccso": False, "color": "tab:brown"},
    "sarl": {"label": "SARL", "short": "SARL", "only_ccso": True, "color": "tab:pink"},
}

base_dir = os.path.join(os.path.join(os.path.dirname(__file__), out_folder))

def sarl_tests(network_params:dict, perception_params:dict):
    metrics_dims = (1,len(tests_n_obstacles),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    reward_function = Reward(
        v_max = robot_vmax,
        progress_to_goal_reward = True,
        high_rotation_penalty_reward = True,
        progress_to_goal_weight = 0.03,
        collision_penalty = -.5,
        angular_speed_bound= 1.,
        angular_speed_penalty_weight = 0.0075,
    )
    policy = policy = SARL(
        reward_function, 
        v_max=robot_vmax, 
        dt=robot_dt, 
        kinematics='unicycle',
    )
    jessi = JESSI(
        lidar_num_rays=100,
        lidar_angular_range=jnp.pi * 2,
        lidar_max_dist=10.0,
        n_stack=5,
        n_stack_for_action_space_bounding=1,
    )
    humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
    for i, n_obstacle in enumerate(tests_n_obstacles):
        for j, n_human in enumerate(tests_n_humans):
            ccso_env_params = {
                'n_stack': 5,
                'lidar_num_rays': 100,
                'lidar_angular_range': jnp.pi * 2,
                'lidar_max_dist': 10.0,
                'n_humans': n_human + n_obstacle * 2,
                'n_obstacles': 0,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'circular_crossing_with_static_obstacles', 
                'ccso_n_static_humans': n_obstacle * 2,
                'ccso_static_humans_radius_mean': 0.3,
                'ccso_static_humans_radius_std': 0.025,
                'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            # Initialize the environments
            ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
            # Test the trained JESSI-MULTITASK policy
            metrics_ccso = policy.evaluate_on_jessi_perception(
                n_trials,
                random_seed,
                ccso_env,
                jessi,
                perception_params,
                network_params,
                humans_radius_hypotheses,
            )
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_ccso)
    return all_metrics

def sarl_star_tests(network_params:dict, perception_params:dict):
    metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    dummy_env = LaserNav(kinematics='unicycle', reward_function=LaserReward(robot_radius=0.3), robot_radius=0.3, humans_dt=0.01, robot_dt=0.25, n_humans=1, n_obstacles=1, scenario='hybrid_scenario')
    reward_function = Reward(
        v_max = robot_vmax,
        progress_to_goal_reward = True,
        high_rotation_penalty_reward = True,
        progress_to_goal_weight = 0.03,
        collision_penalty = -.5,
        angular_speed_bound= 1.,
        angular_speed_penalty_weight = 0.0075,
    )
    policy = policy = SARLStar(
        reward_function, 
        v_max=robot_vmax, 
        dt=robot_dt, 
        kinematics='unicycle',
        grid_size = dummy_env.get_grid_size(),
        use_planner = False,
        safety_margin=0.03,
    )
    jessi = JESSI(
        lidar_num_rays=100,
        lidar_angular_range=jnp.pi * 2,
        lidar_max_dist=10.0,
        n_stack=5,
        n_stack_for_action_space_bounding=1,
    )
    humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
    for i, n_obstacle in enumerate(tests_n_obstacles):
        for j, n_human in enumerate(tests_n_humans):
            seen_env_params = {
                'n_stack': 5,
                'lidar_num_rays': 100,
                'lidar_angular_range': jnp.pi * 2,
                'lidar_max_dist': 10.0,
                'n_humans': n_human,
                'n_obstacles': n_obstacle,
                'robot_radius': 0.3,
                'robot_dt': robot_dt,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'training_scenario', 
                'ccso_n_static_humans': 0,
                'ccso_static_humans_radius_mean': 0.3,
                'ccso_static_humans_radius_std': 0.025,
                'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            ct_env_params = seen_env_params.copy()
            ct_env_params['scenario'] = 'testing_scenario'
            ccso_env_params = seen_env_params.copy()
            ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
            ccso_env_params['ccso_n_static_humans'] = n_obstacle * 2
            ccso_env_params['n_humans'] = n_human + n_obstacle * 2
            # Initialize the environments
            seen_env = LaserNav(**seen_env_params)
            ct_env = LaserNav(**ct_env_params) # Unseen scenario
            ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
            # Test the trained JESSI-MULTITASK policy
            metrics_seen_scenarios = policy.evaluate_on_jessi_perception(
                n_trials,
                random_seed,
                seen_env,
                jessi,
                perception_params,
                network_params,
                humans_radius_hypotheses,
            )
            metrics_ct = policy.evaluate_on_jessi_perception(
                n_trials,
                random_seed,
                ct_env,
                jessi,
                perception_params,
                network_params,
                humans_radius_hypotheses,
            )
            metrics_ccso = policy.evaluate_on_jessi_perception(
                n_trials,
                random_seed,
                ccso_env,
                jessi,
                perception_params,
                network_params,
                humans_radius_hypotheses,
            )
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
            all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
            all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
    return all_metrics

if not os.path.exists(os.path.join(base_dir,"sarl_seed_sensitivity_tests.pkl")):

    # Load JESSI perception
    with open(os.path.join(base_dir, jessi_network_name), 'rb') as f:
        perception_params = pickle.load(f)

    for seed in range(n_training_seeds):
        network_file_name = base_rl_network_name + '_seed' + str(seed) + '.pkl'
        with open(os.path.join(base_dir, network_file_name), 'rb') as f:
            network_params = pickle.load(f)['model_params']
        ### SARL tests ###
        if not os.path.exists(os.path.join(base_dir,f"sarl_seed{seed}_tests.pkl")):
            all_metrics = sarl_tests(network_params, perception_params)
            with open(os.path.join(base_dir,f"sarl_seed{seed}_tests.pkl"), 'wb') as f:
                pickle.dump(all_metrics, f)
        ### SARL-STAR tests ###
        if not os.path.exists(os.path.join(base_dir,f"sarl_star_seed{seed}_tests.pkl")):
            all_metrics = sarl_star_tests(network_params, perception_params)
            with open(os.path.join(base_dir,f"sarl_star_seed{seed}_tests.pkl"), 'wb') as f:
                pickle.dump(all_metrics, f)

    ### AGGREGATE ALL RESULTS ###
    # Load all test results and aggregate them in a single dictionary
    metrics_dims = (n_training_seeds, 3, len(tests_n_obstacles),len(tests_n_humans))
    all_sarl = initialize_metrics_dict(n_trials, metrics_dims)
    for seed in range(n_training_seeds):
        with open(os.path.join(base_dir,f"sarl_seed{seed}_tests.pkl"), 'rb') as f:
                dir_safe_results = pickle.load(f)  
        all_sarl = tree_map(lambda x, y: x.at[seed].set(y), all_sarl, dir_safe_results)
    all_sarl_star = initialize_metrics_dict(n_trials, metrics_dims)
    for seed in range(n_training_seeds):
        with open(os.path.join(base_dir,f"sarl_star_seed{seed}_tests.pkl"), 'rb') as f:
                dir_safe_results = pickle.load(f)  
        all_sarl_star = tree_map(lambda x, y: x.at[seed].set(y), all_sarl_star, dir_safe_results)
    all_results = {
        'sarl': all_sarl,
        'sarl_star': all_sarl_star,
    }
    with open(os.path.join(base_dir,"sarl_seed_sensitivity_tests.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
else:
    with open(os.path.join(base_dir,"sarl_seed_sensitivity_tests.pkl"), 'rb') as f:
        all_results = pickle.load(f)

### PRINT RESULTS SUMMARIES ###
metrics_to_plot = ["successes","collisions_with_human","collisions_with_obstacle","timeouts","times_to_goal","average_jerk","average_angular_jerk","space_compliance"]
higher_is_better = ["successes", "space_compliance"]
# Means
train_scenarios_mean_summary = {p: {} for p in policies.keys()}
test_scenarios_mean_summary = {p: {} for p in policies.keys()}
ccso_scenarios_mean_summary = {p: {} for p in policies.keys()}
# Stds
train_scenarios_std_summary = {p: {} for p in policies.keys()}
test_scenarios_std_summary = {p: {} for p in policies.keys()}
ccso_scenarios_std_summary = {p: {} for p in policies.keys()}
# Maxs
train_scenarios_max_summary = {p: {} for p in policies.keys()}
test_scenarios_max_summary = {p: {} for p in policies.keys()}
ccso_scenarios_max_summary = {p: {} for p in policies.keys()}
# Mins
train_scenarios_min_summary = {p: {} for p in policies.keys()}
test_scenarios_min_summary = {p: {} for p in policies.keys()}
ccso_scenarios_min_summary = {p: {} for p in policies.keys()}
for metric in metrics_to_plot:
    # TRAIN Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][:,0, :, :]) / n_trials
            y_data_std = jnp.nanstd(all_results[p][metric][:,0, :, :]) / n_trials
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,0, :, :], axis=(1,2)) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][:,0, :, :, :])
            y_data_std = jnp.nanstd(all_results[p][metric][:,0, :, :, :])
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,0, :, :, :], axis=(1,2,3))
        train_scenarios_mean_summary[p][metric] = float(y_data)
        train_scenarios_std_summary[p][metric] = float(y_data_std)
        train_scenarios_max_summary[p][metric] = float(jnp.nanmax(y_data_raw))
        train_scenarios_min_summary[p][metric] = float(jnp.nanmin(y_data_raw))
    # TEST Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][:,1, :, :]) / n_trials
            y_data_std = jnp.nanstd(all_results[p][metric][:,1, :, :]) / n_trials
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,1, :, :], axis=(1,2)) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][:,1, :, :, :])
            y_data_std = jnp.nanstd(all_results[p][metric][:,1, :, :, :])
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,1, :, :, :], axis=(1,2,3))
        test_scenarios_mean_summary[p][metric] = float(y_data)
        test_scenarios_std_summary[p][metric] = float(y_data_std)
        test_scenarios_max_summary[p][metric] = float(jnp.nanmax(y_data_raw))
        test_scenarios_min_summary[p][metric] = float(jnp.nanmin(y_data_raw))
    # CCSO Scenario
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][:,idx, :, :]) / n_trials
            y_data_std = jnp.nanstd(all_results[p][metric][:,idx, :, :]) / n_trials
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,idx, :, :], axis=(1,2)) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][:,idx, :, :, :])
            y_data_std = jnp.nanstd(all_results[p][metric][:,idx, :, :, :])
            y_data_std = stats.t.ppf(0.975, 9) * y_data_std / jnp.sqrt(n_training_seeds)
            y_data_raw = jnp.nanmean(all_results[p][metric][:,idx, :, :, :], axis=(1,2,3))
        ccso_scenarios_mean_summary[p][metric] = float(y_data)
        ccso_scenarios_std_summary[p][metric] = float(y_data_std)
        ccso_scenarios_max_summary[p][metric] = float(jnp.nanmax(y_data_raw))
        ccso_scenarios_min_summary[p][metric] = float(jnp.nanmin(y_data_raw))
def print_pretty_table(mean_or_min_summary_dict, std_or_max_summary_dict, title, latex_mode=False, max_min=False):
    print(f"\n{'-'*30} {title.upper()} {'-'*30}")
    if latex_mode:
        headers = [
            "Policy", 
            r"\makecell{SR \\ (\%)}", 
            r"\makecell{CR-H \\ (\%)}", 
            r"\makecell{CR-O \\ (\%)}", 
            r"\makecell{TR \\ (\%)}", 
            r"\makecell{TtG \\ (s)}", 
            r"\makecell{LJ \\ (m/s$^3$)}", 
            r"\makecell{AJ \\ (rad/s$^3$)}", 
            r"\makecell{SC \\ (\%)}"
        ]
    else:
        headers = ["Policy", "SR (%)", "CR-H (%)", "CR-O (%)", "TR (%)", "TtG (s)", "LJ (m/s^3)", "AJ (rad/s^3)", "SC (%)"]
    top_3_values = {}
    for metric in metrics_to_plot:
        valid_vals = [m[metric] for p, m in mean_or_min_summary_dict.items() if m and not jnp.isnan(m.get(metric, float('nan')))]
        if not valid_vals:
            continue
        unique_vals = list(set([round(v, 5) for v in valid_vals]))
        unique_vals.sort(reverse=(metric in higher_is_better))
        top_3_values[metric] = unique_vals[:3]
    table_data = []
    for p, metrics in mean_or_min_summary_dict.items():
        if not metrics: continue
        std_metrics = std_or_max_summary_dict.get(p, {})
        row = [policies[p]["short"]]
        for metric in metrics_to_plot:
            val = metrics.get(metric, float('nan'))
            std_val = std_metrics.get(metric, float('nan'))
            if jnp.isnan(val):
                row.append("N/A")
                continue
            if metric in ['successes', 'collisions_with_human', 'collisions_with_obstacle', 'timeouts','space_compliance']:
                if latex_mode:
                    val_str = f"{val*100:.1f}"
                    std_str = f"{std_val*100:.1f}" if not jnp.isnan(std_val) else "0.0"
                    if not max_min:
                        cell_str = f"{val_str} $\\pm$ {std_str}"
                    else:
                        cell_str = f"{val_str} $-$ {std_str}"
                else:
                    val_str = f"{val*100:.1f}"
                    std_str = f"{std_val*100:.1f}" if not jnp.isnan(std_val) else "0.0"
                    if not max_min:
                        cell_str = f"{val_str} ± {std_str}%"
                    else:
                        cell_str = f"{val_str}% - {std_str}%"
            else:
                if latex_mode:
                    val_str = f"{val:.2f}"
                    std_str = f"{std_val:.2f}" if not jnp.isnan(std_val) else "0.00"
                    if not max_min:
                        cell_str = f"{val_str} $\\pm$ {std_str}"
                    else:
                        cell_str = f"{val_str} $-$ {std_str}"
                else:
                    val_str = f"{val:.2f}"
                    std_str = f"{std_val:.2f}" if not jnp.isnan(std_val) else "0.00"
                    if not max_min:
                        cell_str = f"{val_str} ± {std_str}"
                    else:
                        cell_str = f"{val_str} - {std_str}"
            val_rounded = round(val, 5)
            if metric in top_3_values and val_rounded in top_3_values[metric]:
                if latex_mode:
                    cell_str = f"\\textbf{{{cell_str}}}"
                else:
                    cell_str = f"\033[1m{cell_str}\033[0m"
            row.append(cell_str)
        table_data.append(row)
    if latex_mode:
        num_cols = len(headers)
        col_format = "c" * num_cols
        latex_lines = []
        latex_lines.append(f"\\begin{{table}}[thpb]")
        latex_lines.append(f"\\centering")
        latex_lines.append(f"\\caption{{{title}}}")
        latex_lines.append(f"\\resizebox{{\\columnwidth}}{{!}}{{")
        latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex_lines.append(f"\\toprule")
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append(f"\\midrule")
        for row in table_data:
            latex_lines.append(" & ".join(str(item) for item in row) + " \\\\")
        latex_lines.append(f"\\bottomrule")
        latex_lines.append(f"\\end{{tabular}}")
        latex_lines.append(f"}}")
        latex_lines.append(f"\\label{{tab:{title.lower().replace(' ', '_')}}}")
        latex_lines.append(f"\\end{{table}}\n")
        print("\n".join(latex_lines))
    else:
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))
print_pretty_table(train_scenarios_mean_summary, train_scenarios_std_summary, "First setup", latex_mode=False)
print_pretty_table(test_scenarios_mean_summary, test_scenarios_std_summary, "Second setup", latex_mode=False)
print_pretty_table(ccso_scenarios_mean_summary, ccso_scenarios_std_summary, "Third setup", latex_mode=False)
print_pretty_table(train_scenarios_min_summary, train_scenarios_max_summary, "First setup (min-max)", latex_mode=False, max_min=True)
print_pretty_table(test_scenarios_min_summary, test_scenarios_max_summary, "Second setup (min-max)", latex_mode=False, max_min=True)
print_pretty_table(ccso_scenarios_min_summary, ccso_scenarios_max_summary, "Third setup (min-max)", latex_mode=False, max_min=True)