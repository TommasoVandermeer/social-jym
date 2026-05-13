import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
from tabulate import tabulate

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import initialize_metrics_dict


# Hyperparameters
random_seed = 1_000_000 # Make sure test episodes are not the same as the training ones
n_trials = 100
# Tests
tests_n_humans = [1, 3, 5, 10]
tests_n_obstacles = [1, 3, 5]

# Plot utils
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
    "jessi_multitask": {"label": "JESSI-MULTITASK", "short": "JESSI-MT", "only_ccso": False, "color": "tab:blue"},
    "jessi_modular": {"label": "JESSI-MODULAR", "short": "JESSI-MD", "only_ccso": False, "color": "tab:orange"},
    "jessi_policy": {"label": "JESSI-POLICY", "short": "JESSI-PO", "only_ccso": False, "color": "tab:green"},
}


def jessi_tests(jessi_params):
    metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    policy = JESSI(
        lidar_num_rays=100,
        lidar_angular_range=jnp.pi * 2,
        lidar_max_dist=10.0,
        n_stack=5,
        n_stack_for_action_space_bounding=1,
    )
    for i, n_obstacle in enumerate(tests_n_obstacles):
        for j, n_human in enumerate(tests_n_humans):
            seen_env_params = {
                'n_stack': 5,
                'lidar_num_rays': 100,
                'lidar_angular_range': jnp.pi * 2,
                'lidar_max_dist': 10.0,
                'lidar_dt': 0.13,
                'odometry_dt': 0.05,
                # 'tau_linear_velocity': 0.39,
                # 'tau_angular_velocity': 0.19,
                'control_delay_mean': 0.1, #0.27,
                'control_delay_sigma': 0.01, #0.01,
                'wheels_max_linear_acceleration': 0.87,
                'wheels_distance': policy.wheels_distance,
                'n_humans': n_human,
                'n_obstacles': n_obstacle,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
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
            ccso_env_params['ccso_n_static_humans'] = n_obstacle
            ccso_env_params['n_humans'] = n_human + n_obstacle
            # Initialize the environments
            seen_env = LaserNav(**seen_env_params)
            ct_env = LaserNav(**ct_env_params) # Unseen scenario
            ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
            # Test the trained JESSI-MULTITASK policy
            metrics_seen_scenarios = policy.evaluate(
                n_trials,
                random_seed,
                seen_env,
                jessi_params,
            )
            metrics_ct = policy.evaluate(
                n_trials,
                random_seed,
                ct_env,
                jessi_params,
            )
            metrics_ccso = policy.evaluate(
                n_trials,
                random_seed,
                ccso_env,
                jessi_params,
            )
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
            all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
            all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
    return all_metrics

if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests_velocity_dynamics.pkl")):
    ### JESSI-MULTITASK tests ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests_velocity_dynamics.pkl")):
        # Load JESSI-MULTITASK policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_multitask_rl_out.pkl'), 'rb') as f:
            _, jessi_params, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests_velocity_dynamics.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)    

    ### JESSI-MODULAR tests ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_modular_tests_velocity_dynamics.pkl")):
        # Load JESSI-MODULAR policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_modular_rl_out.pkl'), 'rb') as f:
            jessi_params, _, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_modular_tests_velocity_dynamics.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)
        
    ### JESSI-POLICY TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_policy_tests_velocity_dynamics.pkl")):
        # Load JESSI-POLICY policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_policy_rl_out.pkl'), 'rb') as f:
            jessi_params, _, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_policy_tests_velocity_dynamics.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)

    ### AGGREGATE ALL RESULTS ###
    # Load all test results and aggregate them in a single dictionary
    with open(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests_velocity_dynamics.pkl"), 'rb') as f:
            jessi_multitask_results = pickle.load(f)  
    with open(os.path.join(os.path.dirname(__file__),"jessi_modular_tests_velocity_dynamics.pkl"), 'rb') as f:
            jessi_modular_results = pickle.load(f) 
    with open(os.path.join(os.path.dirname(__file__),"jessi_policy_tests_velocity_dynamics.pkl"), 'rb') as f:
            jessi_policy_results = pickle.load(f)  
    all_results = {
        'jessi_multitask': jessi_multitask_results,
        'jessi_modular': jessi_modular_results,
        'jessi_policy': jessi_policy_results,
    }
    with open(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests_velocity_dynamics.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests_velocity_dynamics.pkl"), 'rb') as f:
        all_results = pickle.load(f)

### PRINT RESULTS SUMMARIES ###
metrics_to_plot = ["successes","collisions_with_human","collisions_with_obstacle","timeouts","times_to_goal","average_jerk","average_angular_jerk","space_compliance"]
higher_is_better = ["successes", "space_compliance"]
train_scenarios_summary = {p: {} for p in policies.keys()}
test_scenarios_summary = {p: {} for p in policies.keys()}
train_and_test_scenarios_summary = {p: {} for p in policies.keys()}
ccso_scenarios_summary = {p: {} for p in policies.keys()}
complete_summary = {p: {} for p in policies.keys()}
for metric in metrics_to_plot:
    # TRAIN Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :, :])
        train_scenarios_summary[p][metric] = float(y_data)
    # TEST Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :, :])
        test_scenarios_summary[p][metric] = float(y_data)
    # CCSO Scenario
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :, :])
        ccso_scenarios_summary[p][metric] = float(y_data)
    ### Complete summary & Train and Test scenarios summary
    for p in all_results.keys():
        v_train = train_scenarios_summary[p].get(metric, jnp.nan)
        v_test  = test_scenarios_summary[p].get(metric, jnp.nan)
        if not policies[p]["only_ccso"]:
            train_and_test_scenarios_summary[p][metric] = float(jnp.nanmean(jnp.array([v_train, v_test])))
        v_ccso  = ccso_scenarios_summary[p].get(metric, jnp.nan)
        avg_val = float(jnp.nanmean(jnp.array([v_train, v_test, v_ccso])))
        complete_summary[p][metric] = avg_val
def print_pretty_table(summary_dict, title, latex_mode=False):
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
        valid_vals = [m[metric] for p, m in summary_dict.items() if m and not jnp.isnan(m.get(metric, float('nan')))]
        if not valid_vals:
            continue
        unique_vals = list(set([round(v, 5) for v in valid_vals]))
        unique_vals.sort(reverse=(metric in higher_is_better))
        top_3_values[metric] = unique_vals[:3]
    table_data = []
    for p, metrics in summary_dict.items():
        if not metrics: continue
        row = [policies[p]["short"]]
        for metric in metrics_to_plot:
            val = metrics.get(metric, float('nan'))
            if jnp.isnan(val):
                row.append("N/A")
                continue
            if metric in ['successes', 'collisions_with_human', 'collisions_with_obstacle', 'timeouts','space_compliance']:
                val_str = f"{val*100:.1f}" if latex_mode else f"{val*100:.1f}%"
            else:
                val_str = f"{val:.2f}"
            val_rounded = round(val, 5)
            if metric in top_3_values and val_rounded in top_3_values[metric]:
                if latex_mode:
                    val_str = f"\\textbf{{{val_str}}}"
                else:
                    val_str = f"\033[1m{val_str}\033[0m" # Codice ANSI per grassetto terminale  
            row.append(val_str)
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
print_pretty_table(train_and_test_scenarios_summary, "First experimental setup results", latex_mode=True)
print_pretty_table(ccso_scenarios_summary, "Second experimental setup results", latex_mode=True)