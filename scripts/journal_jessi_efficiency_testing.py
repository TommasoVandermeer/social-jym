import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
from tabulate import tabulate
import matplotlib.colors as mcolors
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
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.utils.aux_functions import initialize_metrics_dict
from socialjym.policies.jessi import JESSI

# Hyperparameters
random_seed = 2_000_000 # Make sure test episodes are not the same as the training ones
n_trials = 100
n_steps_perception = 5_000
network_embeddings_dims = [8, 16, 32, 64, 96, 128, 160]
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
    "perception_loss": {"label": r"Perception Loss ($\mathcal{L}_{\text{perc}}$)"},
    "pos_reg_loss": {"label": r"Position Loss ($\mathcal{L}^{\text{pos}}_{\text{NLL}}$)"},
    "vel_reg_loss": {"label": r"Velocity Loss ($\mathcal{L}^{\text{vel}}_{\text{NLL}}$)"},
    "cls_loss": {"label": r"Classification Loss ($\mathcal{L}^{\text{cls}}_{\text{BCE}}$)"}
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
colors = list(mcolors.TABLEAU_COLORS.values())
policies = {
    f"jessi_efficiency_{embeddings_size}": {
        "label": f"JESSI-MULTITASK-{embeddings_size}", 
        "short": f"JESSI-{embeddings_size}", 
        "only_ccso": False, 
        "color": colors[idx]
    } for idx, embeddings_size in enumerate(network_embeddings_dims)
}

def jessi_tests(policy: JESSI, jessi_params: dict, only_perception: bool = False):
    metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
    if only_perception:
        all_metrics = {}
    else:
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    all_perception_losses = jnp.zeros(metrics_dims)
    all_position_losses = jnp.zeros(metrics_dims)
    all_velocity_losses = jnp.zeros(metrics_dims)
    all_classification_losses = jnp.zeros(metrics_dims)
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
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'training_scenario', 
                # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
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
            ## NAVIGATION METRICS
            # Test the trained JESSI-MULTITASK policy
            if not only_perception:
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
            ## PERCEPTION METRICS
            perception_metrics_seen_scenarios = policy.evaluate_perception(
                n_steps_perception,
                random_seed,
                seen_env,
                jessi_params,
            )
            perception_metrics_unseen_scenarios = policy.evaluate_perception(
                n_steps_perception,
                random_seed,
                ct_env,
                jessi_params,
            )
            all_perception_losses = all_perception_losses.at[0,i,j].set(perception_metrics_seen_scenarios["perception_loss"])
            all_perception_losses = all_perception_losses.at[1,i,j].set(perception_metrics_unseen_scenarios["perception_loss"])
            all_perception_losses = all_perception_losses.at[2,i,j].set(jnp.nan) # CCSO scenario is not evaluated for perception metrics
            all_position_losses = all_position_losses.at[0,i,j].set(perception_metrics_seen_scenarios["pos_reg_loss"])
            all_position_losses = all_position_losses.at[1,i,j].set(perception_metrics_unseen_scenarios["pos_reg_loss"])
            all_position_losses = all_position_losses.at[2,i,j].set(jnp.nan) # CCSO scenario is not evaluated for perception metrics
            all_velocity_losses = all_velocity_losses.at[0,i,j].set(perception_metrics_seen_scenarios["vel_reg_loss"])
            all_velocity_losses = all_velocity_losses.at[1,i,j].set(perception_metrics_unseen_scenarios["vel_reg_loss"])
            all_velocity_losses = all_velocity_losses.at[2,i,j].set(jnp.nan) # CCSO scenario is not evaluated for perception metrics
            all_classification_losses = all_classification_losses.at[0,i,j].set(perception_metrics_seen_scenarios["cls_loss"])
            all_classification_losses = all_classification_losses.at[1,i,j].set(perception_metrics_unseen_scenarios["cls_loss"])
            all_classification_losses = all_classification_losses.at[2,i,j].set(jnp.nan) # CCSO scenario is not evaluated for perception metrics
    all_metrics["perception_loss"] = all_perception_losses
    all_metrics["pos_reg_loss"] = all_position_losses
    all_metrics["vel_reg_loss"] = all_velocity_losses
    all_metrics["cls_loss"] = all_classification_losses
    return all_metrics

if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_efficiency_tests.pkl")):

    ### JESSI-MULTITASK EFFICIENCY TESTS ###
    for embeddings_size in network_embeddings_dims:
        if not os.path.exists(os.path.join(os.path.dirname(__file__),f"jessi_multitask_efficiency_{embeddings_size}_tests.pkl")):
            # Load JESSI-MULTITASK policy parameters
            with open(os.path.join(os.path.dirname(__file__), f"jessi_multitask_rl_out_{embeddings_size}.pkl"), 'rb') as f:
                _, jessi_params, _ = pickle.load(f)
            # Execute tests
            policy = JESSI(
                lidar_num_rays=100,
                lidar_angular_range=jnp.pi * 2,
                lidar_max_dist=10.0,
                n_stack=5,
                n_stack_for_action_space_bounding=1,
                embedding_dim=embeddings_size,
            )
            all_metrics = jessi_tests(policy, jessi_params)
            with open(os.path.join(os.path.dirname(__file__),f"jessi_multitask_efficiency_{embeddings_size}_tests.pkl"), 'wb') as f:
                pickle.dump(all_metrics, f)
        else:
            print(f"JESSI-{embeddings_size} tests already exist. Checking if perception losses are available...")
            with open(os.path.join(os.path.dirname(__file__),f"jessi_multitask_efficiency_{embeddings_size}_tests.pkl"), 'rb') as f:
                all_metrics = pickle.load(f)
                keys = [*all_metrics.keys()]
            if "perception_loss" not in keys or "pos_reg_loss" not in keys or "vel_reg_loss" not in keys or "cls_loss" not in keys:
                print(f"Perception losses not found for JESSI-{embeddings_size}. Running perception tests...")
                # Load JESSI-MULTITASK policy parameters
                with open(os.path.join(os.path.dirname(__file__), f"jessi_multitask_rl_out_{embeddings_size}.pkl"), 'rb') as f:
                    _, jessi_params, _ = pickle.load(f)
                # Execute tests
                policy = JESSI(
                    lidar_num_rays=100,
                    lidar_angular_range=jnp.pi * 2,
                    lidar_max_dist=10.0,
                    n_stack=5,
                    n_stack_for_action_space_bounding=1,
                    embedding_dim=embeddings_size,
                )
                perception_metrics = jessi_tests(policy, jessi_params, only_perception=True)
                for key in perception_metrics.keys():
                    all_metrics[key] = perception_metrics[key]
                with open(os.path.join(os.path.dirname(__file__),f"jessi_multitask_efficiency_{embeddings_size}_tests.pkl"), 'wb') as f:
                    pickle.dump(all_metrics, f)
            
    ### AGGREGATE ALL RESULTS ###
    # Load all test results and aggregate them in a single dictionary
    all_results = {}
    for embeddings_size in network_embeddings_dims:
         with open(os.path.join(os.path.dirname(__file__),f"jessi_multitask_efficiency_{embeddings_size}_tests.pkl"), 'rb') as f:
            all_results[f"jessi_efficiency_{embeddings_size}"] = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"jessi_efficiency_tests.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_efficiency_tests.pkl"), 'rb') as f:
        all_results = pickle.load(f)
        # for k, v in all_results.items(): print(k, ": ", v["perception_loss"])

### PRINT RESULTS SUMMARIES ###
n_humans_indexes_to_evaluate = [2] # referring to tests_n_humans
metrics_to_plot = ["returns","perception_loss","successes","collisions_with_human","collisions_with_obstacle","timeouts","times_to_goal","average_jerk","average_angular_jerk","space_compliance"]
higher_is_better = ["returns", "successes", "space_compliance"]
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
            y_data = jnp.nanmean(all_results[p][metric][0, :, n_humans_indexes_to_evaluate]) / n_trials
        elif metric == 'perception_loss':
            y_data = jnp.nanmean(all_results[p][metric][0, :, n_humans_indexes_to_evaluate])
        else:
            y_data = jnp.nanmean(all_results[p][metric][0, :, n_humans_indexes_to_evaluate, :])
        train_scenarios_summary[p][metric] = float(y_data)
    # TEST Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][1, :, n_humans_indexes_to_evaluate]) / n_trials
        elif metric == 'perception_loss':
            y_data = jnp.nanmean(all_results[p][metric][1, :, n_humans_indexes_to_evaluate])
        else:
            y_data = jnp.nanmean(all_results[p][metric][1, :, n_humans_indexes_to_evaluate, :])
        test_scenarios_summary[p][metric] = float(y_data)
    # CCSO Scenario
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, n_humans_indexes_to_evaluate]) / n_trials
        elif metric == 'perception_loss':
            y_data = jnp.nanmean(all_results[p][metric][idx, :, n_humans_indexes_to_evaluate])
        else:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, n_humans_indexes_to_evaluate, :])
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
            r"\makecell{Ret}", 
            r"\makecell{$\mathcal{L}_{perc}$}", 
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
        headers = ["Policy","Ret", "PL", "SR (%)", "CR-H (%)", "CR-O (%)", "TR (%)", "TtG (s)", "LJ (m/s^3)", "AJ (rad/s^3)", "SC (%)"]
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
print_pretty_table(complete_summary, "Overall Results", latex_mode=False)
print_pretty_table(train_and_test_scenarios_summary, "First experimental setup results", latex_mode=False)
print_pretty_table(train_scenarios_summary, "Train Scenarios Results", latex_mode=False)
print_pretty_table(test_scenarios_summary, "Test Scenarios Results", latex_mode=False)
print_pretty_table(ccso_scenarios_summary, "Second experimental setup results", latex_mode=False)

### PLOTS
## Plot Perception Loss and returns vs number of humans for each policy emebedding size
# Two columns: left column for perception loss, right column for returns
# Two rows: first row for train scenarios, second row for test scenarios
metrics_to_plot = [
    "perception_loss","pos_reg_loss","vel_reg_loss","cls_loss","returns", # First row: train scenarios
    "perception_loss","pos_reg_loss","vel_reg_loss","cls_loss","returns" # Second row: test scenarios
]
figure, ax = plt.subplots(2, 5, figsize=(20, 8))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 5
    j = m % 5
    ax[i,j].set(
        xlabel='N° humans',
        ylabel=metrics[metric]['label'],
        title=f"{'Train' if i == 0 else 'Test'} scenarios",
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][i, :, :], axis=0) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        elif metric in ['perception_loss', 'pos_reg_loss', 'vel_reg_loss', 'cls_loss']:
            y_data = jnp.nanmean(all_results[p][metric][i, :, :], axis=0)
        else:
            y_data = jnp.nanmean(all_results[p][metric][i, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(tests_n_humans)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_efficiency_perception_loss_and_returns.eps"), format='eps')