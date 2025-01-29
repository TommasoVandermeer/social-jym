import os
import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

### PARAMETERS
train_scenarios = ["delayed_circular_crossing"] # ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "robot_crowding", "delayed_circular_crossing", "hybrid_scenario"]
train_envs = ["sfm","hsfm"] # ["sfm", "hsfm"]
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty'] # ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
test_scenarios = ["delayed_circular_crossing"] # ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "robot_crowding", "delayed_circular_crossing", "hybrid_scenario"]
test_envs = ["sfm", "hsfm"] # ["sfm", "hsfm"]
test_n_humans = [5, 15, 25] # [5, 15, 25]
test_n_trials = 1000

### INITIALIZATION
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "timeouts": {"label": "Timeout Rate", "episodic": False}, 
    "returns": {"label": "Discounted return ($\gamma = 0.9$)", "episodic": True},
    "times_to_goal": {"label": "Time to goal ($s$)", "episodic": True},
    "average_speed": {"label": "Linear speed ($m/s$)", "episodic": True},
    "average_acceleration": {"label": "Linear acceleration ($m/s^2$)", "episodic": True},
    "average_jerk": {"label": "Linear jerk ($m/s^3$)", "episodic": True},
    "average_angular_speed": {"label": "Angular speed ($rad/s$)", "episodic": True},
    "average_angular_acceleration": {"label": "Angular acceleration ($rad/s^2$)", "episodic": True},
    "average_angular_jerk": {"label": "Angular jerk ($rad/s^3$)", "episodic": True},
    "min_distance": {"label": "Minimum distance to humans ($m$)", "episodic": True},
    "space_compliance": {"label": "Space compliance", "episodic": True},
    "episodic_spl": {"label": "Episodic SPL", "episodic": True},
    "path_length": {"label": "Path length ($m$)", "episodic": True},
}
scenarios = {
    "circular_crossing": {"label": "CC"},
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "robot_crowding": {"label": "RC"},
    "delayed_circular_crossing": {"label": "DCC"},
    "hybrid_scenario": {"label": "HS"},
}
envs = {
    "sfm": {"label": "SFM"},
    "hsfm": {"label": "HSFM"},
}
rewards = {
    0: {"label": "R0"},
    1: {"label": "R1"},
    2: {"label": "R2"},
    3: {"label": "R3"},
    4: {"label": "R4"},
    5: {"label": "R5"},
    6: {"label": "R6"},
    7: {"label": "R7"},
}

### LOAD ALL DATA
with open(os.path.join(os.path.dirname(__file__),'results', "metrics_after_il_ablation_study.pkl"), "rb") as f:
    all_metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__),'results', "metrics_after_rl_ablation_study.pkl"), "rb") as f:
    all_metrics_after_rl = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__),'results', "training_data_ablation_study.pkl"), "rb") as f:
    training_data = pickle.load(f)

### PLOTS
# Create figure folder
if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"))
figure_folder = os.path.join(os.path.dirname(__file__), "figures")

# Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
exclude_metrics = ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]
for r_idx in range(2**len(reward_terms)):
    for e_idx, test_env in enumerate(test_envs):
        figure, ax = plt.subplots(math.ceil((len(metrics)-len(exclude_metrics))/3), 3, figsize=(10,10))
        figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
        figure.suptitle(f"Metrics after RL - Test env: {envs[test_env]['label']} - Reward:{rewards[r_idx]['label']}")
        legend_elements = [
            Line2D([0], [0], color="lightblue", lw=4, label="HSFM"),
            Line2D([0], [0], color="lightcoral", lw=4, label="SFM")
        ]
        figure.legend(handles=legend_elements, loc="center right", title="Training\nEnvironment")
        idx = 0
        for key, values in all_metrics_after_rl.items():
            if key in exclude_metrics:
                continue
            else:
                i = idx // 3
                j = idx % 3
                ax[i,j].set(
                    xlabel='N° humans',
                    title=metrics[key]['label'],)
                ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
                ax[i,j].grid()
                # First train env
                unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,0,r_idx,0,e_idx,h_idx,:].flatten())
                data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
                data = data.dropna()
                ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', edgecolor='lightblue', alpha=0.7),
                    tick_labels=test_n_humans,
                    whiskerprops=dict(color='blue', alpha=0.7),
                    capprops=dict(color='blue', alpha=0.7),
                    medianprops=dict(color='blue', alpha=0.7),
                    meanprops=dict(markerfacecolor='blue', markeredgecolor='blue'), 
                    showfliers=False,
                    showmeans=True, 
                    zorder=1)
                # Second train env
                unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,1,r_idx,0,e_idx,h_idx,:].flatten())
                data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
                data = data.dropna()
                ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
                        boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
                        tick_labels=test_n_humans,
                        whiskerprops=dict(color="coral", alpha=0.4),
                        capprops=dict(color="coral", alpha=0.4),
                        medianprops=dict(color="coral", alpha=0.4),
                        meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
                        showfliers=False,
                        showmeans=True,
                    zorder=2)
            idx += 1
        figure.savefig(os.path.join(figure_folder,f"metrics_boxplots_after_rl_train_env_benchmark_{test_env}_R{r_idx}.pdf"), format='pdf')
