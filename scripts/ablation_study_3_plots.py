import os
import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
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
    "average_speed": {"label": "Lin. speed ($m/s$)", "episodic": True},
    "average_acceleration": {"label": "Lin. accel. ($m/s^2$)", "episodic": True},
    "average_jerk": {"label": "Lin. jerk ($m/s^3$)", "episodic": True},
    "average_angular_speed": {"label": "Ang. speed ($r/s$)", "episodic": True},
    "average_angular_acceleration": {"label": "Ang. accel. ($r/s^2$)", "episodic": True},
    "average_angular_jerk": {"label": "Ang. jerk ($r/s^3$)", "episodic": True},
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

# Matplotlib font
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 17}
rc('font', **font)

# # Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
# exclude_metrics = ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]
# for r_idx in range(2**len(reward_terms)):
#     for e_idx, test_env in enumerate(test_envs):
#         figure, ax = plt.subplots(math.ceil((len(metrics)-len(exclude_metrics))/3), 3, figsize=(10,10))
#         figure.subplots_adjust(hspace=0.7, wspace=0.5, bottom=0.08, top=0.93, left=0.07, right=0.83)
#         figure.suptitle(f"Metrics after RL - Test env: {envs[test_env]['label']} - Reward:{rewards[r_idx]['label']}", fontsize=10)
#         legend_elements = [
#             Line2D([0], [0], color="lightblue", lw=4, label=train_envs[0].upper()),
#             Line2D([0], [0], color="lightcoral", lw=4, label=train_envs[1].upper())
#         ]
#         figure.legend(handles=legend_elements, loc="center right", title="Train.\nEnv.")
#         idx = 0
#         for key, values in all_metrics_after_rl.items():
#             if key in exclude_metrics:
#                 continue
#             else:
#                 i = idx // 3
#                 j = idx % 3
#                 ax[i,j].set(
#                     xlabel='N° humans',
#                     title=metrics[key]['label'],)
#                 ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
#                 ax[i,j].grid()
#                 # First train env
#                 unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)))
#                 for h_idx in range(len(test_n_humans)):
#                     unclean_data = unclean_data.at[h_idx].set(values[0,0,r_idx,0,e_idx,h_idx,:].flatten())
#                 data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
#                 data = data.dropna()
#                 ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
#                     boxprops=dict(facecolor='lightblue', edgecolor='lightblue', alpha=0.7),
#                     tick_labels=test_n_humans,
#                     whiskerprops=dict(color='blue', alpha=0.7),
#                     capprops=dict(color='blue', alpha=0.7),
#                     medianprops=dict(color='blue', alpha=0.7),
#                     meanprops=dict(markerfacecolor='blue', markeredgecolor='blue'), 
#                     showfliers=False,
#                     showmeans=True, 
#                     zorder=1)
#                 # Second train env
#                 unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)))
#                 for h_idx in range(len(test_n_humans)):
#                     unclean_data = unclean_data.at[h_idx].set(values[0,1,r_idx,0,e_idx,h_idx,:].flatten())
#                 data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
#                 data = data.dropna()
#                 ax[i,j].boxplot(data, widths=0.3, patch_artist=True, 
#                         boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
#                         tick_labels=test_n_humans,
#                         whiskerprops=dict(color="coral", alpha=0.4),
#                         capprops=dict(color="coral", alpha=0.4),
#                         medianprops=dict(color="coral", alpha=0.4),
#                         meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
#                         showfliers=False,
#                         showmeans=True,
#                     zorder=2)
#             idx += 1
#         figure.savefig(os.path.join(figure_folder,f"metrics_boxplots_after_rl_train_env_benchmark_{test_env}_R{r_idx}.pdf"), format='pdf')

# Plot barplot of TtG and Ang.Accel. after RL for base reward trained and tested in different environments (HSFM and SFM)
metrics_to_plot = ["times_to_goal", "average_angular_acceleration"]
colors = ["lightskyblue", "blue", "lightcoral", "red"]
figure, ax = plt.subplots(1, 2, figsize=(10,5))
figure.subplots_adjust(hspace=0.7, wspace=0.2, bottom=0.13, top=0.91, left=0.05, right=0.77)
for m_idx, metric in enumerate(metrics_to_plot):        
    bar_width = 0.2
    p0 = np.arange(len(test_n_humans))
    ax[m_idx].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[m_idx].set_xticks(p0 + (3 / 2) * bar_width, labels=test_n_humans)
    idx = 0
    for te_idx, train_env in enumerate(train_envs):
        for e_idx, test_env in enumerate(test_envs):
            p = p0 + idx * bar_width
            bars = jnp.nanmean(all_metrics_after_rl[metric][0,te_idx,0,0,e_idx, :, :], axis=1)
            color = colors[int(2 * te_idx + e_idx)]
            ax[m_idx].bar(
                p, 
                bars, 
                width=bar_width, 
                color=color, 
                label=f"{envs[train_env]['label']}-{envs[test_env]['label']}", 
                edgecolor='white', 
                zorder=3
            )
            idx += 1
    if m_idx == 0:
        figure.legend(loc="center right", title="Train-test\nenvironments", fontsize=15)
    ax[m_idx].grid(zorder=0)
figure.savefig(os.path.join(figure_folder,f"1.pdf"), format='pdf')

# Plot boxplot of each metric for base reward  in all train envs
exclude_metrics = ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]
figure, ax = plt.subplots(math.ceil((len(metrics)-len(exclude_metrics))/3), 3, figsize=(10,10))
figure.subplots_adjust(hspace=0.7, wspace=0.3, bottom=0.08, top=0.93, left=0.07, right=0.83)
colors = {
    "sfm": "blue",
    "hsfm": "red",
}
legend_elements = [
    Line2D([0], [0], color=colors[train_envs[0]], lw=4, label=train_envs[0].upper()),
    Line2D([0], [0], color=colors[train_envs[1]], lw=4, label=train_envs[1].upper())
]
figure.legend(handles=legend_elements, loc="center right", title="Training\nenv.")
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
        p0 = np.arange(len(test_n_humans))
        box_width = 0.3
        ax[i,j].grid(zorder=0)
        # Train env
        for te_idx, train_env in enumerate(train_envs):
            unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)*len(test_envs)))
            for h_idx in range(len(test_n_humans)):
                unclean_data = unclean_data.at[h_idx].set(values[0,te_idx,0,0,:,h_idx,:].flatten())
            data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
            data = data.dropna()
            ax[i,j].boxplot(
                data, 
                widths=box_width,
                positions=p0 + te_idx * box_width, 
                patch_artist=True, 
                boxprops=dict(facecolor=colors[train_env], edgecolor=colors[train_env]),
                meanprops=dict(markerfacecolor="white", markeredgecolor="white"), 
                medianprops=dict(color='black'),
                showfliers=False,
                showmeans=True, 
                zorder=3,
            )
        ax[i,j].set_xticks(p0 + box_width * 0.5, labels=test_n_humans)
    idx += 1
figure.savefig(os.path.join(figure_folder,f"2.pdf"), format='pdf')

# Plot barplot of TtG and Ang.Accel. after RL for base reward trained in different environments (HSFM and SFM) with different reward (R0 and R7)
metrics_to_plot = ["times_to_goal", "average_angular_acceleration"]
colors = ["deeppink", "green", "chocolate", "grey"]
figure, ax = plt.subplots(1, 2, figsize=(10,5))
figure.subplots_adjust(hspace=0.7, wspace=0.2, bottom=0.13, top=0.91, left=0.05, right=0.77)
for m_idx, metric in enumerate(metrics_to_plot):        
    bar_width = 0.2
    p0 = np.arange(len(test_n_humans))
    ax[m_idx].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[m_idx].set_xticks(p0 + (3 / 2) * bar_width, labels=test_n_humans)
    idx = 0
    for te_idx, train_env in enumerate(train_envs):
        for r_idx, reward in enumerate([0,7]):
            p = p0 + idx * bar_width
            bars = jnp.nanmean(all_metrics_after_rl[metric][0,te_idx,reward,0,:, :, :], axis=(0,2))
            color = colors[int(2 * te_idx + r_idx)]
            ax[m_idx].bar(
                p, 
                bars, 
                width=bar_width, 
                color=color, 
                label=f"{envs[train_env]['label']}-{rewards[reward]['label']}", 
                edgecolor='white', 
                zorder=3
            )
            idx += 1
    if m_idx == 0:
        figure.legend(loc="center right", title="Train env.\n-\nReward", fontsize=15)
    ax[m_idx].grid(zorder=0)
figure.savefig(os.path.join(figure_folder,f"3.pdf"), format='pdf')

# Plot boxplot of each metric for base and full rewards  in all train envs
exclude_metrics = ["successes", "collisions", "timeouts", "episodic_spl", "returns", "min_distance"]
colors = {
        "sfm": {
            "R0": "deeppink",
            "R7": "green",
        },
        "hsfm": {
            "R0": "chocolate",
            "R7": "grey",
        },
    }
for te_idx, train_env in enumerate(train_envs):
    figure, ax = plt.subplots(math.ceil((len(metrics)-len(exclude_metrics))/3), 3, figsize=(10,10))
    figure.subplots_adjust(hspace=0.7, wspace=0.3, bottom=0.08, top=0.93, left=0.07, right=0.83)
    legend_elements = [
        Line2D([0], [0], color=colors[train_env]["R0"], lw=4, label="R0"),
        Line2D([0], [0], color=colors[train_env]["R7"], lw=4, label="R7")
    ]
    figure.legend(handles=legend_elements, loc="center right", title=f"Train env.\n{train_env.upper()}\n\nReward")
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
            p0 = np.arange(len(test_n_humans))
            box_width = 0.3
            ax[i,j].grid(zorder=0)
            # Train env
            for r_idx, reward in enumerate([0,7]):
                unclean_data = jnp.zeros((len(test_n_humans),test_n_trials*len(test_scenarios)*len(test_envs)))
                for h_idx in range(len(test_n_humans)):
                    unclean_data = unclean_data.at[h_idx].set(values[0,te_idx,reward,0,:,h_idx,:].flatten())
                data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
                data = data.dropna()
                ax[i,j].boxplot(
                    data, 
                    widths=box_width,
                    positions=p0 + r_idx * box_width, 
                    patch_artist=True, 
                    boxprops=dict(facecolor=colors[train_env]["R"+str(reward)], edgecolor=colors[train_env]["R"+str(reward)]),
                    meanprops=dict(markerfacecolor="white", markeredgecolor="white"), 
                    medianprops=dict(color='black'),
                    showfliers=False,
                    showmeans=True, 
                    zorder=3,
                )
            ax[i,j].set_xticks(p0 + box_width * 0.5, labels=test_n_humans)
        idx += 1
    figure.savefig(os.path.join(figure_folder,f"{4 + te_idx}.pdf"), format='pdf')

# Plot boxplot of each metric for all rewards in all train envs
colors = [list(mcolors.TABLEAU_COLORS.values())[:len(rewards)] for _ in range(len(train_envs))]
colors[0][0] = "deeppink"
colors[0][7] = "green"
colors[1][0] = "chocolate"
colors[1][7] = "grey"
metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
for te_idx, train_env in enumerate(train_envs):
    legend_elements = [Line2D([0], [0], color=colors[te_idx][r], lw=4, label=rewards[r]["label"]) for r in range(len(rewards))]
    figure, ax = plt.subplots(2, 2, figsize=(10,10))
    figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.93, left=0.07, right=0.87)
    figure.legend(handles=legend_elements, loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nN° Hum.\n{test_n_humans[1]}\n\nReward", fontsize=15)
    for m_idx, metric in enumerate(metrics_to_plot):
        values = all_metrics_after_rl[metric]
        i = m_idx // 2
        j = m_idx % 2
        ax[i,j].set(
            xlabel='Reward function',
            title=metrics[metric]['label'],)
        box_width = 0.4
        ax[i,j].grid(zorder=0)
        for r_idx in range(len(rewards)):
            unclean_data = values[0,te_idx,r_idx,0,:,1,:].flatten()
            data = pd.DataFrame(np.transpose(unclean_data), columns=[r_idx])
            data = data.dropna()
            ax[i,j].boxplot(
                data, 
                widths=box_width,
                positions=[r_idx],
                boxprops=dict(facecolor=colors[te_idx][r_idx], edgecolor=colors[te_idx][r_idx]),
                meanprops=dict(markerfacecolor="white", markeredgecolor="white", markersize=5), 
                medianprops=dict(color='black'),
                patch_artist=True, 
                showfliers=False,
                showmeans=True, 
                zorder=3,
            )
        ax[i,j].set_xticks(np.arange(len(rewards)), labels=[rewards[r]["label"] for r in range(len(rewards))])
    figure.savefig(os.path.join(figure_folder,f"{6 + te_idx}.pdf"), format='pdf')


# # Plot boxplot of each metric for all rewards in all train envs
# colors = list(mcolors.TABLEAU_COLORS.values())[:len(rewards)]
# metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
# legend_elements = [Line2D([0], [0], color=colors[r], lw=4, label=rewards[r]["label"]) for r in range(len(rewards))]
# for te_idx, train_env in enumerate(train_envs):
#     figure, ax = plt.subplots(2, 2, figsize=(10,10))
#     figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.93, left=0.06, right=0.87)
#     figure.legend(handles=legend_elements, loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nReward", fontsize=15)
#     for m_idx, metric in enumerate(metrics_to_plot):
#         values = all_metrics_after_rl[metric]
#         i = m_idx // 2
#         j = m_idx % 2
#         ax[i,j].set(
#             xlabel='N° humans',
#             title=metrics[metric]['label'],)
#         box_width = 0.22
#         ax[i,j].grid(zorder=0)
#         for h_idx in range(len(test_n_humans)):
#             for r_idx in range(len(rewards)):
#                 unclean_data = values[0,te_idx,r_idx,0,:,h_idx,:].flatten()
#                 data = pd.DataFrame(np.transpose(unclean_data), columns=[r_idx])
#                 data = data.dropna()
#                 ax[i,j].boxplot(
#                     data, 
#                     widths=box_width,
#                     positions=[2 * h_idx + r_idx * box_width],
#                     boxprops=dict(facecolor=colors[r_idx], edgecolor=colors[r_idx]),
#                     meanprops=dict(markerfacecolor="white", markeredgecolor="white", markersize=5), 
#                     medianprops=dict(color='black', linewidth=0),
#                     patch_artist=True, 
#                     showfliers=False,
#                     showmeans=True, 
#                     whis=0,
#                     showcaps=False,
#                     zorder=3,
#                 )
#         ax[i,j].set_xticks(2 * np.arange(len(test_n_humans)) + (7 / 2) * box_width, labels=test_n_humans)
#     figure.savefig(os.path.join(figure_folder,f"{6 + te_idx}.pdf"), format='pdf')

# # Plot barplots of each metric for all rewards in all train envs
# colors = list(mcolors.TABLEAU_COLORS.values())[:len(rewards)]
# metrics_to_plot = ["times_to_goal", "path_length", "average_angular_acceleration", "space_compliance"]
# for te_idx, train_env in enumerate(train_envs):
#     figure, ax = plt.subplots(2, 2, figsize=(10,10))
#     figure.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.08, top=0.95, left=0.06, right=0.87)
#     for m_idx, metric in enumerate(metrics_to_plot):
#         values = all_metrics_after_rl[metric]
#         i = m_idx // 2
#         j = m_idx % 2
#         ax[i,j].set(
#             xlabel='N° humans',
#             title=metrics[metric]['label'],)
#         box_width = 0.22
#         ax[i,j].grid(zorder=0)
#         for h_idx in range(len(test_n_humans)):
#             for r_idx in range(len(rewards)):
#                 bar = jnp.nanmean(values[0,te_idx,r_idx,0,:,h_idx,:], axis=(0,1))
#                 ax[i,j].bar(
#                     2 * h_idx + r_idx * box_width, 
#                     bar,
#                     width=box_width,
#                     color=colors[r_idx], 
#                     label=f"R{r_idx}", 
#                     edgecolor='white', 
#                     zorder=3,
#                 )
#             if (m_idx == 0) and (h_idx == 0):
#                 figure.legend(loc="center right", title=f"Train\nenv.\n{train_env.upper()}\n\nReward", fontsize=15)
#         ax[i,j].set_xticks(2 * np.arange(len(test_n_humans)) + (7 / 2) * box_width, labels=test_n_humans)
#     figure.savefig(os.path.join(figure_folder,f"barplots_rewards_benchmark_{train_env}.pdf"), format='pdf')