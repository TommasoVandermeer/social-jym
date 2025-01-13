import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import math
import jax.numpy as jnp
import pandas as pd

test_n_humans = [5,15,25]
only_base_and_full_rewards = True
n_test_trials = 1000
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
humans_policy = 'hsfm'
train_scenario = 'hybrid_scenario'
train_hybrid_scenario_subset = jnp.array([1,2,3,4], dtype=jnp.int32) # Exclude nnormal circular crossing
test_scenarios = ['parallel_traffic', 'perpendicular_traffic', 'robot_crowding', 'delayed_circular_crossing']
scenarios_labels = ["PaT", "PeT", "RC", "DCC"]

# Load results
with open(os.path.join(os.path.dirname(__file__), "metrics_after_il_ablation_study.pkl"), "rb") as f:
    all_metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "metrics_after_rl_ablation_study.pkl"), "rb") as f:
    all_metrics_after_rl = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "training_data_ablation_study.pkl"), "rb") as f:
    training_data = pickle.load(f)

#### PLOTS ####
## TRAINING DATA ##

# Plot loss curve during IL for each reward
figure, ax = plt.subplots(figsize=(10,10))
ax.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each reward')
ax.grid()
for loss in range(len(training_data["loss_during_il"])):
    if (only_base_and_full_rewards) and (loss > 0 and loss < 2**(len(reward_terms))-1):
        continue
    ax.plot(
        np.arange(len(training_data["loss_during_il"][loss])), 
        training_data["loss_during_il"][loss],
        color = list(mcolors.TABLEAU_COLORS.values())[loss])
if (only_base_and_full_rewards):
    ax.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    ax.legend(["Reward {}".format(i) for i in range(len(training_data["loss_during_il"]))])
figure.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il_ablation_study.eps"), format='eps')

# Plot returns during RL for each reward
figure, ax = plt.subplots(figsize=(10,10))
figure.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85)
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each reward')
ax.grid()
for reward in range(len(training_data["returns_during_rl"])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    ax.plot(
        np.arange(len(training_data["returns_during_rl"][reward])-(window-1))+window, 
        np.convolve(training_data["returns_during_rl"][reward], np.ones(window,), 'valid') / window,
        color = list(mcolors.TABLEAU_COLORS.values())[reward])
if (only_base_and_full_rewards):
    figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    figure.legend(["Reward {}".format(i) for i in range(len(training_data["returns_during_rl"]))], loc="center right")
figure.savefig(os.path.join(os.path.dirname(__file__),"return_curves_during_rl_ablation_study.eps"), format='eps')

# Plot return after IL and RL for each reward
figure, ax = plt.subplots(int(len(training_data['returns_after_il'])/2), 2, figsize=(10,10))
figure.suptitle(f"Return after IL and RL training for each test - All test scenarios - {n_test_trials} trials")
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="After IL"),
    Line2D([0], [0], color="lightcoral", lw=4, label="After RL")
]
figure.legend(handles=legend_elements, loc="center right")
figure.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.05, top=0.90, right=0.87)
for reward in range(len(training_data['returns_after_il'])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    i = reward // 2
    j = reward % 2
    ax[i,j].set(
        xlabel='N° humans', 
        ylabel='Return', 
        title=f'REWARD {reward}',
        ylim=[-0.5,0.5])
    ax[i,j].grid()
    # Aggregate data by test scenario
    unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
    for h_idx in range(len(test_n_humans)):
        unclean_data = unclean_data.at[h_idx].set(training_data['returns_after_il'][reward,:,h_idx,:].flatten())
    # Clean data from NaNs
    data = pd.DataFrame(np.transpose(unclean_data), columns=test_n_humans)
    data = data.dropna()
    ax[i,j].boxplot(data, widths=0.4, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
                tick_labels=test_n_humans,
                whiskerprops=dict(color="blue", alpha=0.7),
                capprops=dict(color="blue", alpha=0.7),
                medianprops=dict(color="blue", alpha=0.7),
                meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
                showfliers=False,
                showmeans=True, 
                zorder=1)
    # Aggregate data by test scenario
    unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
    for h_idx in range(len(test_n_humans)):
        unclean_data = unclean_data.at[h_idx].set(training_data['returns_after_rl'][reward,:,h_idx,:].flatten())
    # Clean data from NaNs
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
figure.savefig(os.path.join(os.path.dirname(__file__),f"return_boxplots_after_il_and_rl_ablation_study.png"), format='png')

# Plot success rate after IL and RL for each reward
figure, ax = plt.subplots(2,1,figsize=(10,10))
figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.90, right=0.85)
ax[0].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after IL training for each test - {n_test_trials} trials', 
    xticks=np.arange(len(test_n_humans)), 
    xticklabels=test_n_humans, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
ax[0].grid()
ax[1].set(
    xlabel='Number of humans', 
    ylabel='Success rate', 
    title=f'Success rate after RL training for each test - {n_test_trials} trials', 
    xticks=np.arange(len(test_n_humans)), 
    xticklabels=test_n_humans, 
    yticks=[i/10 for i in range(11)], 
    ylim=[0,1.1])
ax[1].grid()
for reward in range(len(training_data['success_rate_after_il'])):
    if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
        continue
    ax[0].plot(np.mean(training_data['success_rate_after_il'][reward], axis=0))
    ax[1].plot(np.mean(training_data['success_rate_after_rl'][reward], axis=0))
if (only_base_and_full_rewards):
    figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    figure.legend(["Reward {}".format(i) for i in range(len(training_data["success_rate_after_il"]))], loc="center right")
figure.savefig(os.path.join(os.path.dirname(__file__),f"success_rate_curves_after_il_and_rl_ablation_study.eps"), format='eps')

## TESTING DATA ##
# Plot boxplot of time to goal, path length, angular_speed, space_compliance after RL for base reward and reward with all contributions
figure, ax = plt.subplots(4,len(test_n_humans),figsize=(10,10))
figure.subplots_adjust(hspace=0.7, wspace=0.5, top=0.95, bottom=0.05, left=0.07, right=0.9)
for i, n_humans in enumerate(test_n_humans):
    ax[0, i].grid()
    ax[0, i].set_title(f"{n_humans} humans")
    ax[0, i].set_xlabel("Scenario")
    ax[0, i].set_ylabel("Time to goal (s)")
    ax[1, i].grid()
    ax[1, i].set_title(f"{n_humans} humans")
    ax[1, i].set_xlabel("Scenario")
    ax[1, i].set_ylabel("Path length (m)")
    ax[2, i].grid()
    ax[2, i].set_title(f"{n_humans} humans")
    ax[2, i].set_xlabel("Scenario")
    ax[2, i].set_ylabel("Angular speed (r/s)")
    ax[3, i].grid()
    ax[3, i].set_title(f"{n_humans} humans")
    ax[3, i].set_xlabel("Scenario")
    ax[3, i].set_ylabel("Space compliance")
    for j, scenario in enumerate(test_scenarios):
        # Base reward
        time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][0,j,i])
        time_data = time_data.dropna()
        ax[0, i].boxplot(
            time_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        length_data = pd.DataFrame(all_metrics_after_rl["path_length"][0,j,i])
        length_data = length_data.dropna()
        ax[1, i].boxplot(
            length_data, widths=0.4, patch_artist=True,
            positions = [j], 
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][0,j,i])
        angular_speed_data = angular_speed_data.dropna()
        ax[2, i].boxplot(
            angular_speed_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][0,j,i])
        space_compliance_data = space_compliance_data.dropna()
        ax[3, i].boxplot(
            space_compliance_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        # Full reward
        time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][-1,j,i])
        time_data = time_data.dropna()
        ax[0, i].boxplot(
            time_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
        length_data = pd.DataFrame(all_metrics_after_rl["path_length"][-1,j,i])
        length_data = length_data.dropna()
        ax[1, i].boxplot(
            length_data, widths=0.4, patch_artist=True,
            positions = [j], 
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
        angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][-1,j,i])
        angular_speed_data = angular_speed_data.dropna()
        ax[2, i].boxplot(
            angular_speed_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
        space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][-1,j,i])
        space_compliance_data = space_compliance_data.dropna()
        ax[3, i].boxplot(
            space_compliance_data, widths=0.4, patch_artist=True, 
            positions = [j],
            tick_labels = [scenarios_labels[j]],
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
    ## Plot boxplot aggregating all scenarios
    # Base reward
    time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][0,:,i].flatten())
    time_data = time_data.dropna()
    ax[0, i].boxplot(
        time_data, widths=0.4, patch_artist=True, 
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
        whiskerprops=dict(color="blue", alpha=0.7),
        capprops=dict(color="blue", alpha=0.7),
        medianprops=dict(color="blue", alpha=0.7),
        meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
        showfliers=False,
        showmeans=True)
    length_data = pd.DataFrame(all_metrics_after_rl["path_length"][0,:,i].flatten())
    length_data = length_data.dropna()
    ax[1, i].boxplot(
        length_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
        whiskerprops=dict(color="blue", alpha=0.7),
        capprops=dict(color="blue", alpha=0.7),
        medianprops=dict(color="blue", alpha=0.7),
        meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
        showfliers=False,
        showmeans=True)
    angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][0,:,i].flatten())
    angular_speed_data = angular_speed_data.dropna()
    ax[2, i].boxplot(
        angular_speed_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
        whiskerprops=dict(color="blue", alpha=0.7),
        capprops=dict(color="blue", alpha=0.7),
        medianprops=dict(color="blue", alpha=0.7),
        meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
        showfliers=False,
        showmeans=True)
    space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][0,:,i].flatten())
    space_compliance_data = space_compliance_data.dropna()
    ax[3, i].boxplot(
        space_compliance_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
        whiskerprops=dict(color="blue", alpha=0.7),
        capprops=dict(color="blue", alpha=0.7),
        medianprops=dict(color="blue", alpha=0.7),
        meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
        showfliers=False,
        showmeans=True)
    # Full reward
    time_data = pd.DataFrame(all_metrics_after_rl["times_to_goal"][-1,:,i].flatten())
    time_data = time_data.dropna()
    ax[0, i].boxplot(
        time_data, widths=0.4, patch_artist=True, 
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
        whiskerprops=dict(color="coral", alpha=0.4),
        capprops=dict(color="coral", alpha=0.4),
        medianprops=dict(color="coral", alpha=0.4),
        meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
        showfliers=False,
        showmeans=True)
    length_data = pd.DataFrame(all_metrics_after_rl["path_length"][-1,:,i].flatten())
    length_data = length_data.dropna()
    ax[1, i].boxplot(
        length_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
        whiskerprops=dict(color="coral", alpha=0.4),
        capprops=dict(color="coral", alpha=0.4),
        medianprops=dict(color="coral", alpha=0.4),
        meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
        showfliers=False,
        showmeans=True)
    angular_speed_data = pd.DataFrame(all_metrics_after_rl["average_angular_speed"][-1,:,i].flatten())
    angular_speed_data = angular_speed_data.dropna()
    ax[2, i].boxplot(
        angular_speed_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
        whiskerprops=dict(color="coral", alpha=0.4),
        capprops=dict(color="coral", alpha=0.4),
        medianprops=dict(color="coral", alpha=0.4),
        meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
        showfliers=False,
        showmeans=True)
    space_compliance_data = pd.DataFrame(all_metrics_after_rl["space_compliance"][-1,:,i].flatten())
    space_compliance_data = space_compliance_data.dropna()
    ax[3, i].boxplot(
        space_compliance_data, widths=0.4, patch_artist=True,
        positions = [len(test_scenarios)],
        tick_labels = ["All"],
        boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
        whiskerprops=dict(color="coral", alpha=0.4),
        capprops=dict(color="coral", alpha=0.4),
        medianprops=dict(color="coral", alpha=0.4),
        meanprops=dict(markerfacecolor="red", markeredgecolor="red"),
        showfliers=False,
        showmeans=True)
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="Reward 0"),
    Line2D([0], [0], color="lightcoral", lw=4, label="Reward {}".format(2**len(reward_terms)-1))
]
figure.legend(handles=legend_elements, loc="center right")
# Save figure
figure.savefig(os.path.join(os.path.dirname(__file__),f"some_metrics_boxplots_after_rl_full_and_base_all_scenarios_ablation_study.png"), format='png')


# Plot boxplot of each metric (aggregatedby test scenario) after RL for base reward and reward with all contributions
figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
figure.suptitle(f"Metrics after RL training for each test for base and full rewards - All test scenarios - {n_test_trials} trials")
figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="Reward 0"),
    Line2D([0], [0], color="lightcoral", lw=4, label="Reward {}".format(2**len(reward_terms)-1))
]
figure.legend(handles=legend_elements, loc="center right")
idx = 0
for key, values in all_metrics_after_rl.items():
    if key == "successes" or key == "collisions" or key == "timeouts":
        continue
    else:
        i = idx // 3
        j = idx % 3
        ax[i,j].set(
            xlabel='N° humans', 
            ylabel=key)
        ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
        ax[i,j].grid()
        # Base reward
        unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
        for h_idx in range(len(test_n_humans)):
            unclean_data = unclean_data.at[h_idx].set(values[0,:,h_idx,:].flatten())
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
        # Full reward
        unclean_data = jnp.zeros((len(test_n_humans),n_test_trials*len(test_scenarios)))
        for h_idx in range(len(test_n_humans)):
            unclean_data = unclean_data.at[h_idx].set(values[-1,:,h_idx,:].flatten())
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
figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_boxplots_after_rl_full_and_base_ablation_study.png"), format='png')

# Plot curves of each metric (aggregated by test scenario) after RL for each reward
figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
figure.suptitle(f"Metrics after RL training for each test - All test scenarios - {n_test_trials} trials")
figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
idx = 0
for key, values in all_metrics_after_rl.items():
    if key == "successes" or key == "collisions" or key == "timeouts":
        continue
    else:
        i = idx // 3
        j = idx % 3
        ax[i,j].set(
            xlabel='N° humans', 
            ylabel=key)
        ax[i,j].set_xticks(test_n_humans, labels=test_n_humans)
        ax[i,j].grid()
        for reward in range(len(values)):
            if (only_base_and_full_rewards) and (reward > 0 and reward < 2**(len(reward_terms))-1):
                continue
            ax[i,j].plot(test_n_humans, np.nanmean(values[reward], axis=(0,2)), color=list(mcolors.TABLEAU_COLORS.values())[reward])
        idx += 1
if (only_base_and_full_rewards):
    figure.legend(["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]], loc="center right")
else:
    figure.legend(["Reward {}".format(i) for i in range(len(all_metrics_after_rl["times_to_goal"]))], loc="center right")
figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_after_rl_ablation_study.eps"), format='eps')

# Plot boxplots side to side for each metric (aggregated by test scenario) after RL for tests with all rewards (one figure for each n_humans test)
for test, n_humans in enumerate(test_n_humans):
    figure, ax = plt.subplots(math.ceil((len(all_metrics_after_rl)-4)/3), 3, figsize=(10,10))
    figure.suptitle(f"Metrics after RL training for each test for base and full rewards - {n_humans} humans - {n_test_trials} trials")
    figure.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.05, top=0.90, left=0.1, right=0.87)
    idx = 0
    for key, values in all_metrics_after_rl.items():
        if key == "successes" or key == "collisions" or key == "timeouts":
            continue
        else:
            i = idx // 3
            j = idx % 3
            ax[i,j].set(
                xlabel='Reward', 
                ylabel=key)
            ax[i,j].grid()
            # All rewards
            if only_base_and_full_rewards:
                unclean_data = jnp.zeros((2,n_test_trials*len(test_scenarios)))
                for r, reward in enumerate([0, 2**len(reward_terms)-1]):
                    unclean_data = unclean_data.at[r].set(values[reward,:,test].flatten())
                data = pd.DataFrame(np.transpose(unclean_data), columns=["Reward {}".format(i) for i in [0, 2**len(reward_terms)-1]])
            else:
                unclean_data = jnp.zeros((len(values),n_test_trials*len(test_scenarios)))
                for reward in range(len(values)):
                    unclean_data = unclean_data.at[reward].set(values[reward,:,test].flatten())
                data = pd.DataFrame(np.transpose(unclean_data), columns=["Reward {}".format(i) for i in range(len(values))])
            data = data.dropna()
            bplots = ax[i,j].boxplot(
                data, 
                widths=0.4, 
                patch_artist=True,
                tick_labels=np.arange(len(values)) if not only_base_and_full_rewards else [0, 2**len(reward_terms)-1],
                showfliers=False,
                showmeans=True, 
                zorder=1)
            for patch, color in zip(bplots['boxes'], list(mcolors.TABLEAU_COLORS.values())):
                patch.set_facecolor(color)
            idx += 1
    legend_elements = []
    for i in range(len(values)):
        if only_base_and_full_rewards and (i > 0 and i < 2**(len(reward_terms))-1):
            continue
        legend_elements.append(Line2D([0], [0], color=list(mcolors.TABLEAU_COLORS.values())[i], lw=4, label=f"Reward {i}"))
    figure.legend(handles=legend_elements, loc="center right")
    figure.savefig(os.path.join(os.path.dirname(__file__),f"metrics_boxplots_after_rl_{n_humans}humans_ablation_study.eps"), format='eps')
