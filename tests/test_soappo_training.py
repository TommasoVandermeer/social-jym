import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import random
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
from socialjym.utils.rollouts.act_cri_rollouts import actor_critic_il_rollout
from socialjym.utils.rollouts.ppo_rollouts import ppo_rl_rollout
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer
from socialjym.policies.soappo import SOAPPO

### Hyperparameters
n_humans_for_tests = [5, 10, 15, 20, 25]
test_robot_visibility = [False, True]
n_trials = 100
n_parallel_envs = 50 
training_updates = 10_000
rl_debugging_interval = 10
robot_vmax = 1
training_hyperparams = {
    'random_seed': 0,
    'policy_name': 'sarl-ppo',
    'n_humans': 5, 
    'il_buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'il_training_episodes': 2_000,
    'il_actor_learning_rate': 0.001,
    'il_critic_learning_rate': 0.01,
    'il_num_epochs': 50, # Number of epochs to train the model after ending IL
    'il_batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
    'rl_training_updates': training_updates,
    'rl_parallel_envs': n_parallel_envs,
    'rl_actor_learning_rate': 3e-5, # 3e-5
    'rl_critic_learning_rate': 3e-4, # 3e-4
    'rl_buffer_capacity': 3_000, # Number of experiences to sample from the replay buffer for each model update
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 10, # 10
    'rl_num_batches': 30, # 30
    'rl_beta_entropy': 5e-4, # 5e-4
    'lambda_gae': 0.95, # 0.95
    'humans_policy': 'hsfm',
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4], jnp.int32), # Subset of the hybrid scenarios to use for training
    'reward_function': 'socialnav_reward2',
    'custom_episodes': False, # If True, the episodes are loaded from a predefined set
    'gradient_norm_scale': 0.5, # Scale the gradient norm by this value
}

# Initialize reward function
if training_hyperparams['reward_function'] == 'socialnav_reward1': 
    reward_function = Reward1(
        kinematics='unicycle',
    )
elif training_hyperparams['reward_function'] == 'socialnav_reward2':
    reward_function = Reward2(
        v_max = robot_vmax,
        progress_to_goal_reward = True,
        progress_to_goal_weight = 0.03,
        high_rotation_penalty_reward=True,
        angular_speed_bound=1.,
        angular_speed_penalty_weight=0.0075,
    )
else:
    raise ValueError(f"{training_hyperparams['reward_function']} is not a valid reward function")
# Environment parameters
env_params = {
    'robot_radius': 0.3,
    'n_humans': training_hyperparams['n_humans'],
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': training_hyperparams['scenario'],
    'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
    'humans_policy': training_hyperparams['humans_policy'],
    'circle_radius': 7,
    'reward_function': reward_function,
    'kinematics': 'unicycle',
}
# Initialize environment
env = SocialNav(**env_params)
_, _, obs, info, _ = env.reset(random.PRNGKey(training_hyperparams['random_seed']))
# Initialize robot policy and vnet params
policy = SOAPPO(
    env.reward_function, 
    v_max=robot_vmax, 
    dt=env_params['robot_dt'], 
)
initial_actor_params, initial_critic_params = policy.init_nns(
    training_hyperparams['random_seed'],
    obs,
    info,
)
# Initialize replay buffer
replay_buffer = BaseACBuffer(training_hyperparams['il_buffer_size'], training_hyperparams['il_batch_size'])
# Initialize IL optimizer
actor_optimizer = optax.sgd(learning_rate=training_hyperparams['il_actor_learning_rate'], momentum=0.9)
critic_optimizer = optax.sgd(learning_rate=training_hyperparams['il_critic_learning_rate'], momentum=0.9)
# Initialize buffer state
buffer_state = {
    'inputs': jnp.empty((training_hyperparams['il_buffer_size'], env.n_humans, policy.vnet_input_size)),
    'critic_targets': jnp.empty((training_hyperparams['il_buffer_size'],)),
    'sample_actions': jnp.empty((training_hyperparams['il_buffer_size'], 2)),
}
# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    il_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/il_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    il_custom_episodes_path = None
# Initialize IL rollout params
il_rollout_params = {
    'initial_actor_params': initial_actor_params,
    'initial_critic_params': initial_critic_params,
    'train_episodes': training_hyperparams['il_training_episodes'],
    'random_seed': training_hyperparams['random_seed'],
    'actor_optimizer': actor_optimizer,
    'critic_optimizer': critic_optimizer,
    'buffer_state': buffer_state,
    'current_buffer_size': 0,
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'buffer_capacity': training_hyperparams['il_buffer_size'],
    'num_epochs': training_hyperparams['il_num_epochs'],
    'batch_size': training_hyperparams['il_batch_size'],
    'custom_episodes': il_custom_episodes_path
}

# IMITATION LEARNING ROLLOUT
il_out = actor_critic_il_rollout(**il_rollout_params)

## Execute tests to evaluate return after IL
# Initialize the dictionary to store the metrics
outcomes_arrays = jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests),))
other_metrics_arrays = jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests), n_trials))
metrics_after_il = {
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
    "scenario": jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests), n_trials), dtype=jnp.int32),
}
for v, visibility in enumerate(test_robot_visibility):
    print(f"\n##############\nROBOT {'VISIBLE' if visibility else 'NOT VISIBLE'}")
    for test, n_humans in enumerate(n_humans_for_tests):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': visibility,
            'scenario': training_hyperparams['scenario'],
            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
            'humans_policy': training_hyperparams['humans_policy'],
            'circle_radius': 7,
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        test_env = SocialNav(**test_env_params)
        trial_out = test_k_trials(
            n_trials, 
            training_hyperparams['il_training_episodes'], 
            test_env, 
            policy, 
            il_out['actor_params'], 
            reward_function.time_limit
        )
        # Store trail metrics
        metrics_after_il = tree_map(lambda x, y: x.at[v,test].set(y), metrics_after_il, trial_out)
# Save metrics
with open(os.path.join(os.path.dirname(__file__),"metrics_after_il.pkl"), 'wb') as f:
    pickle.dump(metrics_after_il, f)
### Plot losses during IL
figure, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Actor Loss during IL training'
)
ax[0].plot(
    jnp.arange(len(il_out['actor_losses'])), 
    il_out['actor_losses'],
)
ax[1].set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Critic Loss during IL training'
)
ax[1].plot(
    jnp.arange(len(il_out['critic_losses'])), 
    il_out['critic_losses'],
)
figure.savefig(os.path.join(os.path.dirname(__file__),"loss_curves_during_il.eps"), format='eps')
plt.close(figure)
## Save IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'wb') as f:
    pickle.dump(il_out, f)

# Load IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'rb') as f:
    il_out = pickle.load(f)
il_actor_params = il_out['actor_params']
il_critic_params = il_out['critic_params']  

# Initialize RL optimizer
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
    optax.adam(
        learning_rate=optax.schedules.linear_schedule(
            init_value=training_hyperparams['rl_actor_learning_rate'], 
            end_value=0., 
            transition_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
            transition_begin=0
        ), 
        eps=1e-7, 
        b1=0.9,
    ),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
    optax.sgd(
        learning_rate=training_hyperparams['rl_critic_learning_rate'], 
        momentum=0.9
    ),
)

# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    rl_custom_episodes_path = None

# Initialize RL replay buffer
replay_buffer = PPOBuffer(training_hyperparams['rl_buffer_capacity'], int(training_hyperparams['rl_buffer_capacity']/training_hyperparams['rl_num_batches']))
# Initialize RL buffer state
buffer_state = {
    'inputs': jnp.empty((training_hyperparams['rl_buffer_capacity'], env.n_humans, policy.vnet_input_size)),
    'critic_targets': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
    'sample_actions': jnp.empty((training_hyperparams['rl_buffer_capacity'], 2)),
    'old_values': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
    'old_neglogpdfs': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
}

# Initialize RL rollout params
rl_rollout_params = {
    'initial_actor_params': il_actor_params,
    'initial_critic_params': il_critic_params,
    'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
    'train_updates': training_hyperparams['rl_training_updates'],
    'random_seed': training_hyperparams['random_seed'] + training_hyperparams['il_training_episodes'],
    'actor_optimizer': actor_optimizer,
    'critic_optimizer': critic_optimizer,
    'buffer_state': buffer_state,
    'buffer_capacity': training_hyperparams['rl_buffer_capacity'],
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'clip_range': training_hyperparams['rl_clip_frac'],
    'n_epochs': training_hyperparams['rl_num_epochs'],
    'beta_entropy': training_hyperparams['rl_beta_entropy'],
    'lambda_gae': training_hyperparams['lambda_gae'],
    'debugging': True,
    'debugging_interval': rl_debugging_interval,
}

# REINFORCEMENT LEARNING ROLLOUT
rl_out = ppo_rl_rollout(**rl_rollout_params)

# Save RL rollout output
with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'wb') as f:
    pickle.dump(rl_out, f)

# Load RL rollout output
with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'rb') as f:
    rl_out = pickle.load(f)
    print(f"Total episodes simulated: {jnp.sum(rl_out['aux_data']['episodes'])}")

# Save the training returns
rl_actor_params = rl_out['actor_params']
returns_during_rl = rl_out['aux_data']['returns']  
actor_losses = rl_out['aux_data']['actor_losses']
critic_losses = rl_out['aux_data']['critic_losses']
entropy_losses = rl_out['aux_data']['entropy_losses']
success_during_rl = rl_out['aux_data']['successes']
failure_during_rl = rl_out['aux_data']['failures']
timeout_during_rl = rl_out['aux_data']['timeouts']
episodes_during_rl = rl_out['aux_data']['episodes']
stds_during_rl = rl_out['aux_data']['stds']
episode_count = jnp.sum(episodes_during_rl)
window = 500 if training_updates > 1000 else 50

## Plot RL training stats
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 18}
rc('font', **font)
figure, ax = plt.subplots(4,2,figsize=(15,15))
figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95, right=0.95, left=0.1, wspace=0.35)
# Plot returns during RL
ax[0,0].grid()
ax[0,0].set(
    xlabel='Training Update', 
    ylabel=f'Return ({window} upd. window)', 
    title='Return'
)
ax[0,0].plot(
    jnp.arange(len(returns_during_rl)-(window-1))+window, 
    jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
)
# Plot success, failure, and timeout rates during RL
success_rate_during_rl = success_during_rl / rl_out['aux_data']['episodes']
failure_rate_during_rl = failure_during_rl / rl_out['aux_data']['episodes']
timeout_rate_during_rl = timeout_during_rl / rl_out['aux_data']['episodes']
ax[0,1].grid()
ax[0,1].set(
    xlabel='Training Update', 
    ylabel=f'Rate ({window} upd. window)', 
    title='Success, Failure, and Timeout rates',
    ylim=(-0.1,1.1)
)
ax[0,1].plot(
    jnp.arange(len(success_rate_during_rl)-(window-1))+window, 
    jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Success rate',
    color='g',
)
ax[0,1].plot(
    jnp.arange(len(failure_rate_during_rl)-(window-1))+window, 
    jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Failure rate',
    color='r',
)
ax[0,1].plot(
    jnp.arange(len(timeout_rate_during_rl)-(window-1))+window, 
    jnp.convolve(timeout_rate_during_rl, jnp.ones(window,), 'valid') / window,
    label='Timeout rate',
    color='yellow',
)
ax[0,1].legend()
# Plot actor loss during RL
ax[1,0].grid()
ax[1,0].set(
    xlabel='Training Update', 
    ylabel=f'Loss ({window} upd. window)', 
    title='Actor Loss'
)
ax[1,0].plot(
    jnp.arange(len(actor_losses)-(window-1))+window, 
    jnp.convolve(actor_losses, jnp.ones(window,), 'valid') / window,
)
# Plot critic loss during RL
ax[1,1].grid()
ax[1,1].set(
    xlabel='Training Update', 
    ylabel=f'Loss ({window} upd. window)', 
    title='Critic Loss'
)
ax[1,1].plot(
    jnp.arange(len(critic_losses)-(window-1))+window, 
    jnp.convolve(critic_losses, jnp.ones(window,), 'valid') / window,
)
# Plot entropy loss during RL
entropy_window = window // 10
ax[2,0].grid()
ax[2,0].set(
    xlabel='Training Update', 
    ylabel=f'Loss ({entropy_window} upd. window)', 
    title='Entropy Loss'
)
ax[2,0].plot(
    jnp.arange(len(entropy_losses)-(entropy_window-1))+entropy_window, 
    jnp.convolve(entropy_losses, jnp.ones(entropy_window,), 'valid') / entropy_window,
)
# Plot episodes during RL
ax[2,1].grid()
ax[2,1].set(
    xlabel='Training Update', 
    ylabel=f'Episodes', 
    title='Simulated episodes'
)
ax[2,1].plot(
    jnp.arange(len(episodes_during_rl)), 
    jnp.cumsum(episodes_during_rl),
)
# Plot stds[0] during RL
ax[3,0].grid()
ax[3,0].set(
    xlabel='Training Update',
    ylabel='Standard deviation',
    title='First action std',
    ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
)
ax[3,0].plot(
    jnp.arange(len(stds_during_rl)),
    stds_during_rl[:,0],
)
# Plot stds[1] during RL
ax[3,1].grid()
ax[3,1].set(
    xlabel='Training Update',
    ylabel='Standard deviation',
    title='Second action std',
    ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
)
ax[3,1].plot(
    jnp.arange(len(stds_during_rl)),
    stds_during_rl[:,1],
)
figure.savefig(os.path.join(os.path.dirname(__file__),"rl_training_plots.eps"), format='eps')

## Execute tests to evaluate metrics after RL 
# Initialize the dictionary to store the metrics
outcomes_arrays = jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests),))
other_metrics_arrays = jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests), n_trials))
metrics_after_rl = {
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
    "scenario": jnp.zeros((len(test_robot_visibility), len(n_humans_for_tests), n_trials), dtype=jnp.int32),
}
for v, visibility in enumerate(test_robot_visibility):
    print(f"\n##############\nROBOT {'VISIBLE' if visibility else 'NOT VISIBLE'}")
    for test, n_humans in enumerate(n_humans_for_tests):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': visibility,
            'scenario': training_hyperparams['scenario'],
            'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
            'humans_policy': training_hyperparams['humans_policy'],
            'circle_radius': 7,
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        test_env = SocialNav(**test_env_params)
        trial_out = test_k_trials(
            n_trials, 
            training_hyperparams['il_training_episodes'] + episode_count, 
            test_env, 
            policy, 
            rl_actor_params, 
            reward_function.time_limit
        )
        # Store trial metrics
        metrics_after_rl = tree_map(lambda x, y: x.at[v,test].set(y), metrics_after_rl, trial_out)
# Save metrics
with open(os.path.join(os.path.dirname(__file__),"metrics_after_rl.pkl"), 'wb') as f:
    pickle.dump(metrics_after_rl, f)

### Plot metrics after RL and after IL
# Load metrics files
with open(os.path.join(os.path.dirname(__file__),"metrics_after_il.pkl"), 'rb') as f:
    metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__),"metrics_after_rl.pkl"), 'rb') as f:
    metrics_after_rl = pickle.load(f)
all_metrics = [metrics_after_il, metrics_after_rl]
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 18}
rc('font', **font)
# Plot all metrics for RL and IL with robot visible and not visible
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
    for v, visibility in enumerate(test_robot_visibility):
        for p, metrics in enumerate(all_metrics):
            ax[metrics_data[key]["row_position"], metrics_data[key]["col_position"]].plot(
                jnp.arange(len(n_humans_for_tests)), 
                jnp.nanmean(metrics[key][v], axis=1) if key != "successes" else metrics[key][v] / n_trials,
                color=list(mcolors.TABLEAU_COLORS.values())[p],
                linewidth=2,
                linestyle='--' if v == 0 else '-',
                label=f"{'IL' if p == 0 else 'RL'} {'visible' if visibility else 'not visible'}",
            )
handles, labels = ax[0,0].get_legend_handles_labels()
figure.legend(labels, loc="center right", title=f"Policy:", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
figure.savefig(os.path.join(os.path.dirname(__file__),f"final_tests_plots.eps"), format='eps')
# Plot success rate after RL and after IL with robot visible and not visible
from socialjym.envs.base_env import SCENARIOS
scenarios_data = {
    "circular_crossing": {"label": "CC"},
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "robot_crowding": {"label": "RC"},
    "delayed_circular_crossing": {"label": "DCC"},
    "circular_crossing_with_static_obstacles": {"label": "CCSO"},
    "crowd_navigation": {"label": "CN"},
}
figure, ax = plt.subplots(2,2, figsize=(12,6))
figure.subplots_adjust(right=0.83, top=0.93, bottom=0.125, left=0.1, hspace=0.5, wspace=0.3)
for v, visibility in enumerate(test_robot_visibility):
    for p, metrics in enumerate(all_metrics):
        ax[v,p].set(
            xlabel='Number of humans',
            ylabel='Success rate',
            title=f"{'IL' if p == 0 else 'RL'} {'visible' if visibility else 'not visible'}",
        )
        ax[v,p].set_xticks(jnp.arange(len(n_humans_for_tests)), labels=[i for i in n_humans_for_tests])
        ax[v,p].set_ylim(metrics_data["successes"]["ylim"])
        ax[v,p].set_yticks(metrics_data["successes"]["yticks"])
        ax[v,p].set_yticklabels([i/10 for i in range(11)], fontsize=14)
        ax[v,p].grid()
        for s, scenario in enumerate(training_hyperparams['hybrid_scenario_subset']):
            successes = jnp.sum(
                jnp.where(
                    (metrics["scenario"][v] == scenario) & ~(jnp.isnan(metrics["times_to_goal"][v])), 
                    jnp.ones_like(metrics["scenario"][v]), 
                    jnp.zeros_like(metrics["scenario"][v]),
                ),
                axis=1,
            )
            total_episodes_successes = jnp.sum(
                jnp.where(
                    metrics["scenario"][v] == scenario,
                    jnp.ones_like(metrics["scenario"][v]),
                    jnp.zeros_like(metrics["scenario"][v]),
                ), 
                axis=1,
            )
            ax[v,p].plot(
                jnp.arange(len(n_humans_for_tests)), 
                successes / total_episodes_successes,
                color=list(mcolors.TABLEAU_COLORS.values())[s],
                linewidth=2,
                label=scenarios_data[SCENARIOS[scenario]]["label"],
            )
handles, labels = ax[0,0].get_legend_handles_labels()
figure.legend(labels, loc="center right", title=f"Scenario:", bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
figure.savefig(os.path.join(os.path.dirname(__file__),f"success_rate_tests.eps"), format='eps')