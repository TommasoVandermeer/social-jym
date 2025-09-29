from jax import random, vmap, debug, lax, jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import optax
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import linear_decay, test_k_trials, save_policy_params
from socialjym.utils.replay_buffers.uniform_vnet_replay_buffer import UniformVNetReplayBuffer
from socialjym.utils.rollouts.vnet_rollouts import vnet_il_rollout, vnet_rl_rollout

### Hyperparameters
random_seed = 0
n_epochs = 50
kinematics = 'unicycle'
unicycle_box = False
training_hyperparams = {
    'random_seed': 0,
    'kinematics': kinematics,
    'policy_name': 'cadrl', # 'cadrl' or 'sarl'
    'n_humans': 1,  # CADRL uses 1, SARL uses 5
    'il_training_episodes': 2_000,
    'il_learning_rate': 0.01,
    'il_num_epochs': n_epochs, # Number of epochs to train the model after ending IL
    'rl_training_episodes': 10_000,
    'rl_learning_rate': 0.001,
    'rl_num_batches': 100, # Number of batches to train the model after each RL episode
    'batch_size': 100, # Number of experiences to sample from the replay buffer for each model update
    'epsilon_start': 0.5,
    'epsilon_end': 0.1,
    'epsilon_decay': 4_000,
    'buffer_size': 100_000, # Maximum number of experiences to store in the replay buffer (after exceeding this limit, the oldest experiences are overwritten with new ones)
    'target_update_interval': 50, # Number of episodes to wait before updating the target network for RL (the one used to compute the target state values)
    'humans_policy': 'sfm',
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1], np.int32), # Subset of the hybrid scenarios to use for training
    'reward_function': 'socialnav_reward1',
    'custom_episodes': False, # If True, the episodes are loaded from a predefined set
}
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
reward_function = Reward1(**reward_params)
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
    'kinematics': training_hyperparams['kinematics'],
}

### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy
# Initialize robot policy and vnet params
if training_hyperparams['policy_name'] == "cadrl": 
    policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics, unicycle_box_action_space=unicycle_box)
    initial_vnet_params = policy.model.init(random.key(training_hyperparams['random_seed']), jnp.zeros((policy.vnet_input_size,)))
elif training_hyperparams['policy_name'] == "sarl":
    policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics, unicycle_box_action_space=unicycle_box)
    initial_vnet_params = policy.model.init(random.key(training_hyperparams['random_seed']), jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
else: raise ValueError(f"{training_hyperparams['policy_name']} is not a valid policy name")

### Plot action space
# Plot (v,w) action space
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 17}
rc('font', **font)
figure, ax = plt.subplots(figsize=(6,10))
figure.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.1)
# unsafe_actions = jnp.where(
#     (
#         (policy.action_space[:,0] > policy.v_max * 0.53333) | \
#         ((policy.action_space[:,1] > 0) & (policy.action_space[:,1] > ((2 * 0.56 * (policy.v_max - policy.action_space[:,0] * 0.5333))/policy.wheels_distance))) | \
#         ((policy.action_space[:,1] < 0) & (policy.action_space[:,1] < ((2 * (policy.action_space[:,0] * 0.5333 - policy.v_max))/policy.wheels_distance)))
#     ), 
#     True, 
#     False
# )
# ax.scatter(policy.action_space[unsafe_actions,0], policy.action_space[unsafe_actions,1], zorder=2, label="Unsafe", color='red', s=60)
# ax.scatter(policy.action_space[~unsafe_actions,0], policy.action_space[~unsafe_actions,1], zorder=2, label="Safe", color='green', s=60)
ax.scatter(policy.action_space[:,0], policy.action_space[:,1], zorder=3, label="Sampled actions", color='blue', s=60)
actions_space_bound = Polygon(
    jnp.array([[policy.v_max,0.],[0.,policy.v_max*2/policy.wheels_distance],[0.,-policy.v_max*2/policy.wheels_distance]]), 
    closed=True, 
    fill=None, 
    edgecolor='black',
    linewidth=2,
    zorder=1,
)
ax.add_patch(actions_space_bound)
# ax.gca().set_aspect('equal', adjustable='box')
# ax.set_title(f"Action space (V,w) - Wheelbase: {policy.wheels_distance/2}m - Vmax: {policy.v_max}m/s")
ax.set_xlabel("$v$ $(m/s)$")
ax.set_ylabel("$\omega$ $(rad/s)$", labelpad=-16)
ax.legend(fontsize=27)
ax.grid()
plt.show()
# figure.savefig(os.path.join(os.path.dirname(__file__),"unicycle_action_space.pdf"), format='pdf')
# Plot (px,py,theta) for each action starting from (0,0,0) with analytical integration
def exact_integration_of_action_space(x:jnp.ndarray, action:jnp.ndarray) -> jnp.ndarray:
    @jit
    def exact_integration_with_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + action[0] * jnp.cos(x[2]) * env_params['robot_dt'])
        x = x.at[1].set(x[1] + action[0] * jnp.sin(x[2]) * env_params['robot_dt'])
        return x
    @jit
    def exact_integration_with_non_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + (action[0]/action[1]) * (jnp.sin(x[2] + action[1] * env_params['robot_dt']) - jnp.sin(x[2])))
        x = x.at[1].set(x[1] + (action[0]/action[1]) * (jnp.cos(x[2]) - jnp.cos(x[2] + action[1] * env_params['robot_dt'])))
        x = x.at[2].set(x[2] + action[1] * env_params['robot_dt'])
        return x
    x = lax.cond(
        action[1] != 0,
        exact_integration_with_non_zero_omega,
        exact_integration_with_zero_omega,
        x)
    return x
pxy_theta = vmap(exact_integration_of_action_space, in_axes=(0, 0))(jnp.zeros((len(policy.action_space),3)), policy.action_space)
orientations = jnp.ones((len(policy.action_space), 2)) * 0.05 * jnp.array([jnp.cos(pxy_theta[:,2]), jnp.sin(pxy_theta[:,2])]).T
plt.scatter(pxy_theta[:,0], pxy_theta[:,1])
for i, orientation in enumerate(orientations):
    plt.arrow(pxy_theta[i,0], pxy_theta[i,1], orientation[0], orientation[1], color='black', head_width=0.005, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("All robot positions and orientations applying action space (V,w) starting from Px=Py=0m, theta=0r and using exact (analytical) integration")
plt.xlabel("x (x)")
plt.ylabel("y (m)")
plt.show()

### LEARNING
loss_during_il = np.empty((n_epochs,))
returns_after_il = np.empty((1000,))
returns_during_rl = np.empty((10_000,))
returns_after_rl = np.empty((1000,))
# Initialize replay buffer
replay_buffer = UniformVNetReplayBuffer(training_hyperparams['buffer_size'], training_hyperparams['batch_size'])
# Initialize IL optimizer
optimizer = optax.sgd(learning_rate=training_hyperparams['il_learning_rate'], momentum=0.9)
# Initialize buffer state
buffer_state = {
    'vnet_inputs': jnp.empty((training_hyperparams['buffer_size'], env.n_humans, policy.vnet_input_size)),
    'targets': jnp.empty((training_hyperparams['buffer_size'],1)),
}
# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    il_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/il_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    il_custom_episodes_path = None
# Initialize IL rollout params
il_rollout_params = {
    'initial_vnet_params': initial_vnet_params,
    'train_episodes': training_hyperparams['il_training_episodes'],
    'random_seed': training_hyperparams['random_seed'],
    'optimizer': optimizer,
    'buffer_state': buffer_state,
    'current_buffer_size': 0,
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'buffer_size': training_hyperparams['buffer_size'],
    'num_epochs': training_hyperparams['il_num_epochs'],
    'batch_size': training_hyperparams['batch_size'],
    'custom_episodes': il_custom_episodes_path
}

# IMITATION LEARNING ROLLOUT
il_out = vnet_il_rollout(**il_rollout_params)

# Save the IL model parameters, buffer state, and keys
il_model_params = il_out['model_params']
buffer_state = il_out['buffer_state']
current_buffer_size = il_out['current_buffer_size']
loss_during_il = il_out['losses']

# Execute tests to evaluate return after IL
metrics_after_il = test_k_trials(
    1000, 
    training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
    env, 
    policy, 
    il_model_params, 
    reward_function.time_limit)
returns_after_il = metrics_after_il['returns']

# Initialize RL optimizer
optimizer = optax.sgd(learning_rate=training_hyperparams['rl_learning_rate'], momentum=0.9)

# Initialize custom episodes path
if training_hyperparams['custom_episodes']:
    rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
else:
    rl_custom_episodes_path = None

# Initialize RL rollout params
rl_rollout_params = {
    'initial_vnet_params': il_model_params,
    'train_episodes': training_hyperparams['rl_training_episodes'],
    'random_seed': training_hyperparams['random_seed'] + training_hyperparams['il_training_episodes'],
    'model': policy.model,
    'optimizer': optimizer,
    'buffer_state': buffer_state,
    'current_buffer_size': current_buffer_size,
    'policy': policy,
    'env': env,
    'replay_buffer': replay_buffer,
    'buffer_size': training_hyperparams['buffer_size'],
    'num_batches': training_hyperparams['rl_num_batches'],
    'epsilon_decay_fn': linear_decay,
    'epsilon_start': training_hyperparams['epsilon_start'],
    'epsilon_end': training_hyperparams['epsilon_end'],
    'decay_rate': training_hyperparams['epsilon_decay'],
    'target_update_interval': training_hyperparams['target_update_interval'],
    'custom_episodes': rl_custom_episodes_path,
}

# REINFORCEMENT LEARNING ROLLOUT
rl_out = vnet_rl_rollout(**rl_rollout_params)

# Save the training returns
rl_model_params = rl_out['model_params']
returns_during_rl = rl_out['returns']  

# Execute tests to evaluate return after RL
metrics_after_rl = test_k_trials(
    1000, 
    training_hyperparams['il_training_episodes'] + training_hyperparams['rl_training_episodes'], 
    env, 
    policy, 
    rl_model_params, 
    reward_function.time_limit)
returns_after_rl = metrics_after_rl['returns']  

### SAVE TRAINED POLICY PARAMS
save_policy_params(
    training_hyperparams['policy_name'], 
    rl_model_params, 
    env.get_parameters(), 
    reward_function.get_parameters(), 
    training_hyperparams, 
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/"))

### FINAL PLOTS
# Plot loss curve during IL for each seed
figure0, ax0 = plt.subplots(figsize=(10,10))
ax0.set(
    xlabel='Epoch', 
    ylabel='Loss', 
    title='Loss during IL training for each seed')
ax0.plot(
    np.arange(len(loss_during_il)), 
    loss_during_il,
    color = list(mcolors.TABLEAU_COLORS.values())[0])
figure0.savefig(os.path.join(os.path.dirname(__file__),"CADRL_UNICYCLE_loss_curves_during_il.eps"), format='eps')

# Plot return during RL curve for each seed
figure, ax = plt.subplots(figsize=(10,10))
window = 500
ax.set(
    xlabel='Training episode', 
    ylabel=f"Return moving average over {window} episodes", 
    title='Return during RL training for each seed')
ax.plot(
    np.arange(len(returns_during_rl)-(window-1))+window, 
    jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
    color = list(mcolors.TABLEAU_COLORS.values())[0])
figure.savefig(os.path.join(os.path.dirname(__file__),"CADRL_UNICYCLE_return_curves_during_rl.eps"), format='eps')

# Plot boxplot of the returns for each seed
figure2, ax2 = plt.subplots(figsize=(10,10))
ax2.set(xlabel='Seed', ylabel='Return', title='Return after IL and RL training for each seed')
ax2.boxplot(returns_after_il, widths=0.4, patch_artist=True, 
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True, 
            zorder=1)
ax2.boxplot(returns_after_rl, widths=0.3, patch_artist=True, 
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral", alpha=0.4),
            whiskerprops=dict(color="coral", alpha=0.4),
            capprops=dict(color="coral", alpha=0.4),
            medianprops=dict(color="coral", alpha=0.4),
            meanprops=dict(markerfacecolor="coral", markeredgecolor="coral"), 
            showfliers=False,
            showmeans=True,
            zorder=2)
legend_elements = [
    Line2D([0], [0], color="lightblue", lw=4, label="After IL"),
    Line2D([0], [0], color="lightcoral", lw=4, label="After RL")
]
ax2.legend(handles=legend_elements, loc="upper right")
figure2.savefig(os.path.join(os.path.dirname(__file__),"CADRL_UNICYCLE_return_curves_after_il_and_rl.png"), format='png')