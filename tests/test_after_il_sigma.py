from jax import random, nn
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl_ppo import SARLPPO

### Hyperparameters
random_seed = 0
n_episodes = 50
kinematics = 'unicycle'
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'progress_to_goal_reward': True,
    'progress_to_goal_weight': 0.03,
}
reward_function = Reward2(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize policy
policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
# Load actor parameters
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'rb') as f:
    il_out = pickle.load(f)
actor_params = il_out['actor_params']
print("Logsigma parameter: ", actor_params['actor']['logsigma'])
print("Sigma: ", jnp.exp(actor_params['actor']['logsigma']))

### Watch n_samples action sampled from actor output at the initial state
n_samples = 1_000
state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
mean_action, _, _, sampled_action, distrs = policy.act(random.PRNGKey(random_seed), obs, info, actor_params, False)
# Print pdf value of mean action
pdf_mean_action = jnp.exp(-policy._compute_neg_log_pdf_value(distrs["mu1"], distrs["mu2"], distrs["logsigma"], sampled_action))
print(f"Probability of mean action: {pdf_mean_action}\nMeans: [{distrs['mu1']}, {distrs['mu2']}]\nAction: [{sampled_action[0]}, {sampled_action[1]}]")
keys = random.split(random.PRNGKey(random_seed), n_samples)
actions, sampled_actions = policy.batch_sample_action(
    distrs["mu1"],
    distrs["mu2"],
    jnp.exp(distrs["logsigma"]),
    keys,
)
# Action samples in the (v,omega) space
figure, ax = plt.subplots(figsize=(10,10))
figure.suptitle(f'{n_samples} actor outputs at the initial state - Sigma {jnp.exp(distrs["logsigma"])}')
# ax.axis('equal')
ax.plot(actions[:,0], actions[:,1], 'o')
if kinematics == 'unicycle':
    ax.set_xlabel('v ($m/s$)')
    ax.set_ylabel('$\omega$ $(rad/s)$')
    ax.set_xlim(0, policy.v_max)
    ax.set_ylim(-policy.v_max * 2/ policy.wheels_distance, policy.v_max * 2 / policy.wheels_distance)
else:
    ax.set_xlabel('vx ($m/s$)')
    ax.set_ylabel('vy $(m/s)$')
ax.plot(mean_action[0], mean_action[1], 'ro')
plt.show()
# Distribution of each action component
figure, ax = plt.subplots(1,2, figsize=(10,10))
figure.suptitle(f'{n_samples} actor outputs at the initial state - Sigma {jnp.exp(distrs["logsigma"])}')
ax[0].hist(actions[:,0], bins=100, density=True)
ax[1].hist(actions[:,1], bins=100, density=True)
ax[0].axvline(mean_action[0], color='red', linestyle='dashed', linewidth=2)
ax[1].axvline(mean_action[1], color='red', linestyle='dashed', linewidth=2)
if kinematics == 'unicycle':
    ax[0].set_title('v ($m/s$)')
    ax[1].set_title('$\omega$ $(rad/s)$')
else:
    ax[0].set_title('vx ($m/s$)')
    ax[1].set_title('vy $(m/s)$')
plt.show()
# Distribution of v_left and v_right in case of unicycle kinematics
if kinematics == 'unicycle':
    figure, ax = plt.subplots(1,2, figsize=(10,10))
    figure.suptitle(f'{n_samples} actor outputs at the initial state - Sigma {jnp.exp(distrs["logsigma"])}')
    v_right = (2 * actions[:,0] + actions[:,1] * policy.wheels_distance) / 2
    v_left = (2 * actions[:,0] - actions[:,1] * policy.wheels_distance) / 2
    mean_action_v_right = (2 * mean_action[0] + mean_action[1] * policy.wheels_distance) / 2
    mean_action_v_left = (2 * mean_action[0] - mean_action[1] * policy.wheels_distance) / 2
    ax[0].hist(v_left, bins=100, density=True)
    ax[1].hist(v_right, bins=100, density=True)
    ax[0].axvline(mean_action_v_left, color='red', linestyle='dashed', linewidth=2)
    ax[1].axvline(mean_action_v_right, color='red', linestyle='dashed', linewidth=2)
    ax[0].set_title('v_left ($m/s$)')
    ax[1].set_title('v_right $(m/s)$')
    plt.show()