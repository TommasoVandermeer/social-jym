import jax.numpy as jnp
from jax import random, vmap, lax, jit
from jax_tqdm import loop_tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import animate_trajectory, plot_state, plot_trajectory, load_socialjym_policy
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

random_seed = 0
n_episodes = 1000
n_humans = 5
robot_policy = 'sarl' # 'cadrl', 'sarl', 'random'
# Reward terms params
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.035 # High rotation penalty weight
w_bound = 1. # Rotation bound

# Initialize and reset environment
env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'circular_crossing',
    'hybrid_scenario_subset': jnp.array([0,1], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': Reward1(kinematics='unicycle', discomfort_distance=ds),
    'kinematics': 'unicycle',
}
env = SocialNav(**env_params)

# Initialize robot policy and vnet params
if (robot_policy == 'cadrl') or (robot_policy == 'random'):
    policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics='unicycle')
    vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/cadrl_k1_nh1_hp1_s4_r1_20_11_2024.pkl"))
elif robot_policy == 'sarl':
    policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics='unicycle')
    vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/sarl_hsfm_unicycle_reward_0_circular_crossing_09_12_2024.pkl"))

# Epsilon greedy policy
if robot_policy == 'random':
    epsilon = 1.
else:
    epsilon = 0.

### Initialize all the rewards we need to save all contributions
# Progress to goal reward
reward_params_1 = {
    'target_reached_reward': False,
    'collision_penalty_reward': False,
    'discomfort_penalty_reward': False,
    'progress_to_goal_reward': True,
    'time_penalty_reward': False,
    'high_rotation_penalty_reward': False,
    'progress_to_goal_weight': wp,
}
reward_function_1 = Reward2(**reward_params_1)
# Time penalty reward
reward_params_2 = {
    'target_reached_reward': False,
    'collision_penalty_reward': False,
    'discomfort_penalty_reward': False,
    'progress_to_goal_reward': False,
    'time_penalty_reward': True,
    'high_rotation_penalty_reward': False,
    'time_penalty': wt,
}
reward_function_2 = Reward2(**reward_params_2)
# High rotation penalty reward
reward_params_3 = {
    'target_reached_reward': False,
    'collision_penalty_reward': False,
    'discomfort_penalty_reward': False,
    'progress_to_goal_reward': False,
    'time_penalty_reward': False,
    'high_rotation_penalty_reward': True,
    'angular_speed_bound': w_bound,
    'angular_speed_penalty_weight': wr,
}
reward_function_3 = Reward2(**reward_params_3)

@loop_tqdm(n_episodes)
@jit
def _fori_body(i:int, for_val:tuple):
    @jit
    def _while_body(while_val:tuple):
        # Retrieve data from the tuple
        state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards_0, all_rewards_1, all_rewards_2, all_rewards_3 = while_val
        # Make a step in the environment
        action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, epsilon)
        # Compute additional rewards
        all_rewards_1 = all_rewards_1.at[steps].set(reward_function_1(obs, info, env_params['robot_dt'])[0])
        all_rewards_2 = all_rewards_2.at[steps].set(reward_function_2(obs, info, env_params['robot_dt'])[0])
        all_rewards_3 = all_rewards_3.at[steps].set(reward_function_3(obs, info, env_params['robot_dt'])[0])
        # Step the environment
        state, obs, info, reward, outcome = env.step(state,info,action,test=True)
        # Save data
        all_actions = all_actions.at[steps].set(action)
        all_states = all_states.at[steps].set(state)
        all_rewards_0 = all_rewards_0.at[steps].set(reward)
        # Update step counter
        steps += 1
        return state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards_0, all_rewards_1, all_rewards_2, all_rewards_3

    ## Retrieve data from the tuple
    seed, metrics = for_val
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
    ## Reset the environment
    state, reset_key, obs, info, init_outcome = env.reset(reset_key)
    ## Episode loop
    all_actions = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
    all_states = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, env.n_humans+1, 6))
    all_rewards_0 = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
    all_rewards_1 = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
    all_rewards_2 = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
    all_rewards_3 = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1,))
    while_val_init = (state, obs, info, init_outcome, policy_key, 0, all_actions, all_states, all_rewards_0, all_rewards_1, all_rewards_2, all_rewards_3)
    _, _, _, outcome, policy_key, episode_steps, all_actions, all_states, all_rewards_0, all_rewards_1, all_rewards_2, all_rewards_3 = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
    ## Update metrics
    metrics["successes"] = lax.cond(outcome["success"], lambda x: x + 1, lambda x: x, metrics["successes"])
    metrics["collisions"] = lax.cond(outcome["failure"], lambda x: x + 1, lambda x: x, metrics["collisions"])
    metrics["timeouts"] = lax.cond(outcome["timeout"], lambda x: x + 1, lambda x: x, metrics["timeouts"])
    @jit
    def _compute_state_value_for_body(j:int, t:int, tup:tuple):
        value, rewards = tup
        value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * rewards[j]
        return value, rewards
    metrics["returns0"] = metrics["returns0"].at[i].set(lax.fori_loop(0, episode_steps, lambda rr, x: _compute_state_value_for_body(rr, 0, x), (0., all_rewards_0))[0])
    metrics["returns1"] = metrics["returns1"].at[i].set(lax.fori_loop(0, episode_steps, lambda rr, x: _compute_state_value_for_body(rr, 0, x), (0., all_rewards_1))[0])
    metrics["returns2"] = metrics["returns2"].at[i].set(lax.fori_loop(0, episode_steps, lambda rr, x: _compute_state_value_for_body(rr, 0, x), (0., all_rewards_2))[0])
    metrics["returns3"] = metrics["returns3"].at[i].set(lax.fori_loop(0, episode_steps, lambda rr, x: _compute_state_value_for_body(rr, 0, x), (0., all_rewards_3))[0])
    seed += 1
    return seed, metrics

### Simulate some episodes
# Initialize metrics
metrics = {
    "successes": 0, 
    "collisions": 0, 
    "timeouts": 0, 
    "returns0": jnp.empty((n_episodes,)),
    "returns1": jnp.empty((n_episodes,)),
    "returns2": jnp.empty((n_episodes,)),
    "returns3": jnp.empty((n_episodes,)),}
# Execute n_episodes tests
print(f"\nExecuting {n_episodes} tests with {env.n_humans} humans...")
_, metrics = lax.fori_loop(0, n_episodes, _fori_body, (random_seed, metrics))

# Plot cumulative returns contributions for all episodes
figure, ax = plt.subplots(figsize=(10, 10))
ax.set(
    xlabel='Reward type', 
    ylabel='Return', 
    title=f'Cumulative reward for each reward type - {n_episodes} trials - {n_humans} humans - robot policy: {robot_policy}\nDs: {ds}, Wp: {wp}, Wt: {wt}, Wr: {wr}, Wbound: {w_bound}',  
    xticklabels=['Base', 'Progress to goal', 'Time penalty', 'High rotation penalty'],
    ylim=(-2, 1))
ax.grid()
ax.boxplot(np.transpose(np.array([metrics['returns0'], metrics['returns1'], metrics['returns2'], metrics['returns3']])), widths=0.4, patch_artist=True, 
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue", alpha=0.7),
            whiskerprops=dict(color="blue", alpha=0.7),
            capprops=dict(color="blue", alpha=0.7),
            medianprops=dict(color="blue", alpha=0.7),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True, 
            zorder=1)
figure.savefig(os.path.join(os.path.dirname(__file__),f"{robot_policy}_return_contributions_for_reward_type_{n_humans}humans.png"), format='png')