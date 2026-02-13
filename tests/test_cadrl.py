from jax import random, vmap
import jax.numpy as jnp
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1 as SocialReward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.policies.cadrl import CADRL

# Hyperparameters
random_seed = 0
n_episodes = 100
kinematics = 'unicycle'
env_params = {
    # 'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'n_humans': 5,
    'n_obstacles': 0,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': 'hybrid_scenario', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 0,
    'reward_function': SocialReward(kinematics=kinematics),
    'kinematics': kinematics,
    'lidar_noise': False,
}

# Initialize the environment
env = SocialNav(**env_params)

# Initialize the policy
policy = CADRL(
    reward_function=env.reward_function,
    kinematics=kinematics,
)
with open(os.path.join(os.path.dirname(__file__), 'cadrl.pkl'), 'rb') as f:
    network_params = pickle.load(f)['policy_params']

# Execute tests
# metrics = policy.evaluate(
#     n_episodes,
#     random_seed,
#     env,
# )

# Simulate some episodes
for i in range(n_episodes):
    reset_key, env_key, policy_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    step = 0
    max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
    all_states = jnp.array([state])
    all_observations = jnp.array([obs])
    all_robot_goals = jnp.array([info['robot_goal']])
    all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
    all_actions = jnp.zeros((max_steps, 2))
    # all_actions_values = jnp.zeros((max_steps,len(policy.action_space)))
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, _, _ = policy.act(policy_key, obs, info, network_params, epsilon=0.)
        # Step the environment (SocialNav)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
        # # Step the environment (Lasernav)
        # state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Save data for animation
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_observations = jnp.vstack((all_observations, jnp.array([obs])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
        all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
        all_actions = all_actions.at[step].set(action)
        # all_actions_values = all_actions_values.at[step].set(actions_cost)
        # Increment step
        step += 1
    all_actions = all_actions[:step]
    # all_actions_values = all_actions_values[:step]
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
    policy.animate_socialnav_trajectory(
        all_states[:-1],
        all_actions,
        # all_actions_values,
        all_robot_goals[:-1],
        all_humans_radii[:-1],
        env,
    )