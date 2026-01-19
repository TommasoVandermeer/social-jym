from jax import random, vmap, jit, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 1
n_episodes = 50
kinematics = 'unicycle'
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': 2*jnp.pi,
    'lidar_max_dist': 10.,
    'n_humans': 3,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': False,
    'scenario': 'hybrid_scenario',
    'reward_function': Reward(robot_radius=0.3),
    'kinematics': kinematics,
    'lidar_noise': True,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = JESSI()
# with open(os.path.join(os.path.dirname(__file__), 'perception_network.pkl'), 'rb') as f:
#     encoder_params = pickle.load(f)
# with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'rb') as f:
#     actor_params = pickle.load(f)
# network_params = policy.merge_nns_params(encoder_params, actor_params)
with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
    network_params, _ = pickle.load(f)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    step = 0
    max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
    all_states = jnp.array([state])
    all_observations = jnp.array([obs])
    all_robot_goals = jnp.array([info['robot_goal']])
    all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
    all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
    all_actions = jnp.zeros((max_steps, 2))
    all_rewards = jnp.zeros((max_steps,))
    all_predicted_state_values = jnp.zeros((max_steps,))
    all_actor_distrs = {
        'alphas': jnp.zeros((max_steps, 3)),
        'vertices': jnp.zeros((max_steps, 3, 2)),
    }
    bigauss = {
        "means": jnp.zeros((max_steps,policy.n_detectable_humans,2)),
        "logsigmas": jnp.zeros((max_steps,policy.n_detectable_humans,2)),
        "correlation": jnp.zeros((max_steps,policy.n_detectable_humans)),
    }
    all_encoder_distrs = {
        "pos_distrs": bigauss,
        "vel_distrs": bigauss,
        "weights": jnp.zeros((max_steps,policy.n_detectable_humans)),
    }
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, _, _, _, perception_distr, actor_distr, state_value = policy.act(random.PRNGKey(0), obs, info, network_params, sample=False)
        print("Dirichlet distribution parameters: ", actor_distr['alphas'])
        # print("Predicted HCGs scores", [f"{w:.2f}" for w in perception_distr['weights']])
        # Step the environment
        state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Save data for animation
        all_actions = all_actions.at[step].set(action)
        all_rewards = all_rewards.at[step].set(reward)
        all_predicted_state_values = all_predicted_state_values.at[step].set(state_value)
        all_actor_distrs = tree_map(lambda x, y: x.at[step].set(y), all_actor_distrs, actor_distr)
        all_encoder_distrs = tree_map(lambda x, y: x.at[step].set(y), all_encoder_distrs, perception_distr)
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_observations = jnp.vstack((all_observations, jnp.array([obs])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
        all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
        all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
        # Increment step
        step += 1
    all_encoder_distrs = tree_map(lambda x: x[:step], all_encoder_distrs)
    all_actor_distrs = tree_map(lambda x: x[:step], all_actor_distrs)
    all_actions = all_actions[:step]
    all_rewards = all_rewards[:step]
    all_predicted_state_values = all_predicted_state_values[:step]
    ## Check predicted state values and actual discounted returns
    @jit
    def _discounted_cumsum(rewards):
        def scan_fun(carry, reward):
            new_carry = reward + carry * jnp.power(policy.gamma, policy.dt * policy.v_max)
            return new_carry, new_carry
        _, discounted_cumsums = lax.scan(scan_fun, 0.0, rewards[::-1])
        return discounted_cumsums[::-1]
    discounted_returns = _discounted_cumsum(all_rewards)
    # [print("Step {} -  critic prediction: {:.2f} VS discounted return: {:.2f}".format(i, all_predicted_state_values[i], discounted_returns[i])) for i in range(len(discounted_returns))]
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
    ## Animate only trajectory
    angles = vmap(lambda robot_yaw: jnp.linspace(robot_yaw - env.lidar_angular_range/2, robot_yaw + env.lidar_angular_range/2, env.lidar_num_rays))(all_states[:,-1,4])
    lidar_measurements = vmap(lambda mes, ang: jnp.stack((mes, ang), axis=-1))(all_observations[:,0,6:], angles)
    # animate_trajectory(
    #     all_states, 
    #     info['humans_parameters'][:,0], 
    #     env.robot_radius, 
    #     'hsfm',
    #     info['robot_goal'],
    #     info['current_scenario'],
    #     static_obstacles=info['static_obstacles'][-1],
    #     robot_dt=env_params['robot_dt'],
    #     # lidar_measurements=lidar_measurements,
    #     kinematics=kinematics,
    # )
    ## Animate trajectory with JESSI's perception and action distribution
    policy.animate_lasernav_trajectory(
        all_states[:-1],
        all_observations[:-1],
        all_actions,
        all_actor_distrs,
        all_encoder_distrs,
        all_robot_goals[:-1],
        all_static_obstacles[:-1],
        all_humans_radii[:-1]
    )