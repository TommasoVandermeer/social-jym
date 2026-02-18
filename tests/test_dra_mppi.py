from jax import random, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1 as SocialReward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.policies.dra_mppi import DRAMPPI
from socialjym.policies.jessi import JESSI

# Hyperparameters
use_ground_truth_data = False
random_seed = 0
n_episodes = 100
kinematics = 'unicycle'

if use_ground_truth_data:
    env_params = {
        'n_humans': 5,
        'n_obstacles': 5,
        'robot_radius': 0.3,
        'robot_dt': 0.25,
        'humans_dt': 0.01,      
        'robot_visible': True,
        'scenario': 'hybrid_scenario', 
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
        'ccso_n_static_humans': 0,
        'reward_function': SocialReward(kinematics=kinematics),
        'kinematics': kinematics,
    }
    # Initialize the environment
    env = SocialNav(**env_params)
    # Initialize the policy
    policy = DRAMPPI()
    # Execute tests
    # metrics = policy.evaluate(
    #     n_episodes,
    #     random_seed,
    #     env,
    # )
    # Simulate some episodes on GROUND TRUTH DATA
    for i in range(n_episodes):
        reset_key, env_key, policy_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
        state, reset_key, obs, info, outcome = env.reset(reset_key)
        step = 0
        max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
        all_states = jnp.array([state])
        all_observations = jnp.array([obs])
        all_robot_goals = jnp.array([info['robot_goal']])
        all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
        all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
        all_actions = jnp.zeros((max_steps, 2))
        all_u_means = jnp.zeros((max_steps, policy.horizon, 2))
        all_trajectories = jnp.zeros((max_steps, policy.num_samples, policy.horizon+1, 3))
        all_trajectories_costs = jnp.zeros((max_steps,policy.num_samples))
        all_humans_distrs = {
            "means": jnp.zeros((max_steps, policy.horizon, env.n_humans,2)),
            "logsigmas": jnp.zeros((max_steps,policy.horizon, env.n_humans,2)),
            "correlation": jnp.zeros((max_steps,policy.horizon, env.n_humans)),
        }
        u_mean, beta = policy.init_u_mean_and_beta()
        while outcome["nothing"]:
            # Compute action 
            action, u_mean, beta, trajectories, costs, hum_distrs, policy_key = policy.act(obs, info, u_mean, beta, policy_key)
            # Step the environment (SocialNav)
            state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
            # Save data for animation
            all_states = jnp.vstack((all_states, jnp.array([state])))
            all_observations = jnp.vstack((all_observations, jnp.array([obs])))
            all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
            all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
            all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
            all_actions = all_actions.at[step].set(action)
            all_u_means = all_u_means.at[step].set(u_mean)
            all_trajectories = all_trajectories.at[step].set(trajectories)
            all_trajectories_costs = all_trajectories_costs.at[step].set(costs)
            all_humans_distrs = tree_map(lambda x, y: x.at[step].set(y), all_humans_distrs, hum_distrs)
            # Increment step
            step += 1
        all_actions = all_actions[:step]
        all_u_means = all_u_means[:step]
        all_trajectories = all_trajectories[:step]
        all_trajectories_costs = all_trajectories_costs[:step]
        all_humans_distrs = tree_map(lambda x: x[:step], all_humans_distrs)
        print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
        policy.animate_socialnav_trajectory(
            all_states[:-1],
            all_actions,
            all_u_means,
            all_trajectories,
            all_trajectories_costs,
            all_robot_goals[:-1],
            all_static_obstacles[:-1],
            all_humans_radii[:-1],
            all_humans_distrs,
            env,
        )
else:
    env_params = {
        'n_stack': 5,
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
        'reward_function': LaserReward(robot_radius=0.3),
        'kinematics': kinematics,
        'lidar_noise': True,
    }
    # Initialize the environment
    env = LaserNav(**env_params)
    # Initialize the policy
    policy = DRAMPPI()
    jessi =  JESSI(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
    )
    with open(os.path.join(os.path.dirname(__file__), 'sarl.pkl'), 'rb') as f:
        network_params = pickle.load(f)['policy_params']
    with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
        perception_params = pickle.load(f)
    # with open(os.path.join(os.path.dirname(__file__), 'jessi_e2e_rl_out.pkl'), 'rb') as f:
    #     jessi_network_params, _, _ = pickle.load(f)
    # Execute tests
    # metrics = policy.evaluate_on_jessi_perception(
    #     n_episodes,
    #     random_seed,
    #     env,
    #     jessi,
    #     perception_params,
    #     network_params,
    #     humans_radius_hypothesis=jnp.full((jessi.n_detectable_humans,), .3),
    # )
    # Simulate some episodes on PERCEIVED DATA
    for i in range(n_episodes):
        reset_key, env_key, policy_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
        state, reset_key, obs, info, outcome = env.reset(reset_key)
        step = 0
        max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
        all_states = jnp.array([state])
        all_observations = jnp.array([obs])
        all_robot_goals = jnp.array([info['robot_goal']])
        all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
        all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
        all_actions = jnp.zeros((max_steps, 2))
        all_u_means = jnp.zeros((max_steps, policy.horizon, 2))
        all_trajectories = jnp.zeros((max_steps, policy.num_samples, policy.horizon+1, 3))
        all_trajectories_costs = jnp.zeros((max_steps,policy.num_samples))
        all_humans_distrs = {
            "means": jnp.zeros((max_steps, policy.horizon, jessi.n_detectable_humans,2)),
            "logsigmas": jnp.zeros((max_steps,policy.horizon, jessi.n_detectable_humans,2)),
            "correlation": jnp.zeros((max_steps,policy.horizon, jessi.n_detectable_humans)),
        }
        bigauss = {
            "means": jnp.zeros((max_steps,jessi.n_detectable_humans,2)),
            "logsigmas": jnp.zeros((max_steps,jessi.n_detectable_humans,2)),
            "correlation": jnp.zeros((max_steps,jessi.n_detectable_humans)),
        }
        all_encoder_distrs = {
            "pos_distrs": bigauss,
            "vel_distrs": bigauss,
            "weights": jnp.zeros((max_steps,jessi.n_detectable_humans)),
        }
        u_mean, beta = policy.init_u_mean_and_beta()
        while outcome["nothing"]:
            # Compute action 
            action, u_mean, beta, trajectories, costs, hum_distrs, perception_distrs, policy_key = policy.act_on_jessi_perception(jessi, perception_params, policy_key, obs, info, u_mean, beta)
            # # Step the environment (Lasernav)
            state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
            # Save data for animation
            all_states = jnp.vstack((all_states, jnp.array([state])))
            all_observations = jnp.vstack((all_observations, jnp.array([obs])))
            all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
            all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
            all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
            all_actions = all_actions.at[step].set(action)
            all_u_means = all_u_means.at[step].set(u_mean)
            all_trajectories = all_trajectories.at[step].set(trajectories)
            all_trajectories_costs = all_trajectories_costs.at[step].set(costs)
            all_humans_distrs = tree_map(lambda x, y: x.at[step].set(y), all_humans_distrs, hum_distrs)
            all_encoder_distrs = tree_map(lambda x, y: x.at[step].set(y), all_encoder_distrs, perception_distrs)
            # Increment step
            step += 1
        all_actions = all_actions[:step]
        all_u_means = all_u_means[:step]
        all_trajectories = all_trajectories[:step]
        all_trajectories_costs = all_trajectories_costs[:step]
        all_humans_distrs = tree_map(lambda x: x[:step], all_humans_distrs)
        all_encoder_distrs = tree_map(lambda x: x[:step], all_encoder_distrs)
        print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
        policy.animate_lasernav_trajectory(
            all_states[:-1],
            all_observations[:-1],
            all_actions,
            all_u_means,
            all_trajectories,
            all_trajectories_costs,
            all_robot_goals[:-1],
            all_static_obstacles[:-1],
            all_humans_radii[:-1],
            all_humans_distrs,
            all_encoder_distrs,
            env,
        )