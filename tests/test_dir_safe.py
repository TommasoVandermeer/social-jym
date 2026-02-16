from jax import random, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1 as SocialReward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.jessi import JESSI

# Hyperparameters
use_ground_truth_data = False
random_seed = 0
n_episodes = 100
kinematics = 'unicycle'
n_humans = 1
n_obstacles = 5

if use_ground_truth_data:
    env_params = {
        'n_humans': n_humans,
        'n_obstacles': n_obstacles,
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
    policy = DIRSAFE(
        reward_function=env.reward_function,
    )
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe.pkl'), 'rb') as f:
        network_params = pickle.load(f)['actor_params']
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
        all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
        all_actions = jnp.zeros((max_steps, 2))
        all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
        all_actor_distrs = {
            'alphas': jnp.zeros((max_steps, 3)),
            'vertices': jnp.zeros((max_steps, 3, 2)),
        }
        while outcome["nothing"]:
            # Compute action from trained JESSI
            action, _, _, _, action_distr = policy.act(policy_key, obs, info, network_params, sample=False)
            # Step the environment (SocialNav)
            state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
            # Save data for animation
            all_states = jnp.vstack((all_states, jnp.array([state])))
            all_observations = jnp.vstack((all_observations, jnp.array([obs])))
            all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
            all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
            all_actions = all_actions.at[step].set(action)
            all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
            all_actor_distrs = tree_map(lambda x, y: x.at[step].set(y), all_actor_distrs, action_distr)
            # Increment step
            step += 1
        all_actions = all_actions[:step]
        all_actor_distrs = tree_map(lambda x: x[:step], all_actor_distrs)
        print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
        policy.animate_socialnav_trajectory(
            all_states[:-1],
            all_actions,
            all_robot_goals[:-1],
            all_humans_radii[:-1],
            env,
            static_obstacles=all_static_obstacles[:-1],
            action_distrs=all_actor_distrs,
        )
else:
    env_params = {
        'n_stack': 5,
        'lidar_num_rays': 100,
        'lidar_angular_range': jnp.pi * 2,
        'lidar_max_dist': 10.,
        'n_humans': n_humans,
        'n_obstacles': n_obstacles,
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
    policy = DIRSAFE(
        reward_function=SocialReward(kinematics=kinematics),
    )
    jessi =  JESSI(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
    )
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe.pkl'), 'rb') as f:
        network_params = pickle.load(f)['actor_params']
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
        all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
        all_actions = jnp.zeros((max_steps, 2))
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
        all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
        all_actor_distrs = {
            'alphas': jnp.zeros((max_steps, 3)),
            'vertices': jnp.zeros((max_steps, 3, 2)),
        }
        while outcome["nothing"]:
            # Compute action from trained JESSI
            action, _, _, _, action_distr, perception_distr = policy.act_on_jessi_perception(
                jessi, 
                perception_params, 
                policy_key, 
                obs, 
                info, 
                network_params, 
                jnp.full((jessi.n_detectable_humans,), .3),
                sample=False,
            )
            # # Step the environment (Lasernav)
            state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
            # Save data for animation
            all_states = jnp.vstack((all_states, jnp.array([state])))
            all_observations = jnp.vstack((all_observations, jnp.array([obs])))
            all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
            all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
            all_actions = all_actions.at[step].set(action)
            all_encoder_distrs = tree_map(lambda x, y: x.at[step].set(y), all_encoder_distrs, perception_distr)
            all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
            all_actor_distrs = tree_map(lambda x, y: x.at[step].set(y), all_actor_distrs, action_distr)
            # Increment step
            step += 1
        all_actions = all_actions[:step]
        all_encoder_distrs = tree_map(lambda x: x[:step], all_encoder_distrs)
        all_actor_distrs = tree_map(lambda x: x[:step], all_actor_distrs)
        print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
        policy.animate_socialnav_trajectory(
            all_states[:-1],
            all_actions,
            all_robot_goals[:-1],
            all_humans_radii[:-1],
            env,
            perception_distrs=all_encoder_distrs,
            static_obstacles=all_static_obstacles[:-1],
            action_distrs=all_actor_distrs,
        )