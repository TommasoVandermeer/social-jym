from jax import random, vmap, jit, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
# from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.utils.rewards.lasernav_rewards.reward4 import Reward4 as Reward
from socialjym.policies.jessi_mppi import JESSI_MPPI
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 3
num_samples = 100
robot_vmax = 1
robot_wheel_distance = 0.7
time_limit = 50
n_episodes = 100
kinematics = 'unicycle'
n_stack_for_action_space_bounding = 1
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 200,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    # 'lidar_dt': 0.13,
    # 'odometry_dt': 0.05,
    # 'control_delay_mean': 0.1, 
    # 'control_delay_sigma': 0.01,
    # 'wheels_max_linear_acceleration': 1.8, #0.87,
    'wheels_distance': robot_wheel_distance,
    'n_humans': 5,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': 'hybrid_scenario', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 5,
    'ccso_static_humans_radius_mean': 0.34,
    'ccso_static_humans_radius_std': 0.025,
    'reward_function': Reward(robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
    'kinematics': kinematics,
    'lidar_noise': True,
    # 'leg_dynamics': True,
    # 'noisy_walls': True,
    # 'obstacles_noise': 0.15,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = JESSI_MPPI(
    num_samples=num_samples,
    v_max=robot_vmax,
    wheels_distance=robot_wheel_distance,
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    embedding_dim=32,
)

with open(os.path.join(os.path.dirname(__file__), 'jessi_multitask_rl_out_32.pkl'), 'rb') as f:
    network_params, _, _ = pickle.load(f)
perception_params, actor_critic_params =policy.split_nns_params(network_params)

# _, _, network_params = policy.init_nns(random.PRNGKey(random_seed))

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    step = 0
    max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
    all_states = jnp.array([state])
    all_intermediate_states = jnp.zeros((max_steps, int(env.robot_dt/env.humans_dt), state.shape[0], state.shape[1]))
    all_observations = jnp.array([obs])
    all_robot_goals = jnp.array([info['robot_goal']])
    all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
    all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
    all_actions = jnp.zeros((max_steps, 2))
    all_rewards = jnp.zeros((max_steps,))
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
    all_spatial_attentions = jnp.zeros((max_steps, policy.lidar_num_rays))
    all_temporal_attentions = jnp.zeros((max_steps, policy.n_stack))
    all_predicted_state_values = jnp.zeros((max_steps,))
    all_actor_distrs = {
        'alphas': jnp.zeros((max_steps, 3)),
        'vertices': jnp.zeros((max_steps, 3, 2)),
    }
    all_human_attentions = jnp.zeros((max_steps, policy.n_detectable_humans))
    if env.leg_dynamics:
        all_humans_leg_radii = jnp.array([info['humans_leg_parameters'][:,-1]])
        all_humans_leg_states = jnp.array([info['humans_leg_state']])
    all_trajectories = jnp.zeros((max_steps, policy.num_samples, policy.horizon+1, 3))
    all_trajectories_costs = jnp.zeros((max_steps,policy.num_samples))
    all_u_means = jnp.zeros((max_steps, policy.horizon, 2))
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, u_mean, trajectories, costs, perception_distr, actor_distr, state_value, spatial_attn, temporal_attn, human_attn, key = policy.act(
            policy_key, 
            obs, 
            info, 
            perception_params,
            actor_critic_params,
        )
        # Debug prints
        print(
            f"Action: {action}",
        )
        # Step the environment
        state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Save data for animation
        all_actions = all_actions.at[step].set(action)
        all_u_means = all_u_means.at[step].set(u_mean)
        all_rewards = all_rewards.at[step].set(reward)
        all_predicted_state_values = all_predicted_state_values.at[step].set(state_value)
        all_actor_distrs = tree_map(lambda x, y: x.at[step].set(y), all_actor_distrs, actor_distr)
        all_encoder_distrs = tree_map(lambda x, y: x.at[step].set(y), all_encoder_distrs, perception_distr)
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_intermediate_states = all_intermediate_states.at[step].set(info["intermediate_states"])
        all_observations = jnp.vstack((all_observations, jnp.array([obs])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
        all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
        all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
        all_spatial_attentions = all_spatial_attentions.at[step].set(spatial_attn[0])
        all_temporal_attentions = all_temporal_attentions.at[step].set(temporal_attn[0])
        all_human_attentions = all_human_attentions.at[step].set(human_attn[0])
        if env.leg_dynamics:
            all_humans_leg_radii = jnp.vstack((all_humans_leg_radii, jnp.array([info['humans_leg_parameters'][:,-1]])))
            all_humans_leg_states = jnp.vstack((all_humans_leg_states, jnp.array([info['humans_leg_state']])))
        all_trajectories = all_trajectories.at[step].set(trajectories)
        all_trajectories_costs = all_trajectories_costs.at[step].set(costs)
        # Increment step
        step += 1
    all_encoder_distrs = tree_map(lambda x: x[:step], all_encoder_distrs)
    all_actor_distrs = tree_map(lambda x: x[:step], all_actor_distrs)
    all_intermediate_states = all_intermediate_states[:step]
    all_actions = all_actions[:step]
    all_u_means = all_u_means[:step]
    all_rewards = all_rewards[:step]
    all_spatial_attentions = all_spatial_attentions[:step]
    all_temporal_attentions = all_temporal_attentions[:step]
    all_human_attentions = all_human_attentions[:step]
    all_predicted_state_values = all_predicted_state_values[:step]
    all_trajectories = all_trajectories[:step]
    all_trajectories_costs = all_trajectories_costs[:step]
    # Print outcome and return
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0], " - Return: {:.2f}".format(info['return']))
    ## Animate trajectory with JESSI's perception and action distribution
    policy.animate_lasernav_trajectory(
        env,
        all_states[:-1],
        all_humans_leg_states[:-1] if env.leg_dynamics else None,
        all_observations[:-1],
        all_actions,
        all_actor_distrs,
        humans_distrs=all_encoder_distrs,
        goals=all_robot_goals[:-1],
        static_obstacles=all_static_obstacles[:-1],
        humans_radii=all_humans_radii[:-1],
        humans_leg_radii=all_humans_leg_radii[:-1] if env.leg_dynamics else None,
        spatial_attentions=all_spatial_attentions,
        temporal_attentions=all_temporal_attentions,
        human_attentions=all_human_attentions,
        trajectories=all_trajectories,
        trajectories_costs=all_trajectories_costs,
        control_sequences=all_u_means,
    )