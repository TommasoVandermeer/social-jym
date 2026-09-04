from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.utils.aux_functions import animate_trajectory

n_noised_trajectories = 100
humans_trajectory_noise_std = 0.1
humans_prediction_horizon = 20
# Hyperparameters
random_seed = 3
robot_vmax = .45
robot_wmax = 1.9
robot_wheel_distance = 2 * robot_vmax / robot_wmax
time_limit = 50
n_episodes = 100
kinematics = 'unicycle'
n_stack_for_action_space_bounding = 1
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 200,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'lidar_dt': 0.13,
    'odometry_dt': 0.05,
    'control_delay_mean': 0.1, 
    'control_delay_sigma': 0.01,
    'wheels_max_linear_acceleration': 0.87,
    'wheels_distance': robot_wheel_distance,
    'n_humans': 5,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': 'hybrid_scenario', 
    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 3,
    'ccso_static_humans_radius_mean': 0.3,
    'ccso_static_humans_radius_std': 0.025,
    'reward_function': Reward(robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
    'kinematics': kinematics,
    'lidar_noise': True,
    'leg_dynamics': True,
    'noisy_walls': True,
    'obstacles_noise': 0.15,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = JESSI_S2R(
    humans_trajectory_noise_std=humans_trajectory_noise_std,
    humans_prediction_horizon=humans_prediction_horizon,
    v_max=robot_vmax,
    wheels_distance=robot_wheel_distance,
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    embedding_dim=32,
)

perception_params, actor_params, critic_params, e2e_params = policy.init_nns(
    random.PRNGKey(random_seed),
)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)

    noised_trajectories, trajectories = vmap(policy.predict_humans_trajectory, in_axes=(0,None,None,None,None,None))(
        random.split(policy_key, n_noised_trajectories),
        state[:-1],
        info["visibility"][:-1, :-1],
        info["humans_goal"],
        info["humans_parameters"],
        info["static_obstacles"][:-1]
    )
    trajectory = trajectories[0] # They are all the same without noise

    lidar_measurements = obs[0,11:]  # Shape: (lidar_num_rays)
    lidar_angles = policy.lidar_angles_robot_frame + obs[0,2]  # Shape: (lidar_num_rays)
    xs = lidar_measurements * jnp.cos(lidar_angles) + obs[0,0]
    ys = lidar_measurements * jnp.sin(lidar_angles) + obs[0,1]
    points = jnp.stack((xs, ys), axis=-1)  # Shape: (lidar_num_rays, 2)
    action_space_parameters = policy.bound_action_space(points)
    
    value = policy.critic_forward(
        policy_key,
        critic_params,
        state,
        obs[:policy.n_actions_history,6:8],
        {
            "humans_goal": info["humans_goal"],
            "humans_visibility": info["visibility"][:-1, :-1],
            "humans_parameters": info["humans_parameters"],
            "static_obstacles": info["static_obstacles"],
        }, # env_params
        {
            "robot_goal": info["robot_goal"],
            "robot_radius": policy.robot_radius,
            "v_max": policy.v_max,
            "wheels_distance": policy.wheels_distance,
            "wheels_max_linear_acceleration": env.wheels_max_linear_acceleration,
            "robot_delay": info["robot_delay"],
        }, # robot_params
        action_space_parameters
    )
    print("Critic value: ", value)

    # Print statistics on noised trajectories
    print("Max linear velocity norm: [NOISED] ", jnp.max(jnp.linalg.norm(noised_trajectories[:,:,:,2:4], axis=3)), "[PURE] ", jnp.max(jnp.linalg.norm(trajectories[:,:,:,2:4], axis=3)))
    print("Min linear velocity norm: [NOISED] ", jnp.min(jnp.linalg.norm(noised_trajectories[:,:,:,2:4], axis=3)), "[PURE] ", jnp.min(jnp.linalg.norm(trajectories[:,:,:,2:4], axis=3)))
    print("Max angular velocity norm: [NOISED] ", jnp.max(noised_trajectories[:,:,:,4]), "[PURE] ", jnp.max(trajectories[:,:,:,4]))
    print("Min angular velocity norm: [NOISED] ", jnp.min(noised_trajectories[:,:,:,4]), "[PURE] ", jnp.min(trajectories[:,:,:,4]), "\n\n")

    # Plot state + predicted noised trajectories
    figure, ax = plt.subplots(1,1,figsize=(10,10))
    ax.clear()
    ax.set(xlim=[-10,10], ylim=[-10,10])
    ax.set_xlabel('X', labelpad=-5)
    ax.set_ylabel('Y', labelpad=-16)
    ax.set_aspect('equal', adjustable='datalim')
    for h in range(env.n_humans):
        head = plt.Circle((state[h,0] + jnp.cos(state[h,4]) * info["humans_parameters"][h,0], state[h,1] + jnp.sin(state[h,4]) * info["humans_parameters"][h,0]), 0.1, color='black', alpha=0.7, zorder=1)
        ax.add_patch(head)
        body = plt.Circle((state[h,0], state[h,1]), info["humans_parameters"][h,0], edgecolor='black', facecolor='white', alpha=0.7, fill=True, zorder=1)
        ax.add_patch(body)
        ax.plot(trajectory[:,h,0], trajectory[:,h,1], color='red', zorder=100)
        for n in range(n_noised_trajectories):
            ax.plot(noised_trajectories[n,:,h,0], noised_trajectories[n,:,h,1], color='blue', zorder=99)
    head = plt.Circle((state[-1,0] + policy.robot_radius * jnp.cos(state[-1,4]), state[-1,1] + policy.robot_radius * jnp.sin(state[-1,4])), 0.1, color='black', zorder=1)
    ax.add_patch(head)
    body = plt.Circle((state[-1,0], state[-1,1]), policy.robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
    ax.add_patch(body)
    ax.plot(
        info["robot_goal"][0],
        info["robot_goal"][1],
        marker='*',
        markersize=7,
        color='red',
        zorder=5,
    )
    if info["static_obstacles"][-1].shape[1] > 1: # Polygon obstacles
        for o in info["static_obstacles"][-1]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in info["static_obstacles"][-1]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    plt.show()
