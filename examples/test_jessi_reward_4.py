from jax import random, vmap, jit, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
from socialjym.envs.base_env import wrap_angle
from socialjym.utils.rewards.lasernav_rewards.reward4 import Reward4 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory
from jhsfm.hsfm import get_linear_velocity

plot = {"risk":False, "escape":True}
# Hyperparameters
random_seed = 3
visibility_chance=0.1
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
    'lidar_dt': 0.13,
    'odometry_dt': 0.05,
    'control_delay_mean': 0.1, 
    'control_delay_sigma': 0.01,
    'wheels_max_linear_acceleration': 1.8, #0.87,
    'wheels_distance': robot_wheel_distance,
    'n_humans': 10,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': None,
    'scenario': 'parallel_traffic', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 10,
    'ccso_static_humans_radius_mean': 0.3,
    'ccso_static_humans_radius_std': 0.025,
    'reward_function': Reward(gamma=[0.95, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
    'kinematics': kinematics,
    'lidar_noise': True,
    'leg_dynamics': True,
    'noisy_walls': True,
    'obstacles_noise': 0.15,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = JESSI(
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

# Plot util
def plot_base(old_state, info):
    figure, ax = plt.subplots(1, 1, figsize=(10, 10))
    figure.subplots_adjust(left=0.1, right=0.9, bottom=0.15)
    ax.set_aspect('equal', adjustable='datalim')
    # Plot humans
    humans_poses = old_state[:-1,[0,1,4]]
    color = 'blue'
    alpha = 0.6 
    for h in range(len(humans_poses)):
        head = plt.Circle((humans_poses[h,0] + jnp.cos(humans_poses[h,2]) * info["humans_parameters"][h,0], humans_poses[h,1] + jnp.sin(humans_poses[h,2]) * info["humans_parameters"][h,0]), 0.1, color='black', alpha=alpha, zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((humans_poses[h,0], humans_poses[h,1]), info["humans_parameters"][h,0], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
        ax.add_patch(circle)
    # Plot robot
    robot_position = old_state[-1,:2]
    head = plt.Circle((robot_position[0] + env.robot_radius * jnp.cos(old_state[-1,4]), robot_position[1] + env.robot_radius * jnp.sin(old_state[-1,4])), 0.1, color='black', zorder=3)
    ax.add_patch(head)
    circle = plt.Circle((robot_position[0], robot_position[1]), env.robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=4)
    ax.add_patch(circle)
    # Plot robot goal
    ax.plot(
        info['robot_goal'][0],
        info['robot_goal'][1],
        marker='*',
        markersize=7,
        color='red',
        zorder=5,
    )
    # Plot static obstacles
    if info["static_obstacles"][-1].shape[1] > 1: # Polygon obstacles
        for o in info["static_obstacles"][-1]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in info["static_obstacles"][-1]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    return figure, ax 

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key, visibility_chance=visibility_chance)
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
    all_predicted_state_values = jnp.zeros((max_steps,))
    if policy.ablation_mode == 4:
        all_actor_distrs = {
            'means': jnp.zeros((max_steps, 2)),
            'logsigmas': jnp.zeros((max_steps, 2)),
            'vertices': jnp.zeros((max_steps, 3, 2)),
        }
    elif policy.ablation_mode == 6:
        all_actor_distrs = {
            'locs': jnp.zeros((max_steps, 3)),
            'log_scales': jnp.zeros((max_steps, 3)),
            'vertices': jnp.zeros((max_steps, 3, 2)),
        }
    else:
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
    all_spatial_attentions = jnp.zeros((max_steps, policy.lidar_num_rays))
    all_temporal_attentions = jnp.zeros((max_steps, policy.n_stack))
    all_human_attentions = jnp.zeros((max_steps, policy.n_detectable_humans))
    if env.leg_dynamics:
        all_humans_leg_radii = jnp.array([info['humans_leg_parameters'][:,-1]])
        all_humans_leg_states = jnp.array([info['humans_leg_state']])
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, _, _, _, _, _, perception_distr, actor_distr, state_value, spat_attn, temp_attn, human_attn = policy.act(random.PRNGKey(0), obs, info, network_params, sample=False)
        # Step the environment
        old_state = state
        state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Reward detail
        reward, _, reward_terms, reward_info = env.reward_function(old_state, info['intermediate_states'], action, info, env.robot_dt)
        # Debug prints
        print(
            f"\nStep: {step} - Reward: {reward:.2f} - Risk reward: {reward_terms[0.99]:.3f} - Escape reward: {reward_terms[0.95]:.3f} \
            \nClearance distance: {reward_info['clearance_distance']:.3f} - Next clearance distance: {reward_info['next_clearance_distance']:.3f} \
            \nCurrent max risk: {reward_info['current_max_risk']:.3f} - Next max risk: {reward_info['next_max_risk']:.3f} \
            \nCurrent mean risk: {reward_info['current_mean_risk']:.3f} - Next mean risk: {reward_info['next_mean_risk']:.3f} \
            \nAngular speed: {action[1]:.3f} - Forward bound: {reward_info['action_space_parameters'][0]:.1f} - Left bound {reward_info['action_space_parameters'][1]:.1f} - Right bound {reward_info['action_space_parameters'][2]:.1f} \
            \nEscape turn: {reward_info['escape_turn_reward']:.3f} - Escape stall: {reward_info['escape_stall_penalty']:.3f} - Escape switch {reward_info['escape_switch_penalty']:.3f}"
        )
        # Plot RISK state
        plot_risk_state = (plot["risk"]) & ((jnp.abs(reward_terms[0.99]) > 0.0) | (random.bernoulli(random.PRNGKey(info["step"]), 0.1))) 
        if plot_risk_state:
            print("\nPLOTTING RISK STATE\n")
            _, ax = plot_base(old_state, info)
            # Plot risk horizon evaluation states
            if reward_info["evaluation_states"] is not None:
                for i, s in enumerate(reward_info["evaluation_states"]):
                    circle = plt.Circle((s[-1,0], s[-1,1]), env.robot_radius, edgecolor="green", fill=False, zorder=2, alpha=0.3)
                    ax.add_patch(circle)
                    head = plt.Circle((s[-1,0] + env.robot_radius * jnp.cos(s[-1,4]), s[-1,1] + env.robot_radius * jnp.sin(s[-1,4])), 0.1, color='darkgreen', zorder=2, alpha=0.3)
                    ax.add_patch(head)
                    for h in range(s.shape[0]-1):
                        if i == reward_info["closest_distance_time_index"][h]:
                            circle = plt.Circle((s[h,0], s[h,1]), info["humans_parameters"][h,0], edgecolor="green", fill=False, zorder=2, alpha=0.3)
                            ax.add_patch(circle)
                for i, s in enumerate(reward_info["next_evaluation_states"]):
                    circle = plt.Circle((s[-1,0], s[-1,1]), env.robot_radius, edgecolor="red", fill=False, zorder=2, alpha=0.3)
                    ax.add_patch(circle)
                    head = plt.Circle((s[-1,0] + env.robot_radius * jnp.cos(s[-1,4]), s[-1,1] + env.robot_radius * jnp.sin(s[-1,4])), 0.1, color='darkred', zorder=2, alpha=0.3)
                    ax.add_patch(head)
                    for h in range(s.shape[0]-1):
                        if i == reward_info["next_closest_distance_time_index"][h]:
                            circle = plt.Circle((s[h,0], s[h,1]), info["humans_parameters"][h,0], edgecolor="red", fill=False, zorder=2, alpha=0.3)
                            ax.add_patch(circle)
            plt.show()
        # Plot ESCAPE state
        plot_escape_state = (plot["escape"]) & ((jnp.abs(reward_terms[0.95]) > 0.0) | (random.bernoulli(random.PRNGKey(info["step"]), 0.05))) 
        if plot_escape_state:
            print("\nPLOTTING ESCAPE STATE\n")
            _, ax = plot_base(old_state, info)
            # Plot robot with constant speed
            robot_pos = old_state[-1,:2]
            robot_yaw = old_state[-1,4]
            robot_velocity_unicycle = old_state[-1,2:4]
            next_robot_pos = lax.cond(
                jnp.abs(robot_velocity_unicycle[1]) > 1e-3,
                lambda x: x.at[:].set(jnp.array([
                    x[0] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.sin(robot_yaw + robot_velocity_unicycle[1]) - jnp.sin(robot_yaw)),
                    x[1] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.cos(robot_yaw) - jnp.cos(robot_yaw + robot_velocity_unicycle[1]))
                ])),
                lambda x: x.at[:].set(jnp.array([
                    x[0] + robot_velocity_unicycle[0] * jnp.cos(robot_yaw),
                    x[1] + robot_velocity_unicycle[0] * jnp.sin(robot_yaw)
                ])),
                robot_pos)
            next_robot_yaw = wrap_angle(robot_yaw + robot_velocity_unicycle[1])
            circle = plt.Circle((next_robot_pos[0], next_robot_pos[1]), env.robot_radius, edgecolor="green", fill=False, zorder=2, alpha=0.3)
            ax.add_patch(circle)
            head = plt.Circle((next_robot_pos[0] + env.robot_radius * jnp.cos(next_robot_yaw), next_robot_pos[1] + env.robot_radius * jnp.sin(next_robot_yaw)), 0.1, color='darkgreen', zorder=2, alpha=0.3)
            ax.add_patch(head)
            # Plot robot after action application
            robot_pos = old_state[-1,:2]
            robot_yaw = old_state[-1,4]
            robot_velocity_unicycle = state[-1,2:4]
            next_robot_pos = lax.cond(
                jnp.abs(robot_velocity_unicycle[1]) > 1e-3,
                lambda x: x.at[:].set(jnp.array([
                    x[0] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.sin(robot_yaw + robot_velocity_unicycle[1]) - jnp.sin(robot_yaw)),
                    x[1] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.cos(robot_yaw) - jnp.cos(robot_yaw + robot_velocity_unicycle[1]))
                ])),
                lambda x: x.at[:].set(jnp.array([
                    x[0] + robot_velocity_unicycle[0] * jnp.cos(robot_yaw),
                    x[1] + robot_velocity_unicycle[0] * jnp.sin(robot_yaw)
                ])),
                robot_pos)
            next_robot_yaw = wrap_angle(robot_yaw + robot_velocity_unicycle[1])
            circle = plt.Circle((next_robot_pos[0], next_robot_pos[1]), env.robot_radius, edgecolor="red", fill=False, zorder=2, alpha=0.3)
            ax.add_patch(circle)
            head = plt.Circle((next_robot_pos[0] + env.robot_radius * jnp.cos(next_robot_yaw), next_robot_pos[1] + env.robot_radius * jnp.sin(next_robot_yaw)), 0.1, color='darkred', zorder=2, alpha=0.3)
            ax.add_patch(head)
            plt.show()
        # Save data for animation
        all_actions = all_actions.at[step].set(action)
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
        all_spatial_attentions = all_spatial_attentions.at[step].set(spat_attn[0])
        all_temporal_attentions = all_temporal_attentions.at[step].set(temp_attn[0])
        all_human_attentions = all_human_attentions.at[step].set(human_attn[0])
        if env.leg_dynamics:
            all_humans_leg_radii = jnp.vstack((all_humans_leg_radii, jnp.array([info['humans_leg_parameters'][:,-1]])))
            all_humans_leg_states = jnp.vstack((all_humans_leg_states, jnp.array([info['humans_leg_state']])))
        # Increment step
        step += 1
    all_encoder_distrs = tree_map(lambda x: x[:step], all_encoder_distrs)
    all_actor_distrs = tree_map(lambda x: x[:step], all_actor_distrs)
    all_intermediate_states = all_intermediate_states[:step]
    all_actions = all_actions[:step]
    all_rewards = all_rewards[:step]
    all_spatial_attentions = all_spatial_attentions[:step]
    all_temporal_attentions = all_temporal_attentions[:step]
    all_human_attentions = all_human_attentions[:step]
    all_predicted_state_values = all_predicted_state_values[:step]
    ## Check predicted state values and actual discounted returns
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0], " - Return: {:.2f}".format(info['return']))
    ## Animate trajectory with JESSI's perception and action distribution
    policy.animate_lasernav_trajectory(
        env,
        all_states[:-1],
        all_humans_leg_states[:-1] if env.leg_dynamics else None,
        all_observations[:-1],
        all_actions,
        all_actor_distrs,
        all_encoder_distrs,
        all_robot_goals[:-1],
        all_static_obstacles[:-1],
        all_humans_radii[:-1],
        all_humans_leg_radii[:-1] if env.leg_dynamics else None,
        spatial_attentions=all_spatial_attentions,
        temporal_attentions=all_temporal_attentions,
        human_attentions=all_human_attentions,
    )