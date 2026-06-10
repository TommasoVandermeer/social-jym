from jax import random, vmap
from jax import jit, lax, tree_map
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

custom_obstacles_dir = os.path.join(os.path.dirname(__file__), "san_niccolo_socialjym.pkl")
with open(custom_obstacles_dir, 'rb') as f:
    custom_obstacles = pickle.load(f)
    print("Custom obstacles shape: ", custom_obstacles.shape)

# Hyperparameters
random_seed = 0
robot_vmax = 1
robot_wheel_distance = 0.7
time_limit = 50
n_episodes = 100
kinematics = 'unicycle'
n_stack_for_action_space_bounding = 1
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'lidar_dt': 0.13,
    'odometry_dt': 0.05,
    'control_delay_mean': 0.1, 
    'control_delay_sigma': 0.01,
    'wheels_max_linear_acceleration': 1.8, #0.87,
    'wheels_distance': robot_wheel_distance,
    'n_humans': 1,
    'n_obstacles': custom_obstacles.shape[0],
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': None, 
    'reward_function': Reward(robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
    'kinematics': kinematics,
    'lidar_noise': True,
    'leg_dynamics': True,
}
custom_obstacles = jnp.repeat(custom_obstacles[jnp.newaxis], env_params['n_humans']+1, axis=0) # (n_humans+1, n_obstacles, 1, 2, 2)

# Generate custom_episode_dict
# custom_episode: dictionary with keys:
#             full_state (n_humans+1, 6): initial full state. WARNING: humans' velocities
#                 must be given in the GLOBAL frame; they are converted to the body frame
#                 here since LaserNav humans are driven by HSFM.
#             humans_goal (n_humans, 2): humans' goal positions.
#             robot_goal (2,): robot's goal position.
#             static_obstacles (n_humans+1, n_obstacles, 1, 2, 2): static obstacles.
#             scenario (int): scenario index (use -1 for custom scenario).
#             humans_radius (n_humans,): humans' radii.
#             humans_speed (n_humans,): humans' desired speeds.
custom_episode = {
    'full_state': jnp.array([[10_000., 10_000., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]),
    'humans_goal': jnp.array([[10_000., 10_000.]]),
    'robot_goal': jnp.array([2., -1.]),
    'static_obstacles': custom_obstacles,
    'scenario': -1,
    'humans_radius': jnp.array([0.3]),
    'humans_speed': jnp.array([1.]),
}

# Initialize and reset environment
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
)
# with open(os.path.join(os.path.dirname(__file__), 'realistic_pre_perception_network.pkl'), 'rb') as f:
#     encoder_params = pickle.load(f)
# with open(os.path.join(os.path.dirname(__file__), 'realistic_pre_controller_network.pkl'), 'rb') as f:
#     actor_params = pickle.load(f)
# network_params = policy.merge_nns_params(encoder_params, actor_params)

with open(os.path.join(os.path.dirname(__file__), 'jessi_policy_rl_out.pkl'), 'rb') as f:
    network_params, _, _ = pickle.load(f)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset_custom_episode(reset_key, custom_episode)
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
        # Debug prints
        print(
            f"Step {step} - Goal: {info['robot_goal']}", "\n",
            "Dirichlet distribution parameters: ", actor_distr['alphas'],"\n",
            "Control-sensors delay: ", f"{info['substeps_from_last_scan'] * env.humans_dt:.2f} s","\n",
            "Sensors-sensors delay: ", f"{(info['substeps_from_last_odom_ref_scan'] - info['substeps_from_last_scan']) * env.humans_dt:.2f} s","\n",
        )
        # Step the environment
        state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
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
    @jit
    def _discounted_cumsum(rewards):
        def scan_fun(carry, reward):
            new_carry = reward + carry * jnp.power(0.99, policy.dt * policy.v_max)
            return new_carry, new_carry
        _, discounted_cumsums = lax.scan(scan_fun, 0.0, rewards[::-1])
        return discounted_cumsums[::-1]
    discounted_returns = _discounted_cumsum(all_rewards)
    # [print("Step {} -  critic prediction: {:.2f} VS discounted return: {:.2f}".format(i, all_predicted_state_values[i], discounted_returns[i])) for i in range(len(discounted_returns))]
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0], " - Return: {:.2f}".format(info['return']))
    ## Animate only trajectory
    angles = vmap(lambda robot_yaw: jnp.linspace(robot_yaw - env.lidar_angular_range/2, robot_yaw + env.lidar_angular_range/2, env.lidar_num_rays))(all_states[:,-1,4])
    lidar_measurements = vmap(lambda mes, ang: jnp.stack((mes, ang), axis=-1))(all_observations[:,0,9:], angles)
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        'hsfm',
        info['robot_goal'],
        None,
        static_obstacles=info['static_obstacles'][-1],
        robot_dt=env_params['robot_dt'],
        # lidar_measurements=lidar_measurements,
        kinematics=kinematics,
    )
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