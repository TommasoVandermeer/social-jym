from jax import random, vmap, jit, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
# from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.utils.rewards.lasernav_rewards.reward4 import Reward4 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 3
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
    'ccso_n_static_humans': 0,
    'reward_function': Reward(robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
    'kinematics': kinematics,
    'lidar_noise': True,
    # 'leg_dynamics': True,
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
# with open(os.path.join(os.path.dirname(__file__), 'realistic_pre_perception_network_32.pkl'), 'rb') as f:
#     encoder_params = pickle.load(f)
# with open(os.path.join(os.path.dirname(__file__), 'realistic_pre_controller_network_32.pkl'), 'rb') as f:
#     actor_params = pickle.load(f)
# network_params = policy.merge_nns_params(encoder_params, actor_params)

with open(os.path.join(os.path.dirname(__file__), 'jessi_multitask_rl_out_32.pkl'), 'rb') as f:
    network_params, _, _ = pickle.load(f)

# _, _, network_params = policy.init_nns(random.PRNGKey(random_seed))

# Test the trained JESSI policy
# metrics = policy.evaluate(
#     n_episodes,
#     random_seed,
#     env,
#     network_params,
# )

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
        # Debug prints
        print(
            # "Dirichlet distribution parameters: ", actor_distr['alphas'],"\n",
            # "Info['previous_obs']: ", info["previous_obs"],"\n",
            # "Predicted HCGs scores", [f"{w:.2f}" for w in perception_distr['weights']],"\n",
            # "Substeps from last scan: ", info["substeps_from_last_scan"],"\n",
            # "Substeps from last odom (ref scan): ", info["substeps_from_last_odom_ref_scan"],"\n",
            "Control-sensors delay: ", f"{info['substeps_from_last_scan'] * env.humans_dt:.2f} s","\n",
            "Sensors-sensors delay: ", f"{(info['substeps_from_last_odom_ref_scan'] - info['substeps_from_last_scan']) * env.humans_dt:.2f} s","\n",
            f"Lasers dt (stack): {obs[0,8] - obs[:,6]}",
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
    # angles = vmap(lambda robot_yaw: jnp.linspace(robot_yaw - env.lidar_angular_range/2, robot_yaw + env.lidar_angular_range/2, env.lidar_num_rays))(all_states[:,-1,4])
    # lidar_measurements = vmap(lambda mes, ang: jnp.stack((mes, ang), axis=-1))(all_observations[:,0,9:], angles)
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
    ## Plot velocity dynamics of the simulator with respect to reference commands
    robot_velocities = all_intermediate_states[:,:,-1,2:4].reshape(-1, 2)
    robot_velocities_times = jnp.arange(0, robot_velocities.shape[0]*env.humans_dt, env.humans_dt)
    reference_velocities = all_actions
    reference_velocities_times = jnp.arange(0, reference_velocities.shape[0]*env.robot_dt, env.robot_dt)
    figure, ax = plt.subplots(2, 1, figsize=(15, 10))
    figure.subplots_adjust(left=0.1, right=0.9, bottom=0.15)
    ax[0].step(reference_velocities_times, reference_velocities[:,0], label='Action', where='post', color='green', alpha=0.7)
    ax[0].plot(robot_velocities_times, robot_velocities[:,0], label='Velocity', color='red', linewidth=2)
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Linear Vel [m/s]')
    ax[0].legend(fontsize=16)
    ax[1].step(reference_velocities_times, reference_velocities[:,1], label='Action', where='post', color='green', alpha=0.7)
    ax[1].plot(robot_velocities_times, robot_velocities[:,1], label='Velocity', color='red', linewidth=2)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Angular Vel [m/s]')
    ax[1].legend(fontsize=16)
    plt.show()
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