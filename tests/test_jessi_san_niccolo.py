from jax import random, vmap
import jax.numpy as jnp
import numpy as np
import pickle
import os

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

with_humans = True
custom_obstacles_dir = os.path.join(os.path.dirname(__file__), "san_niccolo_socialjym.pkl")
with open(custom_obstacles_dir, 'rb') as f:
    custom_obstacles = pickle.load(f)
    print("Custom obstacles shape: ", custom_obstacles.shape)

# Hyperparameters
random_seed = 0
robot_vmax = 0.46
robot_wmax = 1.9
robot_wheel_distance = 2*robot_vmax / robot_wmax
time_limit = 600
n_episodes = 100
kinematics = 'unicycle'
n_stack_for_action_space_bounding = 1
if with_humans:
    full_state = jnp.array([
        [-3., 23.61, 0., 0., 0., 0.], # Human 1
        [-3., 22.7, 0., 0., 0., 0.], # Human 2
        # [-6.45, 4.38, 0., 0., 0., 0.], # Human 3
        [-3.77, -9.81, 0., 0., 0., 0.], # Human 4
        [14.69, -10.51, 0., 0., 0., 0.], # Human 5        
        [1.69, -5.69, 0., 0., 0., 0.], # Human 6
        [17.62, 20.306, 0., 0., jnp.pi/2, 0.] # Robot
    ])
    humans_goal = jnp.array([
        [10.5, 23.61], # Human 1
        [10.5, 22.7], # Human 2
        # [-6.45, 18.29], # Human 3
        [16.47, -7.3], # Human 4
        [4.88, -5.57], # Human 5
        [1.93, -10.63], # Human 6
    ])
    humans_radius = jnp.ones((len(humans_goal),)) * 0.3
    humans_speed = jnp.ones((len(humans_goal),)) * 1.
else:
    full_state = jnp.array([[10_000., 10_000., 0., 0., 0., 0.], [17.62, 20.306, 0., 0., jnp.pi/2, 0.]])
    humans_goal = jnp.array([[10_000., 10_000.]])
    humans_radius = jnp.array([0.3])
    humans_speed = jnp.array([1.])
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'lidar_dt': 0.13,
    'odometry_dt': 0.05,
    'control_delay_mean': 0.1, 
    'control_delay_sigma': 0.01,
    'wheels_max_linear_acceleration': 0.87,
    'wheels_distance': robot_wheel_distance,
    'n_humans': len(humans_goal),
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
robot_goals = jnp.array([
    [17.6, 22.3],       # 0
    [14.956, 22.618],   # 1
    [1.503, 22.633],    # 2
    [-5.893, 22.991],   # 3
    [-6.873, 19.245],   # 4
    [-6.873, 8.458],    # 5
    [-6.584, 0.597],    # 6
    [-5.107, -6.143],   # 7
    [4.02, -8.118],     # 8
    [11.00, -8.134],    # 9
    [17.511, -2.1],     # 10
    [17.593, 4.008],    # 11
    [17.593, 12.719],   # 12
    [17.617, 20.306],   # 13
    [17.6, 22.3],       # 14, start again
    [14.956, 22.618],   # 15
    [1.503, 22.633],    # 16
    [-5.893, 22.991],   # 17
    [-6.873, 19.245],   # 18
    [-6.873, 8.458],    # 19
    [-7.162, 3.347],    # 20
    [-6.584, 0.597],    # 21
    [-5.107, -6.143],   # 22
    [4.02, -8.118],     # 23
    [11.00, -8.134],    # 24
    [17.511, -2.1],     # 25
    [17.593, 4.008],    # 26
    [17.593, 12.719],   # 27
    [17.617, 20.306],   # 28
])
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
    'full_state': full_state,
    'humans_goal': humans_goal,
    'robot_goal': robot_goals[0],
    'static_obstacles': custom_obstacles,
    'scenario': -1,
    'humans_radius': humans_radius,
    'humans_speed': humans_speed,
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
with open(os.path.join(os.path.dirname(__file__), 'realistic_jessi_multitask_rl_out_2.pkl'), 'rb') as f:
    network_params, _, _ = pickle.load(f)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset_custom_episode(reset_key, custom_episode)
    step = 0
    max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
    all_states = jnp.array([state])
    all_observations = jnp.array([obs])
    all_robot_goals = jnp.array([info['robot_goal']])
    waypoint_idx = 0
    humans_chase_goal = jnp.ones(env_params['n_humans'], dtype=bool)  # All humans chase their goals initially
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, _, _, _, _, _, perception_distr, actor_distr, state_value, spat_attn, temp_attn, human_attn = policy.act(random.PRNGKey(0), obs, info, network_params, sample=False)
        # Debug prints
        # print(
        #     f"Step {step} - Goal: {info['robot_goal']}", "\n",
        #     "Dirichlet distribution parameters: ", actor_distr['alphas'],"\n",
        #     "Control-sensors delay: ", f"{info['substeps_from_last_scan'] * env.humans_dt:.2f} s","\n",
        #     "Sensors-sensors delay: ", f"{(info['substeps_from_last_odom_ref_scan'] - info['substeps_from_last_scan']) * env.humans_dt:.2f} s","\n",
        # )
        # Update robot goal
        if (waypoint_idx < robot_goals.shape[0] - 1) and (jnp.linalg.norm(state[-1,:2]-info['robot_goal']) < env.robot_radius*2):
            print(f"Waypoint {waypoint_idx} reached! at time {info['time']:.2f}s")
            waypoint_idx += 1
            info['robot_goal'] = info["robot_goal"].at[:].set(robot_goals[waypoint_idx])
        # Update humans goals
        for i in range(len(full_state)-1):
            if jnp.linalg.norm(state[i,:2] - info['humans_goal'][i]) < info['humans_parameters'][i,0]:
                humans_chase_goal = humans_chase_goal.at[i].set(not humans_chase_goal[i])  # Toggle chasing goal
                info['humans_goal'] = info['humans_goal'].at[i].set(humans_goal[i] if humans_chase_goal[i] else full_state[i,:2])   
        # Step the environment
        state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Save data for animation
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_observations = jnp.vstack((all_observations, jnp.array([obs])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
        # Increment step
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0], " - Return: {:.2f}".format(info['return']))
    ## Animate only trajectory
    angles = vmap(lambda robot_yaw: jnp.linspace(robot_yaw - env.lidar_angular_range/2, robot_yaw + env.lidar_angular_range/2, env.lidar_num_rays))(all_states[:,-1,4])
    lidar_measurements = vmap(lambda mes, ang: jnp.stack((mes, ang), axis=-1))(all_observations[:,0,9:], angles)
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        'hsfm',
        all_robot_goals,
        None,
        static_obstacles=info['static_obstacles'][-1],
        robot_dt=env_params['robot_dt'],
        lidar_measurements=lidar_measurements,
        kinematics=kinematics,
    )