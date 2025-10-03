from jax import random, lax
import jax.numpy as jnp
import numpy as np
import time
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.aux_functions import interpolate_humans_boundaries, interpolate_obstacle_segments

try:
    import dwa
except ImportError:
    raise ImportError("DWA package is not installed. Please install it to use this function.\nYou can install it with 'pip3 install dynamic-window-approach'.\n Checkout https://github.com/goktug97/DynamicWindowApproach")

### Hyperparameters
random_seed = 0 
n_inferences = 1000
n_humans_large_scenario = 30
v_max = 1

### Initialize environment with small scenarios
env_params = {
    'robot_radius': 0.3,
    'n_humans': 3,
    'n_obstacles': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'perpendicular_traffic',
    'hybrid_scenario_subset': jnp.array([0, 1, 2, 3, 4, 6]), # All scenarios but circular_crossing_with_static_obstacles
    'humans_policy': 'hsfm',
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
}
env = SocialNav(**env_params)

### Initialize DIR-SAFE
policy = DIRSAFE(DummyReward(kinematics='unicycle'), v_max=v_max, dt=env_params['robot_dt'])
_, _, obs, info, _ = env.reset(random.PRNGKey(0))
actor_params, critic_params = policy.init_nns(random.PRNGKey(0), obs, info)

### Initialize DWA
dwa_config = dwa.Config(
    max_speed=1,
    min_speed=0.0,
    max_yawrate=2/0.7,
    dt = env.robot_dt,
    max_accel=4,
    max_dyawrate=4,
    predict_time = .5,
    velocity_resolution = 0.1, # Discretization of the velocity space
    yawrate_resolution = np.radians(1.0), # Discretization of the yawrate space
    heading = 0.04,
    clearance = 0.2,
    velocity = 0.2,
    base=[-env.robot_radius, -env.robot_radius, env.robot_radius, env.robot_radius],  # [x_min, y_min, x_max, y_max] in meters
)

### Inferences on small scenarios
reset_key = random.PRNGKey(random_seed)
state, reset_key, obs, info, outcome = env.reset(reset_key)
obstacles_point_cloud = interpolate_obstacle_segments(info["static_obstacles"][-1])
all_times_dir_safe = []
all_times_dwa = []
for i in range(n_inferences):
    ### DIR-SAFE
    time_before_inference_dir_safe = time.time()
    action, _, _, _, distr = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
    all_times_dir_safe.append(time.time() - time_before_inference_dir_safe)
    ### DWA
    # Construct point cloud
    humans_point_cloud = interpolate_humans_boundaries(obs[:-1,:2], info['humans_parameters'][:,0])
    point_cloud = jnp.concatenate((obstacles_point_cloud, humans_point_cloud), axis=0)
    # Step the environment
    time_before_inference_dwa = time.time()
    _ = jnp.array(dwa.planning(tuple(map(float, np.append(obs[-1,:2],obs[-1,5]))), tuple(map(float, obs[-1,2:4])), tuple(map(float, info['robot_goal'])), np.array(point_cloud, dtype=np.float32), dwa_config))
    all_times_dwa.append(time.time() - time_before_inference_dwa)
    ### STEP ENV
    state, obs, info, reward, outcome, reset_key = env.step(state,info,action,test=True,reset_key=reset_key,reset_if_done=True) 
print(f"Average DIR-SAFE inference time over {len(all_times_dir_safe)} inferences (small scenario {len(info['static_obstacles'][0])} obstacles, {env_params['n_humans']} humans)): {np.mean(all_times_dir_safe)*1000:.6f} ms ± {np.std(all_times_dir_safe)*1000:.6f} ms")
print(f"Average DWA inference time over {len(all_times_dwa)} inferences (small scenario {len(info['static_obstacles'][0])} obstacles, {env_params['n_humans']} humans)): {np.mean(all_times_dwa)*1000:.6f} ms ± {np.std(all_times_dwa)*1000:.6f} ms")
### Inferences on larger scenarios
## Load custom episodes
with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans_large_scenario}_humans.pkl'), 'rb') as f:
    custom_episodes = pickle.load(f)
## Initialize environment with larger scenarios
trial = 0
stacked_obstacles = jnp.stack([custom_episodes["static_obstacles"][trial,-1] for _ in range(n_humans_large_scenario+1)], axis=0)  # shape: (n_agents, n_obstacles, 4, 2, 2)
test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans_large_scenario,
    'n_obstacles': len(stacked_obstacles[0]),
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': None, # Custom scenario
    'humans_policy': 'hsfm',
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 0,
}
test_env = SocialNav(**test_env_params)
## Reset the environment
state, _, obs, info, outcome = test_env.reset_custom_episode(
    random.PRNGKey(0), # Not used, but required by the function
    {
        "full_state": custom_episodes["full_state"][trial],
        "robot_goal": custom_episodes["robot_goals"][trial,0],
        "humans_goal": custom_episodes["humans_goal"][trial],
        "static_obstacles": stacked_obstacles,
        "scenario": -1,
        "humans_radius": custom_episodes["humans_radius"][trial],
        "humans_speed": custom_episodes["humans_speed"][trial],
    }
)
obstacles_point_cloud = interpolate_obstacle_segments(info["static_obstacles"][-1])
all_times_dir_safe = []
all_times_dwa = []
while outcome['nothing']:
    ### DIR-SAFE
    time_before_inference_dir_safe = time.time()
    action, _, _, _, distr = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
    all_times_dir_safe.append(time.time() - time_before_inference_dir_safe)
    ### DWA
    # Construct point cloud
    humans_point_cloud = interpolate_humans_boundaries(obs[:-1,:2], info['humans_parameters'][:,0])
    point_cloud = jnp.concatenate((obstacles_point_cloud, humans_point_cloud), axis=0)
    # Step the environment
    time_before_inference_dwa = time.time()
    _ = jnp.array(dwa.planning(tuple(map(float, np.append(obs[-1,:2],obs[-1,5]))), tuple(map(float, obs[-1,2:4])), tuple(map(float, info['robot_goal'])), np.array(point_cloud, dtype=np.float32), dwa_config))
    all_times_dwa.append(time.time() - time_before_inference_dwa)
    ### STEP ENV
    state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)

print(f"Average DIR-SAFE inference time over {len(all_times_dir_safe)} inferences (large scenario {len(stacked_obstacles[0])} obstacles, {n_humans_large_scenario} humans): {np.mean(all_times_dir_safe)*1000:.6f} ms ± {np.std(all_times_dir_safe)*1000:.6f} ms")
print(f"Average DWA inference time over {len(all_times_dwa)} inferences (large scenario {len(stacked_obstacles[0])} obstacles, {n_humans_large_scenario} humans): {np.mean(all_times_dwa)*1000:.6f} ms ± {np.std(all_times_dwa)*1000:.6f} ms")