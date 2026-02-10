from jax import random, vmap, lax
import jax.numpy as jnp

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as Reward
from socialjym.policies.dwa import DWA
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 0
n_trials = 50
n_episodes_per_trial = 100
kinematics = 'unicycle'
lidar_n_stack_to_use = 1
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'n_humans': 5,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': 'hybrid_scenario', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 0,
    'reward_function': Reward(robot_radius=0.3),
    'kinematics': kinematics,
    'lidar_noise': False,
}
time_horizons = jnp.array([.25, .5, .75, 1.])
heading_coeffs = jnp.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.])
velocity_coeffs = jnp.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.])
clearence_coeffs = jnp.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.])
distance_coeffs = jnp.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.])

# Initialize the environment
env = LaserNav(**env_params)


success_rates_trials = []
parameter_combinations = []

for trial in range(n_trials):
    print(f"\n\nTrial {trial+1}/{n_trials}")
    # Sample random hyperparameters for this trial
    key = random.PRNGKey(random_seed + trial)
    keys = random.split(key, 5)
    time_horizon = random.choice(keys[0], time_horizons)
    heading_coeff = random.choice(keys[1], heading_coeffs)
    velocity_coeff = random.choice(keys[2], velocity_coeffs)
    clearence_coeff = random.choice(keys[3], clearence_coeffs)
    distance_coeff = random.choice(keys[4], distance_coeffs)
    print(f"Sampled hyperparameters: Time Horizon={time_horizon}, Heading Coeff={heading_coeff}, Velocity Coeff={velocity_coeff}, Clearence Coeff={clearence_coeff}, Distance Coeff={distance_coeff}")
    # Initialize the policy
    policy = DWA(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
        lidar_n_stack_to_use=lidar_n_stack_to_use,
        predict_time_horizon=time_horizon,
        heading_cost_coeff=heading_coeff,
        velocity_cost_coeff=velocity_coeff,
        clearance_cost_coeff=clearence_coeff,
        distance_cost_coeff=distance_coeff,
    )
    # Execute tests
    metrics = policy.evaluate(
        n_episodes_per_trial,
        random_seed,
        env,
    )
    success_rate = metrics['successes'] / n_episodes_per_trial
    success_rates_trials.append(success_rate)
    parameter_combinations.append((time_horizon, heading_coeff, velocity_coeff, clearence_coeff, distance_coeff))

# Find the best hyperparameters
best_index = jnp.argmax(jnp.array(success_rates_trials))
best_parameters = parameter_combinations[best_index]
print(f"Best hyperparameters: Time Horizon={best_parameters[0]}, Heading Coeff={best_parameters[1]}, Velocity Coeff={best_parameters[2]}, Clearence Coeff={best_parameters[3]}, Distance Coeff={best_parameters[4]} with Success Rate={success_rates_trials[best_index]:.2f}")