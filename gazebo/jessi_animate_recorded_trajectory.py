import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import sys
import argparse

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI

def main(args=None):
    parser = argparse.ArgumentParser(description='JESSI Animate Experiment')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parsed_args, unknown_args = parser.parse_known_args(sys.argv)

    save_file_name = parsed_args.save_file

    n_stack = 5
    lidar_num_rays = 100
    lidar_min_angle = -jnp.pi/2
    lidar_max_angle = jnp.pi/2
    lidar_angular_range = lidar_max_angle - lidar_min_angle
    lidar_max_dist = 10

    env_params = {
        'n_stack': n_stack,
        'lidar_num_rays': lidar_num_rays,
        'lidar_angular_range': lidar_angular_range,
        'lidar_max_dist': lidar_max_dist,
        'n_humans': 3,
        'n_obstacles': 5,
        'robot_radius': 0.3,
        'robot_dt': 0.25,
        'humans_dt': 0.01,      
        'robot_visible': True,
        'scenario': 'perpendicular_traffic', 
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]),
        'ccso_n_static_humans': 0,
        'reward_function': Reward(robot_radius=0.3),
        'kinematics': 'unicycle',
        'lidar_noise': True,
    }

    env = LaserNav(**env_params)
    jessi = JESSI(
        v_max=1,
        wheels_distance=.8,
        robot_radius=0.3,
        n_stack=n_stack,
        lidar_num_rays=lidar_num_rays,
        lidar_angular_range=lidar_angular_range,
        lidar_max_dist=lidar_max_dist,
        n_stack_for_action_space_bounding=1
    )

    file_path = os.path.join(os.path.dirname(__file__), save_file_name)
    with open(file_path, 'rb') as f:
        trajectory = pickle.load(f)

    T = len(trajectory)
    print(f"Found {T} steps.")

    all_observations = jnp.array([step['observation'] for step in trajectory])
    all_actions = jnp.array([step['action'] for step in trajectory])
    all_robot_goals = jnp.array([step['robot_goal'] for step in trajectory])
    all_encoder_distrs = tree_map(lambda *xs: jnp.stack(xs), *[step['perception_distr'] for step in trajectory])
    all_actor_distrs = tree_map(lambda *xs: jnp.stack(xs), *[step['actor_distr'] for step in trajectory])

    # ANIMATION
    jessi.animate_lasernav_trajectory(
        env,
        states=None,
        observations=all_observations,
        actions=all_actions,
        actor_distrs=all_actor_distrs,
        humans_distrs=all_encoder_distrs,
        goals=all_robot_goals,
        static_obstacles=None,
        humans_radii=None,
    )

if __name__ == '__main__':
    main()