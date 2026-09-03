import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import sys
import argparse
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.policies.jessi_s2r import JESSI_S2R

def main(args=None):
    parser = argparse.ArgumentParser(description='JESSI Animate Experiment')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parsed_args, unknown_args = parser.parse_known_args(sys.argv)

    save_file_name = parsed_args.save_file

    file_path = os.path.join(os.path.dirname(__file__), save_file_name)
    with open(file_path, 'rb') as f:
        experiment_data = pickle.load(f)
    recorded_params = experiment_data['params']

    n_stack = recorded_params['n_stack']
    lidar_num_rays = recorded_params['lidar_num_rays']
    lidar_min_angle = -jnp.pi
    lidar_max_angle = jnp.pi
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
        'robot_dt': 1.0 / recorded_params.get('frequency', 4.0),
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

    policy_class = JESSI_S2R if recorded_params.get('planner') == 'JESSI-S2R' else JESSI
    policy_kwargs = dict(
        v_max=experiment_data['params']['v_max'],
        wheels_distance=experiment_data['params']['wheels_distance'],
        robot_radius=experiment_data['params']['robot_radius'],
        n_stack=experiment_data['params']['n_stack'],
        lidar_num_rays=experiment_data['params']['lidar_num_rays'],
        lidar_angular_range=experiment_data['params']['lidar_angular_range'],
        lidar_max_dist=experiment_data['params']['lidar_max_dist'],
        n_stack_for_action_space_bounding=experiment_data['params']['n_stack_for_action_space_bounding'],
    )
    if policy_class is JESSI_S2R:
        policy_kwargs.update(
            n_actions_history=recorded_params.get('n_actions_history', n_stack),
            embedding_dim=recorded_params.get('embedding_dim', 32),
            n_sectors=recorded_params.get('n_sectors', 60),
        )
    jessi = policy_class(**policy_kwargs)

    if os.path.isfile(os.path.join(os.path.dirname(__file__), f"lists_{save_file_name}")):
        print(f"Found lists_{save_file_name} - Loading recorded scan, odom and cmd lists for visualization.")
        lists_file_path = os.path.join(os.path.dirname(__file__), f"lists_{save_file_name}")
        with open(lists_file_path, 'rb') as f:
            lists_data = pickle.load(f)
        # recorded_scan_list = lists_data['scan'] #UNUSED for now
        recorded_odom_list = lists_data['odom']
        recorded_cmd_list = lists_data['cmd']

        print("\n📊 --- TRACKING ANALYSIS: CMDs vs ODOMs ---")
        odom_times = []
        odom_v = []
        odom_w = []
        cmd_times = []
        cmd_v = []
        cmd_w = []
        for msg in recorded_odom_list:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            odom_times.append(t)
            odom_v.append(msg.twist.twist.linear.x)
            odom_w.append(msg.twist.twist.angular.z)
        for msg in recorded_cmd_list:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cmd_times.append(t)
            cmd_v.append(msg.twist.linear.x)
            cmd_w.append(msg.twist.angular.z)
        t_start = min(odom_times[0], cmd_times[0])
        odom_times_norm = [t - t_start for t in odom_times]
        cmd_times_norm  = [t - t_start for t in cmd_times]
        odom_times_norm = jnp.array(odom_times_norm, dtype=jnp.float32)
        cmd_times_norm  = jnp.array(cmd_times_norm, dtype=jnp.float32)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.step(cmd_times_norm, cmd_v, label='Cmd Lineare (v)', color='red', linewidth=2, where='post')
        ax1.plot(odom_times_norm[50:], odom_v[50:], label='Odom Lineare (v)', color='blue', alpha=0.7, linewidth=2)
        ax1.set_ylabel('Velocità [m/s]', fontsize=12)
        ax1.set_title('Inseguimento Velocità Lineare', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')
        ax2.step(cmd_times_norm, cmd_w, label='Cmd Angolare (w)', color='orange', linewidth=2, where='post')
        ax2.plot(odom_times_norm[50:], odom_w[50:], label='Odom Angolare (w)', color='green', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Tempo [s]', fontsize=12)
        ax2.set_ylabel('Velocità [rad/s]', fontsize=12)
        ax2.set_title('Inseguimento Velocità Angolare', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    trajectory = experiment_data['trajectory']
    T = len(trajectory)
    print(f"Found {T} steps.")

    all_observations = jnp.array([step['observation'] for step in trajectory])
    all_actions = jnp.array([step['action'] for step in trajectory])
    all_robot_goals = jnp.array([step['robot_goal'] for step in trajectory])
    all_encoder_distrs = tree_map(lambda *xs: jnp.stack(xs), *[step['perception_distr'] for step in trajectory])
    all_actor_distrs = tree_map(lambda *xs: jnp.stack(xs), *[step['actor_distr'] for step in trajectory])
    if 'spatial_attention' in trajectory[0]:
        all_spatial_attentions = jnp.array([step['spatial_attention'] for step in trajectory])
    else:
        all_spatial_attentions = None
    if 'temporal_attention' in trajectory[0]:
        all_temporal_attentions = jnp.array([step['temporal_attention'] for step in trajectory])
    else:
        all_temporal_attentions = None
    if 'human_attention' in trajectory[0]:
        all_human_attentions = jnp.array([step['human_attention'] for step in trajectory])  
    else:
        all_human_attentions = None

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
        spatial_attentions=all_spatial_attentions,
        temporal_attentions=all_temporal_attentions,
        human_attentions=all_human_attentions,
    )

if __name__ == '__main__':
    main()
