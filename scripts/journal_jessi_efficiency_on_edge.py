import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from socialjym.policies.jessi import JESSI
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as Reward

def run_benchmark():
    network_embeddings_dims = [32, 64, 96, 128, 160]
    n_warmup = 5
    n_inferences = 500
    print("Initializing JESSI Benchmark on Raspberry Pi...")
    print(f"Embedding dimensions to test: {network_embeddings_dims}")
    print(f"Iterations per dimension: {n_inferences} (+ {n_warmup} warmup)")
    env_params = {
        'n_stack': 5,
        'lidar_num_rays': 100,
        'lidar_angular_range': jnp.pi * 2,
        'lidar_max_dist': 10.0,
        # 'lidar_dt': 0.13,
        # 'odometry_dt': 0.05,
        # 'control_delay_mean': 0.1,
        # 'control_delay_sigma': 0.01,
        # 'wheels_max_linear_acceleration': 1.8,
        'wheels_distance': 0.7,
        'n_humans': 5,
        'n_obstacles': 5,
        'robot_radius': 0.3,
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': True,
        'scenario': 'hybrid_scenario',
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]),
        'ccso_n_static_humans': 0,
        'reward_function': Reward(robot_radius=0.3, time_limit=50, v_max=1.0),
        'kinematics': 'unicycle',
        'lidar_noise': True,
        'leg_dynamics': True,
    }
    env = LaserNav(**env_params)
    _, _, obs, info, _ = env.reset(jax.random.PRNGKey(42))
    rng_key = jax.random.PRNGKey(0)
    results = {}
    for dim in network_embeddings_dims:
        network_name = f'jessi_multitask_rl_out_{dim}.pkl'
        file_path = os.path.join(os.path.dirname(__file__), network_name)
        print(f"--- Testing Network: {network_name} ---")
        if not os.path.exists(file_path):
            print(f"File '{network_name}' not found. Skipping...")
            continue
        with open(file_path, 'rb') as f:
            network_params, _, _ = pickle.load(f)
        policy = JESSI(
            v_max=1.0,
            wheels_distance=0.7,
            n_stack=env_params['n_stack'],
            robot_radius=env_params['robot_radius'],
            lidar_num_rays=env_params['lidar_num_rays'],
            lidar_angular_range=env_params['lidar_angular_range'],
            lidar_max_dist=env_params['lidar_max_dist'],
            n_stack_for_action_space_bounding=1,
            embedding_dim=dim,
        )
        print("JIT compilation in progress (this might take a while)...")
        warmup_start = time.perf_counter()
        for _ in range(n_warmup):
            action, rng_key, _, _, _, _, _, _, _, _, _, _ = policy.act(
                key=rng_key,
                obs=obs,
                info=info,
                e2e_network_params=network_params,
                sample=False
            )
            action.block_until_ready()
        print(f"Warmup completed in {time.perf_counter() - warmup_start:.2f} s")
        print(f"Executing {n_inferences} inferences...")
        inference_times = []
        for _ in range(n_inferences):
            start_t = time.perf_counter()
            action, rng_key, _, _, _, _, _, _, _, _, _, _ = policy.act(
                key=rng_key,
                obs=obs,
                info=info,
                e2e_network_params=network_params,
                sample=False
            )
            action.block_until_ready()
            end_t = time.perf_counter()
            inference_times.append(end_t - start_t)
        times_ms = np.array(inference_times) * 1000
        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        results[dim] = {"mean": mean_time, "std": std_time}
        print(f"Result for dim {dim}: {mean_time:.2f} ms +/- {std_time:.2f} ms")
    print("==========================================")
    print("RESULTS RECAP (Inference Time)")
    print("==========================================")
    for dim, stats in results.items():
        print(f"Embedding Dim {dim:3d} : {stats['mean']:6.2f} ms +/- {stats['std']:5.2f} ms")
    print("==========================================")
    save_path = os.path.join(os.path.dirname(__file__), 'jessi_inference_times.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")

if __name__ == '__main__':
    run_benchmark()