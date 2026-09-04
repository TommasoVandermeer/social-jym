from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_structure
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import optax
from pathlib import Path
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward as SocialNavDummyReward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rewards.lasernav_rewards.reward4 import Reward4
from socialjym.utils.rollouts.jessi_s2r_rollouts import jessi_s2r_rl_rollout
from socialjym.utils.training_artifacts import ArtifactStore

no_imitation_learning = False
use_legacy_perception_bootstrap = True
### Sim-to-real parameters
lidar_dt = 0.13
odometry_dt = 0.05
control_delay_mean = 0.1
control_delay_sigma = 0.01
wheels_max_linear_acceleration = 0.87
leg_dynamics = True  # Whether to include leg dynamics in the simulation (introduces more realistic trajectories but also more noise in the data)
### Script parameters
save_videos = False  # Whether to save videos of the debug inspections
### Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 0.45
robot_wmax = 1.9
robot_wheel_distance = 2 * robot_vmax / robot_wmax
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 200
scenario = "hybrid_scenario"
hybrid_scenario_subset = jnp.array([0,1,2,3,4,6,9,10,11,12,13,14,15,16])
n_humans = 4
n_obstacles = 5
humans_policy = 'hsfm'
humans_trajectory_noise_std = 0.0
initial_visibility_chance = 0.5
### PRE-TRAIN Hyperparameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 500_000 # Number of labeled examples to train Perception network
n_parallel_envs = 1_000  # Number of parallel environments to simulate to generate the dataset
embeddings_dim = 32  # Dimension of the embeddings used in JESSI policy
n_detectable_humans = 10  # Number of HCGs that can be detected by the policy
max_humans_velocity = 1.5  # Maximum humans velocity (m/s) used to compute the maximum displacement in the prediction horizon
perception_learning_rate = 0.0005
perception_batch_size = 100
policy_learning_rate = 0.005
critic_learning_rate = 0.005
beta_entropy_il = 0.001
policy_batch_size = 200
policy_n_epochs = 10 # Just a few to not overfit on DIR-SAFE data (if action space becomes too deterministic there will be no exploration in RL fine-tuning)
n_max_epochs = 1000
patience = 100  # Early stopping patience
delta_improvement = 0.001  # Minimum validation improvement to reset early stopping patience
data_split = [0.85, 0.1, 0.05]  # Train/Val/Test split ratios
### MULTI-TASK RL Hyperparameters
rl_n_parallel_envs = 200 
rl_training_updates = 1_000
training_hyperparams = {
    'random_seed': 0,
    'n_humans': n_humans, 
    'n_obstacles': n_obstacles,
    'rl_training_updates': rl_training_updates,
    'rl_parallel_envs': rl_n_parallel_envs,
    "rl_critic_learning_rate": 0.01, # 1e-3
    'rl_learning_rate': 1e-4, # 3e-4
    'rl_learning_rate_final': 1e-5, # 2e-4
    'rl_total_batch_size': 10_000, # 50_000 Nsteps for env = rl_total_batch_size / rl_parallel_envs
    'rl_mini_batch_size': 500, # 2_000 Mini-batch size for each model update
    'rl_micro_batch_size': 250, # 1_000 # Micro-batch size for gradient accumulation 
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 4, # 6
    'rl_beta_entropy': 7e-3, # 1e-4
    'lambda_gae': 0.95, # 0.95
    'initial_visibility_chance': initial_visibility_chance,
    # 'humans_policy': 'hsfm', It is set by default in the LaserNav env
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': hybrid_scenario_subset,
    'reward_function': 'lasernav_reward1',
    'gradient_norm_scale': 1, # Scale the gradient norm by this value
    'safety_loss': False,  # Whether to include safety loss in the RL training
    'target_kl': 0.015,  # Target KL divergence for early stopping in each update
}
training_hyperparams['rl_num_batches'] = training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_mini_batch_size']

reward_config = {
    "goal_reward": 5.0,
    "collision_with_humans_penalty": -2.0,
    "collision_with_obstacles_penalty": -2.0,
    "timeout_penalty": -0.5,
    "progress_to_goal_weight": 0.3,
    "high_rotation_penalty_reward": False,
}

experiment_config = {
    "pipeline_schema": 2,
    "random_seed": random_seed,
    "environment": {
        "scenario": scenario,
        "hybrid_scenario_subset": hybrid_scenario_subset,
        "n_humans": n_humans,
        "n_obstacles": n_obstacles,
        "robot_radius": robot_radius,
        "robot_dt": robot_dt,
        "robot_vmax": robot_vmax,
        "robot_wheel_distance": robot_wheel_distance,
        "kinematics": kinematics,
        "lidar_dt": lidar_dt,
        "odometry_dt": odometry_dt,
        "control_delay_mean": control_delay_mean,
        "control_delay_sigma": control_delay_sigma,
        "wheels_max_linear_acceleration": wheels_max_linear_acceleration,
        "leg_dynamics": leg_dynamics,
        "thick_default_obstacle": True,
        "noisy_walls": True,
        "obstacles_noise": 0.15,
    },
    "model": {
        "architecture_version": 2,
        "n_stack": n_stack,
        "lidar_angular_range": lidar_angular_range,
        "lidar_max_dist": lidar_max_dist,
        "lidar_num_rays": lidar_num_rays,
        "embedding_dim": embeddings_dim,
        "n_detectable_humans": n_detectable_humans,
        "max_humans_velocity": max_humans_velocity,
        "humans_trajectory_noise_std": humans_trajectory_noise_std,
        "logistic_normal_latent_dim": 2,
    },
    "pretraining": {
        "n_steps": n_steps,
        "n_parallel_envs": n_parallel_envs,
        "perception_batch_size": perception_batch_size,
        "policy_batch_size": policy_batch_size,
        "data_split": data_split,
        "initial_visibility_chance": initial_visibility_chance,
        "legacy_perception_bootstrap": use_legacy_perception_bootstrap,
    },
    "reinforcement_learning": training_hyperparams,
    "reward": reward_config,
}

legacy_perception_parameter_paths = (
    Path(__file__).resolve().parent / "realistic_pre_perception_network_32.pkl",
    Path(__file__).resolve().parents[1] / "examples" / "realistic_pre_perception_network_32.pkl",
)


def load_compatible_legacy_perception(reference_params):
    """Explicitly warm-start from old perception weights when shapes still match.

    Legacy files never satisfy a stage gate and are never rewritten.  Their
    weights are only an initializer for training the new canonical artifact.
    """
    if not use_legacy_perception_bootstrap:
        return reference_params
    for legacy_path in legacy_perception_parameter_paths:
        if not legacy_path.exists():
            continue
        try:
            with legacy_path.open("rb") as legacy_file:
                candidate = pickle.load(legacy_file)
            same_structure = tree_structure(candidate) == tree_structure(reference_params)
            same_shapes = same_structure and all(
                old.shape == new.shape
                for old, new in zip(tree_leaves(candidate), tree_leaves(reference_params))
            )
            if same_shapes:
                print(f"Warm-starting perception from legacy weights: {legacy_path}")
                return candidate
            print(f"Ignoring incompatible legacy perception weights: {legacy_path}")
        except (OSError, pickle.PickleError, EOFError, AttributeError, TypeError) as error:
            print(f"Could not load legacy perception weights {legacy_path}: {error}")
    return reference_params

artifact_store = ArtifactStore(
    Path(__file__).resolve().parent / "artifacts",
    "jessi_s2r",
    experiment_config,
)
RAW_DATA = "dir_safe_experiences"
ROBOT_CENTRIC_DATA = "robot_centric_experiences"
PERCEPTION_DATA = "perception_dataset"
PERCEPTION_MODEL = "perception_network"
CONTROLLER_DATA = "actor_critic_dataset"
ACTOR_MODEL = "actor_network"
CRITIC_MODEL = "critic_network"
RL_RESULT = "multitask_rl_result"

def make_reward():
    if training_hyperparams['reward_function'] == 'lasernav_reward1':
        return Reward1(
            robot_radius=robot_radius,
            v_max=robot_vmax,
            timeout_penalty_reward=True,
            **reward_config,
        )
    if training_hyperparams['reward_function'] == 'lasernav_reward4':
        return Reward4(robot_radius=robot_radius, v_max=robot_vmax)
    raise ValueError(f"{training_hyperparams['reward_function']} is not a valid reward function")

def laser_env_parameters(*, lidar_noise):
    return {
        'robot_radius': robot_radius,
        'n_humans': n_humans,
        'n_obstacles': n_obstacles,
        'robot_dt': robot_dt,
        'humans_dt': 0.01,
        'robot_visible': None,
        'scenario': scenario,
        'hybrid_scenario_subset': hybrid_scenario_subset,
        'circle_radius': 7,
        'kinematics': kinematics,
        'lidar_noise': lidar_noise,
        'lidar_angular_range': lidar_angular_range,
        'lidar_max_dist': lidar_max_dist,
        'lidar_num_rays': lidar_num_rays,
        'n_stack': n_stack,
        'wheels_distance': robot_wheel_distance,
        'lidar_dt': lidar_dt,
        'odometry_dt': odometry_dt,
        'control_delay_mean': control_delay_mean,
        'control_delay_sigma': control_delay_sigma,
        'wheels_max_linear_acceleration': wheels_max_linear_acceleration,
        'leg_dynamics': leg_dynamics,
        'thick_default_obstacle': True,
        'noisy_walls': True,
        'obstacles_noise': 0.15,
    }
# JESSI policy
jessi = JESSI_S2R(
    v_max=robot_vmax, 
    wheels_distance=robot_wheel_distance,
    dt=robot_dt, 
    lidar_num_rays=lidar_num_rays, 
    lidar_max_dist=lidar_max_dist,
    lidar_angular_range=lidar_angular_range,
    n_stack=n_stack, 
    n_detectable_humans=n_detectable_humans, 
    max_humans_velocity=max_humans_velocity,
    embedding_dim=embeddings_dim,
    humans_trajectory_noise_std=humans_trajectory_noise_std,
)
# Plotting settings
ax_visibility = 2
ax_lims = jnp.array([
    [-lidar_max_dist-ax_visibility,lidar_max_dist+ax_visibility],
    [-lidar_max_dist-ax_visibility, lidar_max_dist+ax_visibility]
])

### Parameters validation
assert n_steps % n_parallel_envs == 0, "n_steps must be divisible by n_parallel_envs"
assert sum(data_split) == 1.0, "data_split must sum to 1.0"
assert n_steps % perception_batch_size == 0, "n_steps must be divisible by batch_size"
assert int(n_steps * data_split[0]) % perception_batch_size == 0, "Training set size must be divisible by batch_size"
assert int(n_steps * data_split[1]) % perception_batch_size == 0, "Validation set size must be divisible by batch_size"
assert int(n_steps * data_split[2]) % perception_batch_size == 0, "Test set size must be divisible by batch_size"

### GENERATE PRE-TRAINING DATASET
if not artifact_store.is_valid(PERCEPTION_DATA, dependencies=(ROBOT_CENTRIC_DATA,)):
    social_env_params = {
        'robot_radius': 0.3,
        'n_humans': n_humans,
        'n_obstacles': n_obstacles,
        'robot_dt': robot_dt,
        'robot_radius': robot_radius, 
        'humans_dt': 0.01,
        'robot_visible': True,
        'scenario': scenario,
        'hybrid_scenario_subset': hybrid_scenario_subset,
        'kinematics': kinematics,
        'lidar_angular_range':lidar_angular_range,
        'lidar_max_dist':lidar_max_dist,
        'lidar_num_rays':lidar_num_rays,
        'lidar_noise': False, # Noise is introduced during training as data augmentation
        'thick_default_obstacle': True,
        'noisy_walls': True,
        'obstacles_noise': 0.15,
        }
    env = SocialNav(**social_env_params, humans_policy=humans_policy, reward_function=SocialNavDummyReward(kinematics=kinematics, v_max=robot_vmax))
    laser_env = LaserNav(**laser_env_parameters(lidar_noise=False), reward_function=make_reward())
    # DIR-SAFE policy
    dir_safe = DIRSAFE(env.reward_function, v_max=robot_vmax, dt=robot_dt)
    with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
        actor_params = pickle.load(f)['actor_params']
    dataset_scenarios_prob = jnp.full(
        (len(hybrid_scenario_subset),), 1.0 / len(hybrid_scenario_subset)
    )
    dataset_visibility_chance = initial_visibility_chance
    def simulate_n_steps(n_steps):
        @loop_tqdm(n_steps//n_parallel_envs, desc="Simulating steps")
        @jit
        def _simulate_steps_with_lidar(i:int, for_val:tuple):
            ## Retrieve data from the tuple
            data, state, info, reset_keys, env_keys, policy_keys, lasernav_obs, episode_starts = for_val
            ## Compute robot action
            obs = vmap(env._get_obs, in_axes=(0, 0))(state, info)
            action, new_policy_keys, _, _, _ = dir_safe.batch_act(policy_keys, obs, info, actor_params, sample=False)
            ## Compute robot-frame action-space parameters from the same LiDAR stack used by JESSI.
            _, point_clouds = vmap(jessi.compute_perception_input)(lasernav_obs)
            action_space_parameters = vmap(jessi.bound_action_space)(point_clouds)
            ## Simulate one authoritative LASERNAV transition.
            final_state, final_lasernav_obs, final_info, (transition_rewards, _), outcomes, (final_reset_keys, final_env_keys) = laser_env.batch_step(
                state,
                info,
                action, 
                reset_keys,
                env_keys,
                test=False,
                reset_if_done=True,
                scenarios_prob=dataset_scenarios_prob,
                visibility_chance=dataset_visibility_chance,
            )
            ## Save output data
            step_out_data = {
                "episode_starts": episode_starts,
                "terminals": ~outcomes["nothing"],
                "lasernav_observations": lasernav_obs,
                "humans_positions": obs[:,:-1,:2],
                "humans_velocities": obs[:,:-1,2:4],
                "humans_radii": info["humans_parameters"][:,:,0],
                "humans_orientations": state[:,:-1,4],
                "robot_positions": obs[:,-1,:2],
                "robot_orientations": obs[:,-1,5],
                "robot_actions": action,
                "robot_goals": info["robot_goal"],
                "static_obstacles": info["static_obstacles"][:,-1],
                "rewards": transition_rewards,
                "humans_visibility": info["humans_visibility_mask"],
                "obstacles_visibility": info["obstacles_visibility_mask"],
                # Critic training stuff
                "states": state,
                "actions_histories": lasernav_obs[:,:jessi.n_actions_history,6:8],
                "humans_goal": info["humans_goal"],
                "humans_pairwise_visibility": info["visibility"][:, :-1, :-1],
                "humans_parameters": info["humans_parameters"],
                "robot_goal": info["robot_goal"],
                "robot_radius": jnp.full((n_parallel_envs,), jessi.robot_radius),
                "v_max": jnp.full((n_parallel_envs,), jessi.v_max),
                "wheels_distance": jnp.full((n_parallel_envs,), jessi.wheels_distance),
                "wheels_max_linear_acceleration": jnp.full((n_parallel_envs,), laser_env.wheels_max_linear_acceleration),
                "robot_delay": info["robot_delay"],
                "action_space_params": action_space_parameters,
            }
            data = tree_map(lambda x, y: x.at[i].set(y), data, step_out_data)
            return data, final_state, final_info, final_reset_keys, final_env_keys, new_policy_keys, final_lasernav_obs, ~outcomes["nothing"]
        # Initialize first episode
        reset_keys = random.split(random.PRNGKey(random_seed), n_parallel_envs)
        state, reset_keys, lasernav_obs, info, _ = laser_env.batch_reset(
            reset_keys,
            scenarios_prob=dataset_scenarios_prob,
            visibility_chance=dataset_visibility_chance,
        )
        env_keys = random.split(random.PRNGKey(random_seed + 1), n_parallel_envs)
        policy_keys = random.split(random.PRNGKey(random_seed + 2), n_parallel_envs)
        episode_starts = jnp.ones((n_parallel_envs,), dtype=bool)
        n_iterations = n_steps // n_parallel_envs
        # Initialize setting data
        data = {
            "episode_starts": jnp.zeros((n_iterations,n_parallel_envs), dtype=bool),
            "terminals": jnp.zeros((n_iterations,n_parallel_envs), dtype=bool),
            "lasernav_observations": jnp.zeros((n_iterations,n_parallel_envs,*lasernav_obs.shape[1:])),
            "humans_positions": jnp.zeros((n_iterations,n_parallel_envs,n_humans,2)),
            "humans_velocities": jnp.zeros((n_iterations,n_parallel_envs,n_humans,2)),
            "humans_orientations": jnp.zeros((n_iterations,n_parallel_envs,n_humans)),
            "humans_radii": jnp.zeros((n_iterations,n_parallel_envs,n_humans)),
            "robot_positions": jnp.zeros((n_iterations,n_parallel_envs,2)),
            "robot_orientations": jnp.zeros((n_iterations,n_parallel_envs)),
            "robot_actions": jnp.zeros((n_iterations,n_parallel_envs,2)),
            "robot_goals": jnp.zeros((n_iterations,n_parallel_envs,2)),
            "static_obstacles": jnp.zeros((n_iterations,n_parallel_envs,*info["static_obstacles"].shape[2:])),
            "rewards": jnp.zeros((n_iterations,n_parallel_envs)),
            "humans_visibility": jnp.zeros((n_iterations,n_parallel_envs,*info["humans_visibility_mask"].shape[1:]), dtype=bool),
            "obstacles_visibility": jnp.zeros((n_iterations,n_parallel_envs,*info["obstacles_visibility_mask"].shape[1:]), dtype=bool),
            ## Critic training stuff
            "states": jnp.zeros((n_iterations,n_parallel_envs,n_humans+1,6)),
            "actions_histories": jnp.zeros((n_iterations,n_parallel_envs,jessi.n_actions_history,2)),
            # Env params
            "humans_goal": jnp.zeros((n_iterations,n_parallel_envs,n_humans,2)),
            "humans_pairwise_visibility": jnp.zeros((n_iterations,n_parallel_envs,n_humans,n_humans), dtype=bool),
            "humans_parameters": jnp.zeros((n_iterations,n_parallel_envs,n_humans,19)),
            # Robot params
            "robot_goal": jnp.zeros((n_iterations,n_parallel_envs,2)),
            "robot_radius": jnp.zeros((n_iterations,n_parallel_envs)),
            "v_max": jnp.zeros((n_iterations,n_parallel_envs)),
            "wheels_distance": jnp.zeros((n_iterations,n_parallel_envs)),
            "wheels_max_linear_acceleration": jnp.zeros((n_iterations,n_parallel_envs)),
            "robot_delay": jnp.zeros((n_iterations,n_parallel_envs)),
            # Action space params
            "action_space_params": jnp.zeros((n_iterations,n_parallel_envs,3)),
        }
        # Step loop
        data, _, _, _, _, _, _, _ = lax.fori_loop(
            0,
            n_iterations,
            _simulate_steps_with_lidar,
            (data, state, info, reset_keys, env_keys, policy_keys, lasernav_obs, episode_starts)
        )
        # Compute returns
        @jit
        def _discounted_cumsum(rewards, dones):
            def scan_fun(carry, x):
                reward, done = x
                new_carry = reward + carry * jnp.power(laser_env.reward_function.gamma, jessi.dt * jessi.v_max) * (1.0 - done)
                return new_carry, new_carry
            _, discounted_cumsums = lax.scan(scan_fun, 0.0, (rewards[::-1], dones[::-1]))
            return discounted_cumsums[::-1]
        data["returns"] = vmap(_discounted_cumsum, in_axes=(1,1))(data["rewards"], data["terminals"]).T
        data = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), data)  # Merge parallel envs
        return data
    ## GENERATE RAW DATASET
    if not artifact_store.is_valid(RAW_DATA):
        # Generate raw data
        raw_data = simulate_n_steps(n_steps)
        # Save raw data dataset
        artifact_store.save(RAW_DATA, raw_data)
    else:
        # Load raw data dataset
        raw_data = artifact_store.load(RAW_DATA)
    ## GENERATE ROBOT-CENTERED DATASET
    if not artifact_store.is_valid(ROBOT_CENTRIC_DATA, dependencies=(RAW_DATA,)):
        robot_centric_data = {
            "episode_starts": raw_data["episode_starts"],
            "lasernav_observations": raw_data["lasernav_observations"],
            "rc_humans_positions": jnp.zeros((n_steps, n_humans, 2)),
            "rc_humans_orientations": jnp.zeros((n_steps, n_humans)),
            "rc_humans_velocities": jnp.zeros((n_steps, n_humans, 2)),
            "humans_radii": raw_data["humans_radii"],
            "robot_actions": raw_data["robot_actions"],
            "robot_positions": raw_data["robot_positions"],
            "robot_orientations": raw_data["robot_orientations"],
            "rc_robot_goals": jnp.zeros((n_steps, 2)),
            "rc_obstacles": jnp.zeros_like(raw_data["static_obstacles"]),
            "humans_visibility": raw_data["humans_visibility"],
            "obstacles_visibility": raw_data["obstacles_visibility"],
        }
        # Compute robot-centered simulation
        robot_centric_data["rc_humans_positions"], robot_centric_data["rc_humans_orientations"], \
        robot_centric_data["rc_humans_velocities"], robot_centric_data["rc_obstacles"], \
        robot_centric_data["rc_robot_goals"] = laser_env.batch_robot_centric_transform(
            raw_data['humans_positions'],
            raw_data['humans_orientations'],
            raw_data['humans_velocities'],
            raw_data['static_obstacles'],
            raw_data['robot_positions'],
            raw_data['robot_orientations'],
            raw_data["robot_goals"],
        )
        ## DEBUG: Plot frames stream for visual inspection
        # Plot robot-centric simulation
        fig, ax = plt.subplots(figsize=(8,8))
        def animate(frame):
            ax.clear()
            ax.set_title('Robot-Centric Frame Inspection')
            ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y', labelpad=-13)
            ax.set_aspect('equal', adjustable='box')
            # Plot robot goal
            ax.scatter(robot_centric_data["rc_robot_goals"][frame,0], robot_centric_data["rc_robot_goals"][frame,1], marker="*", color="red", zorder=2)
            # Plot humans
            for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
                color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
                alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
                if humans_policy == 'hsfm':
                    head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                    ax.add_patch(head)
                circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
                ax.add_patch(circle)
            # Plot robot
            if kinematics == 'unicycle':
                head = plt.Circle((robot_radius, 0.), 0.1, color='black', zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((0.,0.), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
            ax.add_patch(circle)
            # Plot static obstacles
            for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
                for j, s in enumerate(o):
                    color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                    linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                    alpha = 1 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                    ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
            # Plot lidar scans
            for distance, angle in zip(robot_centric_data["lasernav_observations"][frame,0,11:], jessi.lidar_angles_robot_frame):
                ax.plot(
                    [0, distance * jnp.cos(angle)],
                    [0, distance * jnp.sin(angle)],
                    color='blue',
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=0,
                )
        anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
        if save_videos:
            save_path = os.path.join(os.path.dirname(__file__), f'robot_centric_simulation.mp4')
            writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
            anim.save(save_path, writer=writer_video, dpi=300)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()
        # Save robot-centered robot_centric_data
        artifact_store.save(ROBOT_CENTRIC_DATA, robot_centric_data, dependencies=(RAW_DATA,))
    else:
        # Load robot-centered dataset
        robot_centric_data = artifact_store.load(ROBOT_CENTRIC_DATA, dependencies=(RAW_DATA,))
    ### GENERATE PERCEPTION NETWORK TRAINING DATASET
    # Initialize final dataset
    dataset = {
        "observations": robot_centric_data["lasernav_observations"],
        "targets": {
            "gt_poses": robot_centric_data["rc_humans_positions"],
            "gt_vels": robot_centric_data["rc_humans_velocities"],
            "gt_mask": robot_centric_data["humans_visibility"],
        },
    }
    ## DEBUG: Inspect training dataset
    debugging_steps = 100
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Network Inputs Inspection')
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot lidar scans
        point_clouds = jessi.align_lidar(dataset["observations"][frame])[0]  # (n_stack, lidar_num_rays, 2)
        for i, cloud in enumerate(point_clouds):
            # color/alpha fade with i (smaller i -> less faded)
            t = (1 - i / (n_stack - 1))  # in [0,1]
            ax.scatter(
                cloud[:,0],
                cloud[:,1],
                c=0.3 + 0.7 * jnp.ones((lidar_num_rays,)) * t,
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                alpha=0.3 + 0.7 * t,
                zorder=20 + n_stack - i,
            )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=debugging_steps)
    if save_videos:
        save_path = os.path.join(os.path.dirname(__file__), f'network_inputs.mp4')
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    # Save dataset
    artifact_store.save(PERCEPTION_DATA, dataset, dependencies=(ROBOT_CENTRIC_DATA,))
    # Delete robot_centric_data and raw_data to save memory
    del robot_centric_data
    del raw_data
else:
    # Load dataset
    dataset = artifact_store.load(PERCEPTION_DATA, dependencies=(ROBOT_CENTRIC_DATA,))

### PRE-TRAIN PERCEPTION NETWORK
if not artifact_store.is_valid(PERCEPTION_MODEL, dependencies=(PERCEPTION_DATA,)):
    # Initialize network
    params, _, _, _ = jessi.init_nns(random.PRNGKey(random_seed))
    params = load_compatible_legacy_perception(params)
    # Count network parameters
    def count_params(params):
        return sum(jnp.prod(jnp.array(p.shape)) for layer in params.values() for p in layer.values())
    n_params = count_params(params)
    print(f"# Perception network parameters: {n_params}")
    # Split dataset into TRAIN, VAL, TEST
    n_data = dataset["observations"].shape[0]
    n_train_data = int(data_split[0] * n_data)
    n_val_data = int(data_split[1] * n_data)
    n_test_data = n_data - n_train_data - n_val_data
    print(f"# Training dataset size: {n_data} experiences")
    print(f"-> TRAIN size: {n_train_data} experiences")
    print(f"-> VAL size: {n_val_data} experiences")
    print(f"-> TEST size: {n_test_data} experiences")
    shuffle_key = random.PRNGKey(random_seed + 1_000_000)
    indexes = jnp.arange(n_data)
    shuffled_indexes = random.permutation(shuffle_key, indexes)
    train_indexes = shuffled_indexes[:n_train_data]
    val_indexes = shuffled_indexes[n_train_data:n_train_data + n_val_data]
    test_indexes = shuffled_indexes[n_train_data + n_val_data:]
    train_dataset =  tree_map(lambda x: x[train_indexes], dataset)
    val_dataset =  tree_map(lambda x: x[val_indexes], dataset)
    test_dataset =  tree_map(lambda x: x[test_indexes], dataset)
    # Free memory
    del dataset 
    del indexes
    del shuffled_indexes
    del train_indexes
    del val_indexes
    del test_indexes
    # Initialize optimizer and its state
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=perception_learning_rate,
                warmup_steps=n_max_epochs // 100 * (n_train_data // perception_batch_size),
                decay_steps=n_max_epochs * (n_train_data // perception_batch_size),
                end_value=perception_learning_rate/100,
            ),
            weight_decay=1e-2,
        ),
    )
    optimizer_state = optimizer.init(params)

    @jit
    def batch_val_test_loss(
        batch:dict,
        params:dict,
        seed:int,
    ) -> tuple:
        # TODO: Shutdown dropout layers and stochasticity during validation
        inputs = vmap(jessi.compute_perception_input, in_axes=(0))(batch["observations"])[0]
        targets = batch["targets"]
        # Compute loss
        loss = jessi.perception_loss(
            params, 
            inputs,
            targets,
        )
        return loss
    @jit
    def val_test_loss(
        batches:dict,
        params:dict,
        seeds:int,
    ):
        # VMAP version (fast, much memory consumption)
        #return vmap(batch_val_test_loss, in_axes=(0, None, 0))(batches, params, seeds)
        # SCAN version (slow, low memory consumption)
        def _scan_loop(carry, x):
            params = carry
            batch, seed = x
            loss = batch_val_test_loss(
                batch,
                params,
                seed,
            )
            return params, loss
        _, losses = lax.scan(
            _scan_loop,
            params,
            (batches, seeds),
        )
        return losses
    @jit
    def batch_augment_data(
        batch:dict,
        keys:random.PRNGKey,
        base_lidar_noise_std:float = 0.01, # 1cm base noise
        proportional_lidar_noise_std:float = 0.01, # 1% proportional noise
        beam_dropout_prob:float = 0.03, # 3% beams dropout
    ) -> dict:
        return vmap(augment_data, in_axes=(0, 0, None, None, None))(batch, keys, base_lidar_noise_std, proportional_lidar_noise_std, beam_dropout_prob)
    @jit 
    def augment_data(
        data:dict,
        key:random.PRNGKey,
        base_lidar_noise_std:float,
        proportional_lidar_noise_std:float,
        beam_dropout_prob:float,
    ) -> dict:
        # data = {
        #     "inputs": shape (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
        #               each token: [norm_dist, hit, x, y, sin_fixed_theta, cos_fixed_theta, delta_t]
        #     "targets": {
        #         "gt_mask": shape (n_humans,),
        #         "gt_poses": shape (n_humans, 2),
        #         "gt_vels": shape (n_humans, 2),
        #     }
        # }
        input_key, rotation_key, beam_dropout_key = random.split(key, 3)
        ## Gaussian noise to LiDAR scans + Beam dropout
        raw_distances = data['inputs'][:,:,0] * jessi.max_beam_range  # (n_stack, lidar_num_rays)
        sigma = base_lidar_noise_std + proportional_lidar_noise_std * raw_distances  # (n_stack, lidar_num_rays)
        noise = random.normal(input_key, shape=raw_distances.shape) * sigma * data['inputs'][:,:,1]  # (n_stack, lidar_num_rays)
        noisy_distances = jnp.clip(raw_distances + noise, 0., jessi.max_beam_range) # (n_stack, lidar_num_rays)
        is_dropout = random.bernoulli(beam_dropout_key, p=beam_dropout_prob, shape=raw_distances.shape)
        noisy_distances = jnp.where(is_dropout, jessi.max_beam_range, noisy_distances)  # (n_stack, lidar_num_rays)
        new_hit = jnp.where(noisy_distances < jessi.max_beam_range, 1.0, 0.0) * (1.0 - is_dropout)  # (n_stack, lidar_num_rays)
        x = noisy_distances * data['inputs'][:,:,4]  # (n_stack, lidar_num_rays)
        y = noisy_distances * data['inputs'][:,:,5]  # (n_stack, lidar_num_rays)
        data['inputs'] = data['inputs'].at[:,:,0].set(noisy_distances / jessi.max_beam_range)
        data['inputs'] = data['inputs'].at[:,:,1].set(new_hit)
        data['inputs'] = data['inputs'].at[:,:,2].set(x)
        data['inputs'] = data['inputs'].at[:,:,3].set(y)
        ## Random rotation
        alpha = random.uniform(rotation_key, minval=-jnp.pi, maxval=jnp.pi)
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        rot_mat = jnp.array([[ca, -sa], [sa, ca]])
        s_new = data['inputs'][..., 4] * ca + data['inputs'][..., 5] * sa
        c_new = data['inputs'][..., 5] * ca - data['inputs'][..., 4] * sa
        xy_rotated = data['inputs'][..., 2:4]  @ rot_mat.T
        data['inputs'] = data['inputs'].at[..., 2:4].set(xy_rotated)
        data['inputs'] = data['inputs'].at[..., 4].set(s_new) 
        data['inputs'] = data['inputs'].at[..., 5].set(c_new) 
        data['targets']['gt_poses'] = data['targets']['gt_poses'] @ rot_mat.T
        data['targets']['gt_vels'] = data['targets']['gt_vels'] @ rot_mat.T
        return data
    @jit 
    def _epoch_loop(
        epoch_for_val:tuple,
    ) -> tuple:
        early_stopping_info, train_dataset, val_dataset, params, optimizer_state, train_losses, val_losses = epoch_for_val
        i = early_stopping_info['epoch']
        ## TRAINING
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_train_data)
        shuffled_indexes = random.permutation(shuffle_key, indexes)
        train_epoch_data = tree_map(lambda x: x[shuffled_indexes], train_dataset)
        n_train_batches = n_train_data // perception_batch_size
        @jit
        def _batch_train_loop(
            j:int,
            batch_for_val:tuple
        ) -> tuple:
            train_epoch_data, params, optimizer_state, losses, grads_norms = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(perception_batch_size) + j * perception_batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, train_epoch_data)
            ## Tranform batch into training_batch data (it is done during training loop to save memory)
            # training_batch = {
            #     "inputs": jnp.zeros((perception_batch_size, n_stack * lidar_num_rays, 7)),
            #     "targets": {
            #         "gt_mask": jnp.zeros((perception_batch_size, n_humans,)),
            #         "gt_poses": jnp.zeros((perception_batch_size, n_humans, 2)),
            #         "gt_vels": jnp.zeros((perception_batch_size, n_humans, 2)),
            #     }
            # }
            training_batch = {}
            training_batch["inputs"] = vmap(jessi.compute_perception_input, in_axes=(0))(batch["observations"])[0]
            training_batch["targets"] = batch["targets"]
            ## DATA AUGMENTATION
            training_batch = batch_augment_data(
                training_batch,
                random.split(random.PRNGKey(i * n_train_batches + j), perception_batch_size),
            )
            # Update parameters
            params, optimizer_state, loss, grads_norm = jessi.update_perception(
                params, 
                optimizer, 
                optimizer_state,
                training_batch,
                key=random.PRNGKey(i * n_train_batches + j),
            )
            # debug.print("Epoch {x}, Batch {y}, TRAIN Loss: {l}", x=i, y=j, l=loss)
            # Save loss
            losses = losses.at[i,j].set(loss)
            grads_norms = grads_norms.at[j].set(grads_norm)
            return train_epoch_data, params, optimizer_state, losses, grads_norms
        _, params, optimizer_state, train_losses, grads_norms = lax.fori_loop(
            0,
            n_train_batches,
            _batch_train_loop,
            (train_epoch_data, params, optimizer_state, train_losses, jnp.zeros((n_train_batches,)))
        )
        ## VALIDATION
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_val_data)
        shuffled_indexes = random.permutation(shuffle_key, indexes)
        val_epoch_data = tree_map(lambda x: x[shuffled_indexes], val_dataset)
        val_epoch_data = tree_map(lambda x: x.reshape((n_val_data // perception_batch_size, perception_batch_size) + x.shape[1:]), val_epoch_data)
        val_losses = val_losses.at[i].set(
            val_test_loss(
                val_epoch_data,
                params,
                jnp.arange(n_val_data // perception_batch_size),
            )
        )
        current_val_loss = jnp.mean(val_losses[i])
        val_loss_improved = (early_stopping_info['best_val_loss'] - current_val_loss) / jnp.abs(early_stopping_info['best_val_loss']) > delta_improvement
        val_loss_improved = val_loss_improved | (i == 0)
        debug.print("Epoch {x}, TRAIN Loss: {t}, VAL Loss: {v} (Improved: {imp}), GRAD Norm: {g}", x=i, t=jnp.mean(train_losses[i]), v=current_val_loss, imp=val_loss_improved, g=jnp.mean(grads_norms))
        # Update early stopping info
        @jit
        def _update_early_stopping_info_improved(early_stopping_info):
            early_stopping_info['best_val_loss'] = current_val_loss
            early_stopping_info['best_params'] = params
            early_stopping_info['last_improvement'] = i
            return early_stopping_info
        early_stopping_info = lax.cond(val_loss_improved, _update_early_stopping_info_improved, lambda esi: esi, early_stopping_info)
        early_stopping_info['epoch'] = i + 1
        early_stopping_info['stop'] = ((i - early_stopping_info['last_improvement']) >= patience) | (early_stopping_info['epoch'] >= n_max_epochs)
        return early_stopping_info, train_dataset, val_dataset, params, optimizer_state, train_losses, val_losses
    # Epoch loop
    early_stopping_info, _, _, _, optimizer_state, train_losses, val_losses = lax.while_loop(
        lambda x: ~x[0]['stop'],
        _epoch_loop,
        (
            {'stop': False, 'epoch': 0, 'last_improvement': 0, 'best_params': params, 'best_val_loss': jnp.inf}, 
            train_dataset, 
            val_dataset, 
            params, 
            optimizer_state, 
            jnp.zeros((n_max_epochs, int(n_train_data // perception_batch_size))), 
            jnp.zeros((n_max_epochs, int(n_val_data // perception_batch_size)))
        )
    )
    n_epochs = early_stopping_info['epoch']
    params = early_stopping_info['best_params']
    print(f"\nTraining completed in {n_epochs} epochs. - Best val loss: {early_stopping_info['best_val_loss']}\n")
    # Save trained parameters
    artifact_store.save(PERCEPTION_MODEL, params, dependencies=(PERCEPTION_DATA,))
    ## TEST
    n_train_batches = n_train_data // perception_batch_size
    test_losses = jnp.zeros((1, int(n_test_data // perception_batch_size)))
    shuffle_key = random.PRNGKey(random_seed)
    indexes = jnp.arange(n_test_data)
    shuffled_indexes = random.permutation(shuffle_key, indexes)
    test_epoch_data = tree_map(lambda x: x[shuffled_indexes], test_dataset)
    test_epoch_data = tree_map(lambda x: x.reshape((n_test_data // perception_batch_size, perception_batch_size) + x.shape[1:]), test_epoch_data)
    test_losses = val_test_loss(
        test_epoch_data,
        params,
        jnp.arange(n_test_data // perception_batch_size),
    )
    # Plot training and validation loss
    avg_train_losses = jnp.mean(train_losses[:n_epochs], axis=1)
    avg_val_losses = jnp.mean(val_losses[:n_epochs], axis=1)
    avg_test_loss = jnp.mean(test_losses)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(right=0.9)
    fig.suptitle("Perception - Test Loss: {:.4f}".format(avg_test_loss))
    ax.plot(jnp.arange(n_epochs), avg_train_losses, label="Train", color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(jnp.arange(n_epochs), avg_val_losses, label="Val", color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.legend()
    artifact_store.root.mkdir(parents=True, exist_ok=True)
    fig.savefig(artifact_store.figure_path(PERCEPTION_MODEL), format='eps')
    del train_dataset
    del val_dataset
    del test_dataset
else:
    # Load trained parameters
    encoder_params = artifact_store.load(PERCEPTION_MODEL, dependencies=(PERCEPTION_DATA,))

### PRE-TRAIN ACTOR & CRITIC NETWORKS (IMITATION LEARNING)
controller_dependencies = (RAW_DATA, ROBOT_CENTRIC_DATA, PERCEPTION_DATA)
controller_data_valid = artifact_store.is_valid(CONTROLLER_DATA, dependencies=controller_dependencies)
actor_critic_ready = (
    controller_data_valid
    and artifact_store.is_valid(ACTOR_MODEL, dependencies=(CONTROLLER_DATA, PERCEPTION_MODEL))
    and artifact_store.is_valid(CRITIC_MODEL, dependencies=(CONTROLLER_DATA,))
)
if not actor_critic_ready:
    # Load trained perception parameters
    encoder_params = artifact_store.load(PERCEPTION_MODEL, dependencies=(PERCEPTION_DATA,))
    # CREATE ACTOR & CRITIC INPUTS DATASET
    if not controller_data_valid:
        # LOAD DATASETs
        raw_data = artifact_store.load(RAW_DATA)
        robot_centric_data = artifact_store.load(ROBOT_CENTRIC_DATA, dependencies=(RAW_DATA,))
        dataset = artifact_store.load(PERCEPTION_DATA, dependencies=(ROBOT_CENTRIC_DATA,))
        zeros = jnp.zeros((len(raw_data["action_space_params"]),))
        action_vertices = jnp.stack(
            [
                jnp.stack([zeros, raw_data["action_space_params"][:, 1] * robot_wmax], axis=-1),
                jnp.stack([zeros, raw_data["action_space_params"][:, 2] * -robot_wmax], axis=-1),
                jnp.stack([raw_data["action_space_params"][:, 0] * robot_vmax, zeros], axis=-1),
            ],
            axis=1,
        )
        projected_actor_actions = jessi.action_distribution.batch_project_to_support(
            {"vertices": action_vertices}, raw_data["robot_actions"]
        )
        projection_distances = jnp.linalg.norm(projected_actor_actions - raw_data["robot_actions"], axis=-1)
        print(
            "Projected DIR-SAFE targets:",
            f"{float(jnp.mean(projection_distances > 1e-6)):.2%}",
            "mean distance:",
            float(jnp.mean(projection_distances)),
        )
        # Compute actor inputs for the entire dataset
        controller_dataset = {
            # Actor
            "observations": dataset['observations'],
            "rc_robot_goals": robot_centric_data["rc_robot_goals"],
            "actor_actions": projected_actor_actions,
            "actor_projection_distances": projection_distances,
            # Critic
            "states": raw_data['states'],
            "actions_history": raw_data["actions_histories"],
            "humans_goal": raw_data["humans_goal"],
            "humans_visibility": raw_data["humans_pairwise_visibility"],
            "humans_parameters": raw_data["humans_parameters"],
            "static_obstacles": jnp.repeat(raw_data["static_obstacles"][:,None,:,:,:,:], n_humans+1, axis=1),
            "robot_goal": raw_data["robot_goal"],
            "robot_radius": raw_data["robot_radius"],
            "v_max": raw_data["v_max"],
            "wheels_distance": raw_data["wheels_distance"],
            "wheels_max_linear_acceleration": raw_data["wheels_max_linear_acceleration"],
            "robot_delay": raw_data["robot_delay"],
            "action_space_params": raw_data["action_space_params"],
            "returns": raw_data["returns"],
        }
        # Save actor inputs
        artifact_store.save(CONTROLLER_DATA, controller_dataset, dependencies=controller_dependencies)
        # FREE UNUSED MEMORY
        del dataset
        del robot_centric_data
        del raw_data
    else:
        # Load actor inputs
        controller_dataset = artifact_store.load(CONTROLLER_DATA, dependencies=controller_dependencies)
    print("Any critic target is finite?: ", jnp.all(jnp.isfinite(controller_dataset["returns"])))
    print("Any actor target is finite?: ", jnp.all(jnp.isfinite(controller_dataset["actor_actions"])))
    print("Any states is finite?: ", jnp.all(jnp.isfinite(controller_dataset["states"])))
    print("Any actions_history is finite?: ", jnp.all(jnp.isfinite(controller_dataset["actions_history"])))
    print("Any humans_goal is finite?: ", jnp.all(jnp.isfinite(controller_dataset["humans_goal"])))
    print("Any humans_parameters is finite?: ", jnp.all(jnp.isfinite(controller_dataset["humans_parameters"])))
    print("Any static_obstacles is finite?: ", jnp.all(jnp.isfinite(controller_dataset["static_obstacles"])))
    print("Any robot_goal is finite?: ", jnp.all(jnp.isfinite(controller_dataset["robot_goal"])))
    print("Any robot_radius is finite?: ", jnp.all(jnp.isfinite(controller_dataset["robot_radius"])))
    print("Any v_max is finite?: ", jnp.all(jnp.isfinite(controller_dataset["v_max"])))
    print("Any wheels_distance is finite?: ", jnp.all(jnp.isfinite(controller_dataset["wheels_distance"])))
    print("Any wheels_max_linear_acceleration is finite?: ", jnp.all(jnp.isfinite(controller_dataset["wheels_max_linear_acceleration"])))
    print("Any robot_delay is finite?: ", jnp.all(jnp.isfinite(controller_dataset["robot_delay"])))
    print("Any action_space_params is finite?: ", jnp.all(jnp.isfinite(controller_dataset["action_space_params"])))
    # INITIALIZE ACTOR NETWORK
    # Initialize actor network
    _, actor_params, critic_params, _ = jessi.init_nns(random.PRNGKey(random_seed))
    # Count network parameters
    def count_params(params):
        return sum(jnp.prod(jnp.array(p.shape)) for layer in params.values() for p in layer.values())
    n_params = count_params(actor_params)
    print(f"# Actor network parameters: {n_params}")
    n_params = count_params(critic_params)
    print(f"# Critic network parameters: {n_params}")
    # TRAINING LOOP
    # Initialize optimizer and its state
    actor_optimizer = optax.sgd(learning_rate=policy_learning_rate, momentum=0.9)
    actor_optimizer_state = actor_optimizer.init(actor_params)
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4, eps=1e-5)
    )
    critic_optimizer_state = critic_optimizer.init(critic_params)
    n_data = controller_dataset["observations"].shape[0]
    n_train_batches = n_data // policy_batch_size
    print(f"# Training dataset size: {controller_dataset['observations'].shape[0]} experiences")
    @jit
    def batch_augment_data(
        batch:dict,
        keys:random.PRNGKey,
        base_lidar_noise_std:float = 0.01, # 1cm base noise
        proportional_lidar_noise_std:float = 0.01, # 1% proportional noise
        beam_dropout_prob:float = 0.03, # 3% beams dropout
    ) -> dict:
        return vmap(augment_data, in_axes=(0, 0, None, None, None))(batch, keys, base_lidar_noise_std, proportional_lidar_noise_std, beam_dropout_prob)
    @jit 
    def augment_data(
        data:dict,
        key:random.PRNGKey,
        base_lidar_noise_std:float,
        proportional_lidar_noise_std:float,
        beam_dropout_prob:float,
    ) -> dict:
        # data = {
        #     "inputs": shape (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
        #               each token: [norm_dist, hit, x, y, sin_fixed_theta, cos_fixed_theta, delta_t]
        #     "targets": {
        #         "gt_mask": shape (n_humans,),
        #         "gt_poses": shape (n_humans, 2),
        #         "gt_vels": shape (n_humans, 2),
        #     }
        # }
        input_key, beam_dropout_key = random.split(key, 2)
        ## Gaussian noise to LiDAR scans + Beam dropout
        raw_distances = data['inputs'][:,:,0] * jessi.max_beam_range  # (n_stack, lidar_num_rays)
        sigma = base_lidar_noise_std + proportional_lidar_noise_std * raw_distances  # (n_stack, lidar_num_rays)
        noise = random.normal(input_key, shape=raw_distances.shape) * sigma * data['inputs'][:,:,1]  # (n_stack, lidar_num_rays)
        noisy_distances = jnp.clip(raw_distances + noise, 0., jessi.max_beam_range) # (n_stack, lidar_num_rays)
        is_dropout = random.bernoulli(beam_dropout_key, p=beam_dropout_prob, shape=raw_distances.shape)
        noisy_distances = jnp.where(is_dropout, jessi.max_beam_range, noisy_distances)  # (n_stack, lidar_num_rays)
        new_hit = jnp.where(noisy_distances < jessi.max_beam_range, 1.0, 0.0) * (1.0 - is_dropout)  # (n_stack, lidar_num_rays)
        x = noisy_distances * data['inputs'][:,:,4]  # (n_stack, lidar_num_rays)
        y = noisy_distances * data['inputs'][:,:,5]  # (n_stack, lidar_num_rays)
        data['inputs'] = data['inputs'].at[:,:,0].set(noisy_distances / jessi.max_beam_range)
        data['inputs'] = data['inputs'].at[:,:,1].set(new_hit)
        data['inputs'] = data['inputs'].at[:,:,2].set(x)
        data['inputs'] = data['inputs'].at[:,:,3].set(y)
        return data
    @loop_tqdm(policy_n_epochs, desc="Training Controller network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        critic_key, dataset, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses = epoch_for_val
        # Shuffle dataset at the beginning of the epoch
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_data)
        shuffled_indexes = random.permutation(shuffle_key, indexes)
        epoch_data = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(shuffled_indexes, dataset)
        # Batch loop
        @jit
        def _batch_loop(
            j:int,
            batch_for_val:tuple
        ) -> tuple:
            critic_key, epoch_data, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(policy_batch_size) + j * policy_batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, epoch_data)
            # Compute training batch
            perception_input, last_lidar_point_clouds = vmap(jessi.compute_perception_input)(batch["observations"])
            ## DATA AUGMENTATION
            noisy_inputs = batch_augment_data(
                {
                    "inputs": perception_input,
                },
                random.split(random.PRNGKey(i * n_train_batches + j), policy_batch_size),
            )
            hcgs, scan_embeddings, _, _ = jessi.perception.apply(
                encoder_params, 
                None, 
                noisy_inputs["inputs"],
            )
            bounding_parameters = vmap(jessi.bound_action_space)(last_lidar_point_clouds)
            actor_input = vmap(jessi.compute_actor_input)(
                batch["observations"][:,:,:11],
                hcgs,
                bounding_parameters,
                batch["rc_robot_goals"],
            )
            train_batch = {
                # Actor
                "actor_inputs": actor_input,
                "scan_embeddings": scan_embeddings,
                "actor_actions": batch["actor_actions"],
                # Critic
                "states": batch["states"],
                "actions_history": batch["actions_history"],
                "humans_goal": batch["humans_goal"],
                "humans_visibility": batch["humans_visibility"],
                "humans_parameters": batch["humans_parameters"],
                "static_obstacles": batch["static_obstacles"],
                "robot_goal": batch["robot_goal"],
                "robot_radius": batch["robot_radius"],
                "v_max": batch["v_max"],
                "wheels_distance": batch["wheels_distance"],
                "wheels_max_linear_acceleration": batch["wheels_max_linear_acceleration"],
                "robot_delay": batch["robot_delay"],
                "action_space_params": batch["action_space_params"],
                "returns": batch["returns"],
            }
            # Update parameters
            critic_key, subkey = random.split(critic_key)
            actor_params, actor_optimizer_state, actor_loss, critic_params, critic_optimizer_state, critic_loss = jessi.update_il(
                actor_params, 
                actor_optimizer, 
                actor_optimizer_state,
                subkey,
                critic_params,
                critic_optimizer,
                critic_optimizer_state,
                train_batch,
                beta_entropy=beta_entropy_il,
            )
            # Save loss
            losses = losses.at[i,j].set(actor_loss)
            actor_losses = actor_losses.at[i,j].set(actor_loss)
            critic_losses = critic_losses.at[i,j].set(critic_loss)
            return critic_key, epoch_data, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses
        n_batches = n_data // policy_batch_size
        critic_key, _, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
            0,
            n_batches,
            _batch_loop,
            (critic_key, epoch_data, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses)
        )
        return critic_key, dataset, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses
    # Epoch loop
    _, _, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
        0,
        policy_n_epochs,
        _epoch_loop,
        (random.PRNGKey(random_seed), controller_dataset, actor_params, actor_optimizer_state, critic_params, critic_optimizer_state, jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))), jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))), jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))))
    )
    # Save trained parameters
    artifact_store.save(ACTOR_MODEL, actor_params, dependencies=(CONTROLLER_DATA, PERCEPTION_MODEL))
    artifact_store.save(CRITIC_MODEL, critic_params, dependencies=(CONTROLLER_DATA,))
    print("Actor losses: ", actor_losses)
    print("Critic losses: ", critic_losses)
    # FREE MEMORY
    del controller_dataset
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    avg_actor_losses = jnp.mean(actor_losses, axis=1)
    avg_critic_losses = jnp.mean(critic_losses, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(jnp.arange(policy_n_epochs), avg_losses, c='red')
    ax[1].plot(jnp.arange(policy_n_epochs), avg_actor_losses, c='green')
    ax[2].plot(jnp.arange(policy_n_epochs), avg_critic_losses, c='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss (with regularization)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Actor Loss (weighted)")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Critic Loss (weighted)")
    fig.savefig(artifact_store.figure_path(ACTOR_MODEL), format='eps')
else:
    # Load trained parameters
    actor_critic_params = artifact_store.load(ACTOR_MODEL, dependencies=(CONTROLLER_DATA, PERCEPTION_MODEL))

### JESSI-MULTITASK: MULTI-TASK REINFORCEMENT LEARNING
rl_dependencies = (PERCEPTION_MODEL, ACTOR_MODEL, CRITIC_MODEL)
if not artifact_store.is_valid(RL_RESULT, dependencies=rl_dependencies):
    print(f"\nSTARTING JESSI-MULTITASK RL TRAINING\nParallel envs {training_hyperparams['rl_parallel_envs']}\nSteps per env {training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_parallel_envs']}\nTotal batch size {training_hyperparams['rl_total_batch_size']}\nMini-batch size {training_hyperparams['rl_mini_batch_size']}\nBatches per update {training_hyperparams['rl_num_batches']}\nMicro-batch size {training_hyperparams['rl_micro_batch_size']}\nTraining updates {training_hyperparams['rl_training_updates']}\nEpochs per update {training_hyperparams['rl_num_epochs']}\n")
    reward_function = make_reward()
    env_params = laser_env_parameters(lidar_noise=True) | {'reward_function': reward_function}
    # Initialize environment
    env = LaserNav(**env_params)
    _, _, obs, info, _ = env.reset(
        random.PRNGKey(training_hyperparams['random_seed']),
        visibility_chance=initial_visibility_chance,
    )
    # Initialize robot policy and vnet params
    policy = JESSI_S2R(
        robot_radius=env_params['robot_radius'],
        v_max=robot_vmax, 
        wheels_distance=robot_wheel_distance,
        dt=env_params['robot_dt'], 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        beam_dropout_rate=0.2,
        embedding_dim=embeddings_dim,
        humans_trajectory_noise_std=humans_trajectory_noise_std,
    )
    # Load pre-trained weights
    il_encoder_params = artifact_store.load(PERCEPTION_MODEL, dependencies=(PERCEPTION_DATA,))
    if no_imitation_learning:
        _, il_actor_params, il_critic_params, _ = policy.init_nns(
            random.PRNGKey(random_seed),
        )
    else:
        il_actor_params = artifact_store.load(ACTOR_MODEL, dependencies=(CONTROLLER_DATA, PERCEPTION_MODEL))
        il_critic_params = artifact_store.load(CRITIC_MODEL, dependencies=(CONTROLLER_DATA,))
    il_network_params = policy.merge_nns_params(il_encoder_params, il_actor_params)
    def label_params(params):
        labels = {}
        for module_name, module_params in params.items():
            if jessi.perception_name in module_name.lower(): 
                label = 'perception'
            elif jessi.actor_name in module_name.lower():
                label = 'actor'
            labels[module_name] = {k: label for k in module_params.keys()}
        return labels
    network_optimizer = optax.multi_transform(
        {
        'perception': optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(
                learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                    init_value=0.,
                    peak_value=training_hyperparams['rl_learning_rate'],
                    end_value=training_hyperparams['rl_learning_rate_final'], 
                    warmup_steps=(training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches']) // 10,
                    decay_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
                ), 
                eps=1e-7, 
            ),
        ),
        'actor': optax.chain(
            optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    init_value=training_hyperparams['rl_learning_rate'], 
                    end_value=training_hyperparams['rl_learning_rate_final'], 
                    transition_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
                    transition_begin=0
                ), 
                eps=1e-7, 
            ),
        )
        },
        label_params(il_network_params)
    )
    critic_network_optimizer = optax.chain(
        # optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
        optax.sgd(learning_rate=training_hyperparams['rl_critic_learning_rate'], momentum=0.9)
    )
    # Initialize RL rollout params
    rl_rollout_params = {
        'initial_actor_parameters': il_network_params,
        'initial_critic_parameters': il_critic_params,
        'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
        'train_updates': training_hyperparams['rl_training_updates'],
        'random_seed': training_hyperparams['random_seed'],
        'actor_network_optimizer': network_optimizer,
        'critic_network_optimizer': critic_network_optimizer,
        'total_batch_size': training_hyperparams['rl_total_batch_size'],
        'mini_batch_size': training_hyperparams['rl_mini_batch_size'],
        'micro_batch_size': training_hyperparams['rl_micro_batch_size'],
        'policy': policy,
        'env': env,
        'clip_range': training_hyperparams['rl_clip_frac'],
        'n_epochs': training_hyperparams['rl_num_epochs'],
        'beta_entropy': training_hyperparams['rl_beta_entropy'],
        'lambda_gae': training_hyperparams['lambda_gae'],
        'initial_visibility': training_hyperparams['initial_visibility_chance'],
        'safety_loss': training_hyperparams['safety_loss'],
        'training_type': "multitask",
        'target_kl': training_hyperparams['target_kl'],
        'debugging': False,
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = jessi_s2r_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    artifact_store.save(RL_RESULT, rl_out, dependencies=rl_dependencies)
    final_actor_params = rl_out["final_actor_params"]
    best_actor_params = rl_out["best_actor_params"]
    final_critic_params = rl_out["final_critic_params"]
    best_critic_params = rl_out["best_critic_params"]
    metrics = rl_out["metrics"]
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            processed_metrics[key] = jnp.array(value)
        if isinstance(value, dict):
            processed_metrics[key] = tree_map(lambda x: jnp.array(x), value)
    # Other metrics
    losses = processed_metrics['losses']
    perception_losses = processed_metrics['perception_losses']
    actor_losses = processed_metrics['actor_losses']
    critic_losses = processed_metrics['critic_losses']
    entropy_losses = processed_metrics['entropy_losses']
    returns_during_rl = processed_metrics['returns']
    success_during_rl = processed_metrics['successes']
    failure_during_rl = processed_metrics['failures']
    timeout_during_rl = processed_metrics['timeouts']
    episodes_during_rl = processed_metrics['episodes']
    stds_during_rl = processed_metrics['stds']
    grad_norms_during_rl = processed_metrics['grad_norm']
    collisions_humans_during_rl = processed_metrics['collisions_humans']
    collisions_obstacles_during_rl = processed_metrics['collisions_obstacles']
    times_to_goal_during_rl = processed_metrics['times_to_goal']
    approx_kl_during_rl = processed_metrics['approx_kl']
    episode_count = jnp.sum(episodes_during_rl)
    window = 10 if rl_training_updates > 1000 else 1
    ## Plot RL training stats
    from matplotlib import rc
    font = {'weight' : 'regular',
            'size'   : 18}
    rc('font', **font)
    figure, ax = plt.subplots(4,3,figsize=(15,15))
    figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95, right=0.95, left=0.1, wspace=0.35)
    # Plot returns during RL
    ax[0,0].grid()
    ax[0,0].set(
        xlabel='Training Update', 
        ylabel=f'Return ({window} upd. window)', 
        title='Return'
    )
    ax[0,0].plot(
        jnp.arange(len(returns_during_rl)-(window-1))+window, 
        jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
    )
    # Plot success, failure, and timeout rates during RL
    valid_episode_updates = episodes_during_rl > 0
    safe_episode_counts = jnp.maximum(episodes_during_rl, 1)
    success_rate_during_rl = jnp.where(valid_episode_updates, success_during_rl / safe_episode_counts, jnp.nan)
    failure_rate_during_rl = jnp.where(valid_episode_updates, failure_during_rl / safe_episode_counts, jnp.nan)
    timeout_rate_during_rl = jnp.where(valid_episode_updates, timeout_during_rl / safe_episode_counts, jnp.nan)
    ax[0,1].grid()
    ax[0,1].set(
        xlabel='Training Update', 
        ylabel=f'Rate ({window} upd. window)', 
        title='Success, Failure, and Timeout rates',
        ylim=(-0.1,1.1)
    )
    ax[0,1].plot(
        jnp.arange(len(success_rate_during_rl)-(window-1))+window, 
        jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Success rate',
        color='g',
    )
    ax[0,1].plot(
        jnp.arange(len(failure_rate_during_rl)-(window-1))+window, 
        jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Failure rate',
        color='r',
    )
    ax[0,1].plot(
        jnp.arange(len(timeout_rate_during_rl)-(window-1))+window, 
        jnp.convolve(timeout_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Timeout rate',
        color='yellow',
    )
    # ax[0,1].legend()
    # Plot time to goal during RL
    ax[0,2].grid()
    ax[0,2].set(
        xlabel='Training Update',
        ylabel=f'Time  ({window} upd. window)',
        title='Time to Goal',
    )
    ax[0,2].plot(
        jnp.arange(len(times_to_goal_during_rl)-(window-1))+window, 
        jnp.convolve(times_to_goal_during_rl, jnp.ones(window,), 'valid') / window,
    )
    # Plot actor loss during RL
    ax[1,0].grid()
    ax[1,0].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Actor Loss'
    )
    ax[1,0].plot(
        jnp.arange(len(actor_losses)-(window-1))+window, 
        jnp.convolve(actor_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot critic loss during RL
    ax[1,1].grid()
    ax[1,1].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Critic Loss'
    )
    ax[1,1].plot(
        jnp.arange(len(critic_losses)-(window-1))+window, 
        jnp.convolve(critic_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot perception loss during RL
    ax[1,2].grid()
    ax[1,2].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Perception Loss'
    )
    ax[1,2].plot(
        jnp.arange(len(perception_losses)-(window-1))+window, 
        jnp.convolve(perception_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot entropy loss during RL
    ax[2,0].grid()
    ax[2,0].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Entropy Loss'
    )
    ax[2,0].plot(
        jnp.arange(len(entropy_losses)-(window-1))+window, 
        jnp.convolve(entropy_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot stds[0] during RL
    ax[2,1].grid()
    ax[2,1].set(
        xlabel='Training Update',
        ylabel=r'$\sigma(v)$',
        title='Std velocity',
        ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
    )
    ax[2,1].plot(
        jnp.arange(len(stds_during_rl)),
        stds_during_rl[:,0],
    )
    # Plot stds[1] during RL
    ax[2,2].grid()
    ax[2,2].set(
        xlabel='Training Update',
        ylabel=r'$\sigma(\omega)$',
        title='Std ang. vel.',
        ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
    )
    ax[2,2].plot(
        jnp.arange(len(stds_during_rl)),
        stds_during_rl[:,1],
    )
    # Plot actor loss std during RL
    ax[3,0].grid()
    ax[3,0].set(
        xlabel='Training Update',
        ylabel='Average Norm',
        title='Gradients L2 norm',
    )
    ax[3,0].plot(
        jnp.arange(len(grad_norms_during_rl[:])),
        grad_norms_during_rl,
    )
    # Plot Total Loss during RL
    ax[3,1].grid()
    ax[3,1].set(
        xlabel='Training Update',
        ylabel=f'Loss  ({window} upd. window)',
        title='Total Loss',
    )
    ax[3,1].plot(
        jnp.arange(len(losses)-(window-1))+window, 
        jnp.convolve(losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot Collisions with humans and obstacles during RL
    ax[3,2].grid()
    ax[3,2].set(
        xlabel='Training Update',
        ylabel='Collisions',
        title='Colls. hum/obs',
    )
    ax[3,2].plot(
        jnp.arange(len(collisions_humans_during_rl)),
        jnp.where(valid_episode_updates, collisions_humans_during_rl/safe_episode_counts, jnp.nan),
        label='Collisions humans',
        color='blue',
    )
    ax[3,2].plot(
        jnp.arange(len(collisions_obstacles_during_rl)),
        jnp.where(valid_episode_updates, collisions_obstacles_during_rl/safe_episode_counts, jnp.nan),
        label='Collisions obstacles',
        color='black',
    )
    figure.savefig(artifact_store.figure_path(RL_RESULT), format='eps')
else:
    print(f"JESSI-MULTITASK RL training already done and saved at {artifact_store.path(RL_RESULT)}")
