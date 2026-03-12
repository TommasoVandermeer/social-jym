from jax import random, vmap, jit
import jax.numpy as jnp
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as Reward
from socialjym.policies.vanilla_e2e import VanillaE2E

# Hyperparameters
save_video = False
random_seed = 0
n_steps_per_episode = 50
n_episodes = 100
kinematics = 'unicycle'
action_space_bounding = False
n_stack_for_action_space_bounding = 1
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

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy_single_gamma = VanillaE2E(
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    action_space_bounding=action_space_bounding,
)
with open(os.path.join(os.path.dirname(__file__), 'vanilla_e2e_single_gamma_rl_out.pkl'), 'rb') as f:
    network_params_single_gamma, _, _ = pickle.load(f)
policy_multi_gamma = VanillaE2E(
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    action_space_bounding=action_space_bounding,
    critic_heads=3,
)
with open(os.path.join(os.path.dirname(__file__), 'vanilla_e2e_multi_gamma_rl_out.pkl'), 'rb') as f:
    network_params_multi_gamma, _, _ = pickle.load(f)

# Define grid
grid_lims = jnp.array([[-10,-10],[10,10]])
cell_size = 0.2
xs = jnp.arange(grid_lims[0, 0], grid_lims[1, 0], cell_size) + (cell_size / 2)
ys = jnp.arange(grid_lims[0, 1], grid_lims[1, 1], cell_size) + (cell_size / 2)
X, Y = jnp.meshgrid(xs, ys, indexing='ij')
flat_grid_positions = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
n_cells_total = flat_grid_positions.shape[0]

# Util functions
@jit
def get_grid_obses(base_state, base_info):
    def _get_single_obs(pos):
        modified_state = base_state.at[-1, 0:2].set(pos)
        base_info["previous_obs"] = vmap(env._get_current_obs, in_axes=(None,None,None,None,0))(
            modified_state,
            base_info['humans_parameters'][:,0],
            base_info['static_obstacles'][-1],
            jnp.zeros((2,)),
            random.split(random.PRNGKey(0), env.n_stack),
        )
        dummy_action = jnp.array([0., 0.])
        return env._get_obs(modified_state, base_info, dummy_action, random.PRNGKey(0))
    return vmap(_get_single_obs)(flat_grid_positions)

#  Simulate some episodes
for i in range(n_episodes):
    reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_value_maps_single_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_value_maps_multi_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_states = jnp.empty((n_steps_per_episode, *state.shape))
    # Collect critic value landscapes
    for j in range(n_steps_per_episode):
        grid_obses = get_grid_obses(state, info) # Shape: (N_cells, n_stack, obs_dim)
        _, _, _, _, _, actor_distrs_single_gamma, state_values_single_gamma = vmap(policy_single_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), grid_obses, info, network_params_single_gamma, False)
        _, _, _, _, _, actor_distrs_multi_gamma, state_values_multi_gamma = vmap(policy_multi_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), grid_obses, info, network_params_multi_gamma, False)
        value_map_single_gamma = state_values_single_gamma.reshape(X.shape[0], X.shape[1])
        value_map_multi_gamma = jnp.sum(state_values_multi_gamma, axis=-1).reshape(X.shape[0], X.shape[1])
        all_value_maps_single_gamma = all_value_maps_single_gamma.at[j].set(value_map_single_gamma)
        all_value_maps_multi_gamma = all_value_maps_multi_gamma.at[j].set(value_map_multi_gamma)
        all_states = all_states.at[j].set(state)
        state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,jnp.array([0.,0.]),test=True,env_key=env_key)
    # Animate landscape
    vmin = min(jnp.min(all_value_maps_single_gamma), jnp.min(all_value_maps_multi_gamma))
    vmax = max(jnp.max(all_value_maps_single_gamma), jnp.max(all_value_maps_multi_gamma))
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, hspace=0., wspace=0.2)
    fig.suptitle(f"Critic Value Landscape - Episode {i+1}")
    im1 = axs[0].pcolormesh(X, Y, all_value_maps_single_gamma[0], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label('Expected Return (Value)')
    def animate(frame):
        axs[0].clear()
        axs[1].clear()
        humans_positions = all_states[frame,:-1,:2]
        humans_orientations = all_states[frame,:-1,4]
        humans_poses = jnp.concatenate((humans_positions, humans_orientations[:,None]), axis=-1)
        humans_radii = info["humans_parameters"][:, 0]
        robot_goal = info["robot_goal"]
        im1 = axs[0].pcolormesh(X, Y, all_value_maps_single_gamma[frame], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
        axs[0].set_title('Single-Discount')
        im2 = axs[1].pcolormesh(X, Y, all_value_maps_multi_gamma[frame], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
        axs[1].set_title('Multi-Discount')
        for ax in axs:
            ax.set_aspect('equal')
            ax.set_xlim(grid_lims[0, 0], grid_lims[1, 0])
            ax.set_ylim(grid_lims[0, 1], grid_lims[1, 1])
            ax.set_xlabel('X', labelpad=-7)
            ax.set_ylabel('Y', labelpad=-20)
            for h in range(len(humans_poses)):
                head = plt.Circle((humans_poses[h,0] + jnp.cos(humans_poses[h,2]) * humans_radii[h], humans_poses[h,1] + jnp.sin(humans_poses[h,2]) * humans_radii[h]), 0.1, color='black', alpha=0.6, zorder=1)
                ax.add_patch(head)
                circle = plt.Circle((humans_poses[h,0], humans_poses[h,1]), humans_radii[h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                ax.add_patch(circle)
            if info['static_obstacles'][-1].shape[1] > 1: # Polygon obstacles
                for o in info['static_obstacles'][-1]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
            else: # One segment obstacles
                for o in info['static_obstacles'][-1]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
            ax.scatter(robot_goal[0], robot_goal[1], color='red', label='Goal', marker='*', s=30)
    anim = FuncAnimation(fig, animate, interval=policy_single_gamma.dt*1000, frames=n_steps_per_episode)
    if save_video:
        save_path = os.path.join(os.path.dirname(__file__), f'{policy_single_gamma.name}_trajectory.mp4')
        writer_video = FFMpegWriter(fps=int(1/policy_single_gamma.dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()