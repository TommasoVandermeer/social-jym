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
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as DReward
from socialjym.utils.rewards.lasernav_rewards.reward3 import Reward3 as Reward
from socialjym.policies.vanilla_e2e import VanillaE2E

# Hyperparameters
save_video = False
random_seed = 0
n_steps_per_episode = 50
n_episodes = 100
kinematics = 'unicycle'
action_space_bounding = True
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
    'scenario': 'parallel_traffic', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 0,
    'reward_function': DReward(robot_radius=0.3),
    'kinematics': kinematics,
    'lidar_noise': False,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the single gamma policy
reward_fun_single_gamma = Reward(
    robot_radius=0.3,
    collision_with_humans_penalty=-4.,
    collision_with_obstacles_penalty=-0.8,
    progress_to_goal_weight=0.2,
    angular_speed_penalty_weight=0.1,
    gamma=0.9,
)
policy_single_gamma = VanillaE2E(
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    action_space_bounding=action_space_bounding,
    critic_heads=len(reward_fun_single_gamma.unique_gammas),
)
with open(os.path.join(os.path.dirname(__file__), 'bounded_vanilla_e2e_single_gamma_rl_out.pkl'), 'rb') as f:
    network_params_single_gamma, _, _ = pickle.load(f)
# Initialize the multi gamma policy
reward_fun_multi_gamma = Reward(
    robot_radius=0.3,
    collision_with_humans_penalty=-4.,
    collision_with_obstacles_penalty=-0.8,
    progress_to_goal_weight=0.2,
    angular_speed_penalty_weight=0.1,
    gamma=[0.3,0.3,0.3,0.5,0.5,0.9],
)
policy_multi_gamma = VanillaE2E(
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
    action_space_bounding=action_space_bounding,
    critic_heads=len(reward_fun_multi_gamma.unique_gammas),
)
with open(os.path.join(os.path.dirname(__file__), 'bounded_vanilla_e2e_multi_gamma_rl_out.pkl'), 'rb') as f:
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
def get_grid_current_obses(current_state, base_info):
    def _get_single_current_obs(pos):
        modified_state = current_state.at[-1, 0:2].set(pos)
        diff = base_info['robot_goal'] - pos
        modified_state = modified_state.at[-1, 4].set(jnp.atan2(diff[1],diff[0]))
        obs = env._get_current_obs(
            modified_state,
            modified_state,
            base_info['humans_parameters'][:,0],
            base_info['static_obstacles'][-1],
            jnp.zeros((2,)), # Dummy actions
            random.PRNGKey(0),
        )
        return obs
    return vmap(_get_single_current_obs)(flat_grid_positions)
@jit
def get_grid_next_obses_forward_action(current_state, base_info, action):
    def _get_single_current_obs(pos):
        modified_state = current_state.at[-1, 0:2].set(pos)
        diff = base_info['robot_goal'] - pos
        modified_state = modified_state.at[-1, 4].set(jnp.atan2(diff[1],diff[0]))
        next_state, _, _, _, _, _ = env.step(state,info,action,test=True,env_key=random.PRNGKey(0))
        obs = env._get_current_obs(
            next_state,
            next_state,
            base_info['humans_parameters'][:,0],
            base_info['static_obstacles'][-1],
            action,
            random.PRNGKey(0),
        )
        return obs
    return vmap(_get_single_current_obs)(flat_grid_positions)
@jit
def get_grid_states(current_state, base_info):
    def _get_single_current_state(pos):
        modified_state = current_state.at[-1, :2].set(pos)
        diff = base_info['robot_goal'] - pos
        modified_state = modified_state.at[-1, 4].set(jnp.atan2(diff[1],diff[0]))
        return modified_state
    return vmap(_get_single_current_state)(flat_grid_positions)

#  Simulate some episodes
for i in range(n_episodes):
    reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    state, reset_key, _, info, outcome = env.reset(reset_key)
    all_value_maps_single_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_value_maps_multi_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_advantage_maps_single_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_advantage_maps_multi_gamma = jnp.empty((n_steps_per_episode,X.shape[0], X.shape[1]))
    all_states = jnp.empty((n_steps_per_episode, *state.shape))
    init_grid_obs = get_grid_current_obses(state, info)
    grid_obses = jnp.repeat(init_grid_obs[:, None, :], env.n_stack, axis=1)
    # Collect critic value landscapes
    for j in range(n_steps_per_episode):
        print(f"Step: {j}/{n_steps_per_episode}")
        current_grid_obs = get_grid_current_obses(state, info)
        grid_obses = jnp.concatenate(
            [current_grid_obs[:, None, :], grid_obses[:, :-1, :]], 
            axis=1
        ) # Shape: (N_cells, n_stack, obs_dim)
        next_grid_obs = get_grid_next_obses_forward_action(state, info, jnp.array([policy_single_gamma.v_max,0.]))
        next_grid_obses = jnp.concatenate(
            [next_grid_obs[:, None, :], grid_obses[:, :-1, :]], 
            axis=1
        ) # Shape: (N_cells, n_stack, obs_dim)
        _, _, _, _, _, _, state_values_single_gamma = vmap(policy_single_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), grid_obses, info, network_params_single_gamma, False)
        _, _, _, _, _, _, state_values_multi_gamma = vmap(policy_multi_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), grid_obses, info, network_params_multi_gamma, False)
        _, _, _, _, _, _, next_state_values_single_gamma = vmap(policy_single_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), next_grid_obses, info, network_params_single_gamma, False)
        _, _, _, _, _, _, next_state_values_multi_gamma = vmap(policy_multi_gamma.act, in_axes=(None,0,None,None,None))(random.PRNGKey(0), next_grid_obses, info, network_params_multi_gamma, False)
        value_map_single_gamma = state_values_single_gamma.reshape(X.shape[0], X.shape[1])
        value_map_multi_gamma = jnp.sum(state_values_multi_gamma, axis=-1).reshape(X.shape[0], X.shape[1])
        next_value_map_single_gamma = next_state_values_single_gamma.reshape(X.shape[0], X.shape[1])
        next_value_map_multi_gamma = next_state_values_multi_gamma.reshape(X.shape[0], X.shape[1], -1)
        statesss = get_grid_states(state, info)
        rewards_single_gamma,_,_ = vmap(reward_fun_single_gamma, in_axes=(0, None, None, None))(
            statesss,
            jnp.array([policy_single_gamma.v_max,0.]),
            info,
            env.robot_dt
        )
        _,_,rewards_multi_gamma = vmap(reward_fun_multi_gamma, in_axes=(0, None, None, None))(
            statesss,
            jnp.array([policy_single_gamma.v_max,0.]),
            info,
            env.robot_dt
        )
        advantage_single = rewards_single_gamma + (reward_fun_single_gamma.gamma * next_state_values_single_gamma) - state_values_single_gamma
        advantage_map_single_gamma = advantage_single.reshape(X.shape[0], X.shape[1])
        gammas_array = jnp.array(reward_fun_multi_gamma.unique_gammas)
        rewards_matrix = jnp.stack([rewards_multi_gamma[g] for g in reward_fun_multi_gamma.unique_gammas], axis=-1) # Shape: (n_envs, n_gammas)
        advantage_per_head = rewards_matrix + (gammas_array * next_state_values_multi_gamma) - state_values_multi_gamma
        total_advantage_multi_gamma = jnp.sum(advantage_per_head, axis=-1)
        advantage_map_multi_gamma = total_advantage_multi_gamma.reshape(X.shape[0], X.shape[1])
        # Save maps
        all_value_maps_single_gamma = all_value_maps_single_gamma.at[j].set(value_map_single_gamma)
        all_value_maps_multi_gamma = all_value_maps_multi_gamma.at[j].set(value_map_multi_gamma)
        all_advantage_maps_single_gamma = all_advantage_maps_single_gamma.at[j].set(advantage_map_single_gamma)
        all_advantage_maps_multi_gamma = all_advantage_maps_multi_gamma.at[j].set(advantage_map_multi_gamma)
        all_states = all_states.at[j].set(state)
        state, _, info, (reward, _), outcome, (_, env_key) = env.step(state,info,jnp.array([0.,0.]),test=True,env_key=env_key)
    # Animate landscape
    vmin = min(jnp.min(all_value_maps_single_gamma), jnp.min(all_value_maps_multi_gamma))
    vmax = max(jnp.max(all_value_maps_single_gamma), jnp.max(all_value_maps_multi_gamma))
    flat_adv_s = all_advantage_maps_single_gamma.flatten()
    flat_adv_m = all_advantage_maps_multi_gamma.flatten()
    p_min = min(jnp.percentile(flat_adv_s, 2), jnp.percentile(flat_adv_m, 2))
    p_max = max(jnp.percentile(flat_adv_s, 98), jnp.percentile(flat_adv_m, 98))
    max_abs_adv = max(abs(p_min), abs(p_max))
    max_abs_adv = max(max_abs_adv, 0.05) 
    sym_amin, sym_amax = -max_abs_adv, max_abs_adv
    fig, axs = plt.subplots(2, 2, figsize=(13,12), constrained_layout=False) # Altezza aumentata per far spazio a due righe
    fig.subplots_adjust(left=0.05, right=0.88, top=0.98, bottom=0.05, hspace=0., wspace=0.)
    im_val = axs[0, 0].pcolormesh(X, Y, all_value_maps_single_gamma[0], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
    im_adv = axs[1, 0].pcolormesh(X, Y, all_advantage_maps_single_gamma[0], vmin=sym_amin, vmax=sym_amax, cmap='RdBu', shading='nearest')
    cbar_val = fig.colorbar(im_val, ax=axs[0, :], orientation='vertical', fraction=0.03, pad=0.02)
    cbar_val.set_label('Expected return (value)',fontsize=18)
    cbar_adv = fig.colorbar(im_adv, ax=axs[1, :], orientation='vertical', fraction=0.03, pad=0.02, extend='both')
    cbar_adv.set_label('Advantage (forward action)',fontsize=18)
    def animate(frame):
        for ax in axs.flat: ax.clear()
        humans_positions = all_states[frame,:-1,:2]
        humans_orientations = all_states[frame,:-1,4]
        humans_poses = jnp.concatenate((humans_positions, humans_orientations[:,None]), axis=-1)
        humans_radii = info["humans_parameters"][:, 0]
        robot_goal = info["robot_goal"]
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
        axs[0, 0].pcolormesh(X, Y, all_value_maps_single_gamma[frame], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
        axs[0, 0].text(0.5, 0.96, 'Value (single-discount)', transform=axs[0, 0].transAxes,fontsize=18, ha='center', va='top', bbox=bbox_props)
        axs[0, 1].pcolormesh(X, Y, all_value_maps_multi_gamma[frame], vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest')
        axs[0, 1].text(0.5, 0.96, 'Value (multi-discount)', transform=axs[0, 1].transAxes,fontsize=18, ha='center', va='top', bbox=bbox_props)
        # ADVANTAGE MAPS
        axs[1, 0].pcolormesh(X, Y, all_advantage_maps_single_gamma[frame], vmin=sym_amin, vmax=sym_amax, cmap='RdBu', shading='nearest')
        axs[1, 0].text(0.5, 0.96, 'Adv. (single-discount)', transform=axs[1, 0].transAxes,fontsize=18, ha='center', va='top', bbox=bbox_props)
        axs[1, 1].pcolormesh(X, Y, all_advantage_maps_multi_gamma[frame], vmin=sym_amin, vmax=sym_amax, cmap='RdBu', shading='nearest')
        axs[1, 1].text(0.5, 0.96, 'Adv. (multi-discount)', transform=axs[1, 1].transAxes,fontsize=18, ha='center', va='top', bbox=bbox_props)
        for idx, ax in enumerate(axs.flat):
            i = idx // 2
            j = idx % 2
            ax.set_aspect('equal')
            ax.set_xlim(grid_lims[0, 0], grid_lims[1, 0])
            ax.set_ylim(grid_lims[0, 1], grid_lims[1, 1])
            if i == 0:
                ax.tick_params(labelbottom=False) 
                ax.set_xlabel('')
                ax.spines['bottom'].set_visible(False)
            else:
                ax.tick_params(labelbottom=True, bottom=True)
                ax.set_xlabel('X', labelpad=-7)
                ax.spines['top'].set_visible(False)
            if j == 1:
                ax.tick_params(labelleft=False) 
                ax.set_ylabel('')
                ax.spines['left'].set_visible(False)
            else:
                ax.tick_params(labelleft=True, left=True)
                ax.set_ylabel('Y', labelpad=-20)
                ax.spines['right'].set_visible(False)
            for h in range(len(humans_poses)):
                head = plt.Circle((humans_poses[h,0] + jnp.cos(humans_poses[h,2]) * humans_radii[h], humans_poses[h,1] + jnp.sin(humans_poses[h,2]) * humans_radii[h]), 0.1, color='black', alpha=0.6, zorder=1)
                ax.add_patch(head)
                circle = plt.Circle((humans_poses[h,0], humans_poses[h,1]), humans_radii[h], edgecolor='black', facecolor='black', alpha=0.6, fill=True, zorder=1)
                ax.add_patch(circle)
            if info['static_obstacles'][-1].shape[1] > 1: # Polygon obstacles
                for o in info['static_obstacles'][-1]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
            else: # One segment obstacles
                for o in info['static_obstacles'][-1]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
            ax.scatter(robot_goal[0], robot_goal[1], color='magenta', label='Goal', marker='*', s=150, zorder=4)
    anim = FuncAnimation(fig, animate, interval=policy_single_gamma.dt*1000, frames=n_steps_per_episode)
    if save_video:
        save_path = os.path.join(os.path.dirname(__file__), f'value_adv_landscape.mp4')
        writer_video = FFMpegWriter(fps=int(1/policy_single_gamma.dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()