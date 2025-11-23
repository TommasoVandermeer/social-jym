import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from jax import random, lax, vmap
import os
import pickle
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import wrap_angle
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.aux_functions import animate_trajectory, interpolate_humans_boundaries, interpolate_obstacle_segments, plot_state

### Hyperparameters
save_videos = False
trial = 18
n_humans = 30
time_limit = 60.
reward_function = Reward2(
    time_limit=time_limit,
    target_reached_reward = True,
    collision_penalty_reward = True,
    discomfort_penalty_reward = True,
    v_max = 1.,
    progress_to_goal_reward = True,
    progress_to_goal_weight = 0.03,
    high_rotation_penalty_reward=True,
    angular_speed_bound=1.,
    angular_speed_penalty_weight=0.0075,
)

### Load custom episodes
with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans}_humans.pkl'), 'rb') as f:
    custom_episodes = pickle.load(f)

### Add 3 DIRSAFE agents
starting_positions = jnp.array([[0., -7.5], [5., -8.], [13., 8.5], [14, 0.5]])
starting_orientations = jnp.array([0., 0., jnp.pi, -jnp.pi])
final_goals = jnp.array([[9.5, 0.], [0.8, 8.5], [-12.5, -6.], [-7.5, 8.5]])
n_additional_agents = starting_positions.shape[0]
n_robots = n_additional_agents + 1
fictitious_full_states = jnp.zeros((n_additional_agents, custom_episodes["full_state"].shape[2]))
fictitious_full_states = fictitious_full_states.at[:, :2].set(starting_positions)
fictitious_full_states = fictitious_full_states.at[:, 4].set(starting_orientations)

### Find paths to goals for additional agents using A*
from socialjym.utils.global_planners.a_star import AStarPlanner
from socialjym.utils.cell_decompositions.grid import decompose
obstacle_map = custom_episodes["static_obstacles"][trial,-1]
obstacle_points = obstacle_map.reshape((-1,2))
cell_size = 0.95
cells, edges, grid_info = decompose(
    cell_size=cell_size,
    min_grid_size=35.,
    state =custom_episodes["full_state"][trial],
    info = {
        'static_obstacles':  custom_episodes["static_obstacles"][trial],
        'robot_goal': custom_episodes["robot_goals"][trial,0],
    },
    obstacle_points=obstacle_points,
    epsilon=1e-5,
)
## Plot cell decomposition
# for i, row in enumerate(grid_info['grid_cells']):
#     for j, coord in enumerate(row):
#         cell_size = cell_size
#         facecolor = 'red' if grid_info['grid_occupancy'][i,j] else 'none'
#         rect = plt.Rectangle((coord[0]-cell_size/2, coord[1]-cell_size/2), cell_size, cell_size, facecolor=facecolor, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
#         plt.gca().add_patch(rect)
# for o in custom_episodes["static_obstacles"][trial,-1]: plt.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
a_star_planner = AStarPlanner(grid_size=jnp.array([grid_info['grid_cells'].shape[0], grid_info['grid_cells'].shape[1]]))
paths, path_lengths = vmap(a_star_planner.find_path, in_axes=(0,0,None,None))(
    starting_positions,
    final_goals,
    grid_info['grid_cells'],
    grid_info['grid_occupancy'],
)
def downsize_path(path):
    if path.shape[0] < 3:
        return path
    downsized = [path[0]]
    for i in range(1, path.shape[0] - 1):
        prev = path[i - 1]
        curr = path[i]
        next_ = path[i + 1]
        v1 = curr - prev
        v2 = next_ - curr
        # Check if vectors are colinear (cross product close to zero)
        if jnp.abs(jnp.cross(v1, v2)) > 1e-5:
            downsized.append(curr)
    downsized.append(path[-1])
    return jnp.stack(downsized)
paths = [downsize_path(path)[:-1] for path in paths]
paths.append(custom_episodes["robot_goals"][trial])

### Enlarge obstacles
def enlarge_obstacles(obstacles, enlargement_size):
    """
    For each obstacle (single segment), enlarge it to a rectangle (four segments) with the given enlargement size.
    Returns a new array of obstacles, each as a rectangle (4 segments).
    """
    enlarged = []
    for obs in obstacles:
        # obs shape: (1, 2, 2) -> one segment, two endpoints, two coordinates
        p1 = obs[0, 0]
        p2 = obs[0, 1]
        # Direction vector
        d = p2 - p1
        d_norm = d / (jnp.linalg.norm(d) + 1e-8)
        # Perpendicular vector
        perp = jnp.array([-d_norm[1], d_norm[0]])
        # Parallel vector
        parallel = d_norm
        # Offset points (enlarge in both perpendicular and parallel directions)
        p1a = p1 + perp * enlargement_size / 2 - parallel * enlargement_size / 2
        p1b = p1 - perp * enlargement_size / 2 - parallel * enlargement_size / 2
        p2a = p2 + perp * enlargement_size / 2 + parallel * enlargement_size / 2
        p2b = p2 - perp * enlargement_size / 2 + parallel * enlargement_size / 2
        # Rectangle segments: [p1a-p2a], [p2a-p2b], [p2b-p1b], [p1b-p1a]
        rect = jnp.array([
            [p1a, p2a],
            [p2a, p2b],
            [p2b, p1b],
            [p1b, p1a]
        ])
        enlarged.append(rect[None, ...])  # shape (1, 4, 2, 2)
    return jnp.concatenate(enlarged, axis=0)
obstacles = custom_episodes["static_obstacles"][trial,-1]  # Get the last timestep obstacles
enlarged_obstacles = obstacles #enlarge_obstacles(obstacles, enlargement_size=0.2)
n_agents = len(custom_episodes["static_obstacles"][trial])
stacked_obstacles = jnp.stack([enlarged_obstacles for _ in range(n_agents + n_additional_agents)], axis=0)  # shape: (n_agents, n_obstacles, 4, 2, 2)

### Visualize initial configuration
humans_radiuses = custom_episodes['humans_radius'][trial, :]
full_state = custom_episodes['full_state'][trial, :, :]
colors = list(mcolors.TABLEAU_COLORS.values())[1:]
xlims = [jnp.nanmin(stacked_obstacles[-1][:,:,:,0]), jnp.nanmax(stacked_obstacles[-1][:,:,:,0])]
ylims = [jnp.nanmin(stacked_obstacles[-1][:,:,:,1]), jnp.nanmax(stacked_obstacles[-1][:,:,:,1])]
figure, ax = plt.subplots(1,1, figsize=(8,8))
figure.subplots_adjust(left=0.09, right=0.85, top=0.99, bottom=0.05)
ax.set_xlim(xlims[0], xlims[1])
ax.set_ylim(ylims[0], ylims[1])
ax.set_xlabel('X')
ax.set_ylabel('Y', labelpad=-13)
for h in range(len(full_state)-1): 
        head = plt.Circle((full_state[h,0] + jnp.cos(full_state[h,4]) * humans_radiuses[h], full_state[h,1] + jnp.sin(full_state[h,4]) * humans_radiuses[h]), 0.1, color="blue", zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor="blue", facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
for o in obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
for i, path in enumerate(paths):
    ax.plot(path[:,0], path[:,1], linestyle='--', color=colors[i % len(colors)], linewidth=2, zorder=2)
    for j in range(len(path)-1):
        ax.scatter(path[j,0], path[j,1], marker='.', color=colors[i % len(colors)], s=50, zorder=4)
    head = plt.Circle((paths[i][0][0] + jnp.cos(starting_orientations[i]) * humans_radiuses[i], paths[i][0][1] + jnp.sin(starting_orientations[i]) * humans_radiuses[i]), 0.1, color='black', zorder=1)
    ax.add_patch(head)
    circle = plt.Circle((paths[i][0][0],paths[i][0][1]),humans_radiuses[i], edgecolor='black', facecolor=colors[i % len(colors)], fill=True, zorder=1)
    ax.add_patch(circle)
    ax.scatter(path[-1,0], path[-1,1], marker='*', color=colors[i % len(colors)], s=150, zorder=4)  # Goal
ax.set_aspect('equal')
handles = \
    [Line2D([0], [0], color='white', marker='o', markersize=11, markerfacecolor=colors[i % len(colors)], markeredgecolor='black', linewidth=2, label='Robot '+str(i)) for i in range(n_robots)] + \
    [Line2D([0], [0], color='white', marker='o', markersize=11, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans')]
ax.legend(
    # title=f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))}",
    handles=handles,
    loc='center left',    
    bbox_to_anchor=(1., .5),
    fontsize=20,
    title_fontsize=7,
)
plt.show()

### Initialize environment
test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans + n_additional_agents,
    'n_obstacles': len(stacked_obstacles[0]),
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': None, # Custom scenario
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 0,
}
test_env = SocialNav(**test_env_params)

### Run custom episodes
## Initialize robot policy
policy = DIRSAFE(reward_function, v_max=1., dt=0.25)
with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']
## Reset the environment
state, _, obs, info, outcome = test_env.reset_custom_episode(
    random.PRNGKey(0), # Not used, but required by the function
    {
        "full_state": jnp.concatenate([custom_episodes["full_state"][trial][:-1], fictitious_full_states, custom_episodes["full_state"][trial][-1][None, :]], axis=0),
        "robot_goal": custom_episodes["robot_goals"][trial,0],
        "humans_goal": jnp.concatenate([custom_episodes["humans_goal"][trial], final_goals], axis=0),
        "static_obstacles": stacked_obstacles,
        "scenario": -1,
        "humans_radius": jnp.concatenate([custom_episodes["humans_radius"][trial], jnp.ones(n_additional_agents) * 0.3], axis=0),
        "humans_speed": jnp.concatenate([custom_episodes["humans_speed"][trial], jnp.ones(n_additional_agents) * 1.], axis=0),
    }
)
paths = [list(path[1:]) for path in paths]  # Convert to list for easy popping
## Run the episode
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'multi_agent_dir_safe_episode.pkl')):
    all_states = jnp.array([state])
    all_robot_goals = jnp.array([[path[0] for path in paths]])
    previous_actions = jnp.zeros((n_robots, 2))
    while info['time'] < time_limit:
        print(f"Time: {info['time']:.2f}s - Remaining waypoints for robots: {[len(path) for path in paths]}")
        # Update humans goal
        info["humans_goal"] = lax.fori_loop(
            0, 
            n_humans, 
            lambda h, x: lax.cond(
                jnp.linalg.norm(state[h,:2] - info["humans_goal"][h]) <= info["humans_parameters"][h,0],
                lambda y: lax.cond(
                    jnp.all(jnp.isclose(info["humans_goal"][h], custom_episodes["humans_goal"][trial,h])),
                    lambda z: z.at[h].set(custom_episodes["full_state"][trial,h,:2]),
                    lambda z: z.at[h].set(custom_episodes["humans_goal"][trial,h]),
                    y,
                ),
                lambda y: y,
                x
            ),
            info["humans_goal"],
        )
        # Update robots velocities
        for robot in range(n_robots):
            state = state.at[n_humans + robot].set(jnp.array([*state[n_humans + robot,:2], previous_actions[robot,0], 0., state[n_humans + robot,4], previous_actions[robot,1]]))
        # Compute robots actions
        robots_state = jnp.copy(state[n_humans:])
        new_actions = jnp.zeros((n_robots, 2))
        for robot in range(n_robots):
            aux_info = info.copy()
            # Update robot goal
            if (jnp.linalg.norm(state[n_humans + robot,:2] - paths[robot][0]) <= test_env.robot_radius*2) & \
                (len(paths[robot]) > 1):
                paths[robot].pop(0)
            aux_info['robot_goal'] = paths[robot][0]
            # Rearrange observation for action computation
            aux_state = jnp.copy(state)
            temp = aux_state[n_humans + robot].copy()
            aux_state = aux_state.at[n_humans + robot].set(aux_state[-1])
            aux_state = aux_state.at[-1].set(temp)
            aux_obs = test_env._get_obs(aux_state, aux_info, previous_actions[robot])
            # Step the environment
            action, _, _, _, _ = policy.act(random.PRNGKey(0), aux_obs, aux_info, actor_params, sample=False)
            # Apply action
            robots_state = robots_state.at[robot].set(lax.cond(
                    jnp.abs(action[1]) > 1e-5,
                    lambda x: x.at[:].set(jnp.array([
                        robots_state[robot,0]+(action[0]/action[1])*(jnp.sin(robots_state[robot,4]+action[1]*test_env.robot_dt)-jnp.sin(robots_state[robot,4])),
                        robots_state[robot,1]+(action[0]/action[1])*(jnp.cos(robots_state[robot,4])-jnp.cos(robots_state[robot,4]+action[1]*test_env.robot_dt)),
                        *robots_state[robot,2:4],
                        wrap_angle(robots_state[robot,4]+action[1]*test_env.robot_dt),
                        robots_state[robot,5]])),
                    lambda x: x.at[:].set(jnp.array([
                        robots_state[robot,0]+action[0]*test_env.robot_dt*jnp.cos(robots_state[robot,4]),
                        robots_state[robot,1]+action[0]*test_env.robot_dt*jnp.sin(robots_state[robot,4]),
                        *robots_state[robot,2:]])),
                    robots_state[robot]))
            new_actions = new_actions.at[robot].set(action)
        # Update final state
        state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
        state = state.at[n_humans:, :].set(robots_state)
        previous_actions = new_actions
        # Save the state
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([[path[0] for path in paths]])))
        with open(os.path.join(os.path.dirname(__file__), 'multi_agent_dir_safe_episode.pkl'), 'wb') as f:
            pickle.dump({
                'all_states': all_states,
                'all_robot_goals': all_robot_goals,
            }, f)
else:
    with open(os.path.join(os.path.dirname(__file__), 'multi_agent_dir_safe_episode.pkl'), 'rb') as f:
        data = pickle.load(f)
        all_states = data['all_states']
        all_robot_goals = data['all_robot_goals']
### Plot trajectory
figure, ax = plt.subplots(1,1, figsize=(16.04,9.11))
figure.subplots_adjust(left=0.09, right=0.85, top=0.99, bottom=0.05)
ax.set_xlim(xlims[0], xlims[1])
ax.set_ylim(ylims[0], ylims[1])
ax.set_xlabel('X')
ax.set_ylabel('Y', labelpad=-13)
for h in range(len(full_state)-1): 
        head = plt.Circle((full_state[h,0] + jnp.cos(full_state[h,4]) * humans_radiuses[h], full_state[h,1] + jnp.sin(full_state[h,4]) * humans_radiuses[h]), 0.1, color="blue", zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor="blue", facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
for o in obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
for i in range(n_robots):
    ax.plot(all_states[:, n_humans + i, 0], all_states[:, n_humans + i, 1], linestyle='--', color=colors[i % len(colors)], linewidth=2, zorder=2)
    head = plt.Circle((all_states[0, n_humans + i, 0] + jnp.cos(starting_orientations[i]) * humans_radiuses[i], all_states[0, n_humans + i, 1] + jnp.sin(starting_orientations[i]) * humans_radiuses[i]), 0.1, color='black', zorder=1)
    ax.add_patch(head)
    circle = plt.Circle((all_states[0, n_humans + i, 0],all_states[0, n_humans + i, 1]),humans_radiuses[i], edgecolor='black', facecolor=colors[i % len(colors)], fill=True, zorder=1)
    ax.add_patch(circle)
    ax.scatter(paths[i][-1][0], paths[i][-1][1], marker='*', color=colors[i % len(colors)], s=150, zorder=4)  # Goal
ax.set_aspect('equal')
handles = \
    [Line2D([0], [0], color='white', marker='o', markersize=11, markerfacecolor=colors[i % len(colors)], markeredgecolor='black', linewidth=2, label='Robot '+str(i)) for i in range(n_robots)] + \
    [Line2D([0], [0], color='white', marker='o', markersize=11, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans')]
ax.legend(
    # title=f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))}",
    handles=handles,
    loc='center left',    
    bbox_to_anchor=(1., .5),
    fontsize=20,
    title_fontsize=7,
)
figure.savefig(os.path.join(os.path.dirname(__file__), f'multi_agent_dir_safe_trajectory.eps'), dpi=300)
plt.show()
### Animate trajectory
rc('font', weight='regular', size=9)
handles = \
    [Line2D([0], [0], color='white', marker='o', markersize=7, markerfacecolor=colors[i % len(colors)], markeredgecolor='black', linewidth=2, label='Robot '+str(i)) for i in range(n_robots)] + \
    [Line2D([0], [0], color='white', marker='o', markersize=7, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans')]
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.subplots_adjust(left=0.05, right=0.99, wspace=0.13)
def animate(frame):
    ax.clear()
    ax.legend(
        title=f"Time: {'{:.2f}'.format(round(frame*test_env.robot_dt,2))}",
        ncol=3,
        handles=handles,
        loc='lower center',    
        bbox_to_anchor=(.5, 1.),
        fontsize=7,
        title_fontsize=7,
    )
    ax.set(xlim=[xlims[0], xlims[1]], ylim=[ylims[0], ylims[1]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y', labelpad=-13)
    ax.set_aspect('equal', adjustable='box')
    # Plot humans
    for h in range(n_humans):
        human_pos = all_states[frame][h,:2]
        human_yaw = all_states[frame][h,4]
        color = "blue"
        head = plt.Circle((human_pos[0] + jnp.cos(human_yaw) * info['humans_parameters'][h,0], human_pos[1] + jnp.sin(human_yaw) * info['humans_parameters'][h,0]), 0.1, color='black', zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((human_pos[0], human_pos[1]), info['humans_parameters'][h,0], edgecolor='black', facecolor=color, fill=True, zorder=1)
        ax.add_patch(circle)
    # Plot robots and their goals
    for r in range(n_robots):
        color = colors[r % len(colors)]
        robot_state = all_states[frame][n_humans + r]
        head = plt.Circle((robot_state[0] + 0.3 * jnp.cos(robot_state[4]), robot_state[1] + 0.3 * jnp.sin(robot_state[4])), 0.1, color='black', zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((robot_state[0], robot_state[1]), 0.3, edgecolor="black", facecolor=color, fill=True, zorder=3)
        ax.add_patch(circle)
        # Plot robot goals
        ax.plot(
            all_robot_goals[frame][r,0],
            all_robot_goals[frame][r,1],
            marker='*',
            markersize=7,
            color=color,
            zorder=5,
        )
    # Plot static obstacles
    for o in obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
anim = FuncAnimation(fig, animate, interval=test_env.robot_dt*1000, frames=len(all_states))
if save_videos:
    save_path = os.path.join(os.path.dirname(__file__), f'multi_agent_dir_safe.mp4')
    writer_video = FFMpegWriter(fps=int(1/test_env.robot_dt), bitrate=1800)
    anim.save(save_path, writer=writer_video, dpi=300)
anim.paused = False
def toggle_pause(self, *args, **kwargs):
    if anim.paused: anim.resume()
    else: anim.pause()
    anim.paused = not anim.paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)
plt.show()