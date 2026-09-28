"""
Interactive frame-by-frame trajectory generator for LaserNav (clean corridor, head-on).

Drive the robot manually to generate reference trajectories used to design a
proximity-to-humans reward. Two trajectories are meant to be produced:
  - "active_avoidance":      robot deviates early, never cuts the human's path.
  - "last_moment_avoidance": robot heads straight at the human, swerves at the last moment.

Controls (persistent (v, w) command; SPACE advances exactly one robot_dt step):
  up    / w : v += DV            down  / s : v -= DV
  left  / a : w += DW            right / d : w -= DW
  (the (v, w) command is projected into the unicycle feasible triangle:
   |w| <= W_MAX*(1 - v/V_MAX), so at v = V_MAX the robot cannot turn)
  x        : v = 0   (stop)      c        : w = 0  (go straight)
  SPACE    : advance one frame with the current (v, w)
  z / backspace : undo last frame
  p        : save trajectory to pickle
  v        : replay saved trajectory with animate_trajectory
  h        : print this help
"""
import os
import pickle

import numpy as np
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import plot_state, plot_lidar_measurements, animate_trajectory

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
TRAJECTORY_NAME = "last_moment_avoidance"  # change to "last_moment_avoidance" for the 2nd run
SEED = 0
OUT_DIR = os.path.dirname(__file__)

TRAFFIC_LENGTH = 14.0
TRAFFIC_HEIGHT = 3.0
HUMAN_RADIUS = 0.3
HUMAN_SPEED = 1.0

V_MAX = 1.0             # max linear velocity (m/s)
WHEELS_DISTANCE = 0.7   # differential-drive wheelbase (as in the JESSI policy default)
# Unicycle (differential-drive) feasible action set is the triangle with vertices
# (v=0, w=+W_MAX), (v=0, w=-W_MAX), (v=V_MAX, w=0). Hence |w| <= W_MAX*(1 - v/V_MAX):
# at full speed the robot cannot turn.
W_MAX = 2 * V_MAX / WHEELS_DISTANCE
DV = 0.1      # linear velocity increment per key press
DW = 0.2      # angular velocity increment per key press

env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': 2 * jnp.pi,
    'lidar_max_dist': 10.,
    'n_humans': 1,
    'n_obstacles': 2,                 # the two corridor side walls
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,            # humans (HSFM) react to the robot
    'tau_linear_velocity': 0.39,
    'tau_angular_velocity': 0.19,
    'control_delay_mean': 0.0,
    'control_delay_sigma': 0.0,
    'scenario': None,                 # custom scenario -> reset_custom_episode
    'reward_function': DummyReward(robot_radius=0.3, time_limit=120.),
    'kinematics': 'unicycle',
    'leg_dynamics': False,
}

# ---------------------------------------------------------------------------- #
# Environment + custom corridor episode
# ---------------------------------------------------------------------------- #
env = LaserNav(**env_params)

robot_start = jnp.array([-TRAFFIC_LENGTH / 2 + 1, 0.])   # (-6, 0)
robot_goal = jnp.array([TRAFFIC_LENGTH / 2 - 1, 0.])     # ( 6, 0)
human_start = jnp.array([TRAFFIC_LENGTH / 2 - 2, 0.])    # ( 5, 0)
human_goal = jnp.array([-TRAFFIC_LENGTH / 2 - 3, 0.])    # (-10, 0)

# full_state: row 0 = human, row 1 = robot. State layout: [px, py, vx, vy, theta, omega]
full_state = jnp.array([
    [human_start[0], human_start[1], 0., 0., jnp.pi, 0.],  # human faces -x (toward robot)
    [robot_start[0], robot_start[1], 0., 0., 0., 0.],      # robot faces +x
])

# Clean corridor: only the two side walls (no center/diagonal obstacles).
wall_y = TRAFFIC_HEIGHT / 2 + 0.3  # 1.8
wall_x0 = -TRAFFIC_LENGTH / 2 - 1  # -8.0
wall_x1 = TRAFFIC_LENGTH / 2 - 0.5  # 6.5
obstacles = jnp.array([
    [[[wall_x0, wall_y], [wall_x1, wall_y]]],     # top wall    -> shape (1, 2, 2)
    [[[wall_x0, -wall_y], [wall_x1, -wall_y]]],   # bottom wall
])  # shape (n_obstacles=2, 1, 2, 2)
static_obstacles = jnp.repeat(obstacles[None], env_params['n_humans'] + 1, axis=0)  # (2, 2, 1, 2, 2)

custom_episode = {
    "full_state": full_state,
    "humans_goal": human_goal[None],          # (1, 2)
    "robot_goal": robot_goal,
    "static_obstacles": static_obstacles,
    "scenario": -1,
    "humans_radius": jnp.array([HUMAN_RADIUS]),
    "humans_speed": jnp.array([HUMAN_SPEED]),
}

key = random.PRNGKey(SEED)
key, env_key = random.split(key)
state, key, obs, info, outcome = env.reset_custom_episode(key, custom_episode)

# JIT warm-up (first env.step compiles): throwaway step, result discarded.
print("Compiling env.step (JIT warm-up)...")
_ = env.step(state, info, jnp.array([0., 0.]), test=True, env_key=random.PRNGKey(123))
print("Done. Window is interactive (press 'h' for help).")

# ---------------------------------------------------------------------------- #
# Live episode state
# ---------------------------------------------------------------------------- #
cmd = {'v': 0.0, 'w': 0.0}


def clamp_cmd():
    """Project the (v, w) command into the unicycle feasible triangle."""
    cmd['v'] = float(np.clip(cmd['v'], 0.0, V_MAX))
    wlim = W_MAX * (1.0 - cmd['v'] / V_MAX)
    cmd['w'] = float(np.clip(cmd['w'], -wlim, wlim))
    return wlim
sim = {
    'state': state,
    'info': info,
    'obs': obs,
    'env_key': env_key,
    'outcome': outcome,
    'done': False,
}
all_states = [np.asarray(state)]
all_actions = []
all_obs = [np.asarray(obs)]
undo_stack = []  # list of (state, info, obs, env_key, outcome, done, len_actions)

NRAYS = env.lidar_num_rays
LRANGE = env.lidar_angular_range


def _lidar_from_obs(o, robot_yaw):
    dists = jnp.asarray(o[0, 11:])
    angles = jnp.linspace(robot_yaw - LRANGE / 2, robot_yaw + LRANGE / 2, NRAYS)
    return jnp.stack((dists, angles), axis=-1)


fig, ax = plt.subplots(figsize=(10, 5))


def redraw():
    ax.clear()
    s = sim['state']
    o = sim['obs']
    # Corridor walls
    so = np.asarray(sim['info']['static_obstacles'][-1])  # (n_obstacles, 1, 2, 2)
    for ob in so:
        seg = ob[0]
        ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], color="black", linewidth=2, zorder=1)
    # Lidar
    plot_lidar_measurements(ax, _lidar_from_obs(o, s[-1, 4]), s[-1], env.robot_radius)
    # Human + robot
    plot_state(
        ax, float(sim['info']['time']), s, 'hsfm', -1,
        np.asarray(sim['info']['humans_parameters'][:, 0]), env.robot_radius,
        plot_time=False, kinematics='unicycle',
    )
    # Trajectory traces
    hist = np.asarray(all_states)
    ax.plot(hist[:, 0, 0], hist[:, 0, 1], color="blue", linewidth=1, zorder=0)
    ax.plot(hist[:, 1, 0], hist[:, 1, 1], color="red", linewidth=1, zorder=0)
    # Goals
    rg = np.asarray(sim['info']['robot_goal'])
    hg = np.asarray(sim['info']['humans_goal'][0])
    ax.scatter(rg[0], rg[1], marker="*", color="red", s=120, zorder=2)
    ax.scatter(hg[0], hg[1], marker="*", color="blue", s=120, zorder=2)
    # HUD
    d_rh = float(np.linalg.norm(s[-1, :2] - s[0, :2]))
    wall_clear = wall_y - float(np.abs(s[-1, 1])) - env.robot_radius
    oc = sim['outcome']
    status = ("RUNNING" if not sim['done'] else
              ("SUCCESS" if bool(oc["success"]) else
               "COLLISION-HUMAN" if bool(oc["collision_with_human"]) else
               "COLLISION-OBSTACLE" if bool(oc["collision_with_obstacle"]) else
               "TIMEOUT" if bool(oc["timeout"]) else "DONE"))
    wlim = W_MAX * (1.0 - cmd['v'] / V_MAX)
    hud = (
        f"[{TRAJECTORY_NAME}]  step={len(all_actions)}  t={float(sim['info']['time']):.2f}s  "
        f"cmd v={cmd['v']:.2f} w={cmd['w']:.2f} (|w|<={wlim:.2f})  robot_vel={np.asarray(s[-1, 2:4])}\n"
        f"dist(robot,human)={d_rh:.3f} m   wall_clearance={wall_clear:.3f} m   status={status}"
    )
    ax.set_title(hud, fontsize=10, family='monospace')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_xlim([-TRAFFIC_LENGTH / 2 - 3, TRAFFIC_LENGTH / 2 + 1])
    ax.set_ylim([-TRAFFIC_HEIGHT, TRAFFIC_HEIGHT])
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.draw_idle()


def do_step():
    if sim['done']:
        print("Episode finished. Use 'z' to undo or 'p' to save.")
        return
    clamp_cmd()
    undo_stack.append((sim['state'], sim['info'], sim['obs'], sim['env_key'],
                       sim['outcome'], sim['done'], len(all_actions)))
    action = jnp.array([cmd['v'], cmd['w']])
    new_state, new_obs, new_info, _, new_outcome, (_, new_env_key) = env.step(
        sim['state'], sim['info'], action, test=True, env_key=sim['env_key'],
    )
    sim['state'], sim['obs'], sim['info'] = new_state, new_obs, new_info
    sim['env_key'], sim['outcome'] = new_env_key, new_outcome
    sim['done'] = not bool(new_outcome["nothing"])
    all_states.append(np.asarray(new_state))
    all_actions.append(np.asarray(action))
    all_obs.append(np.asarray(new_obs))
    redraw()


def do_undo():
    if not undo_stack:
        print("Nothing to undo.")
        return
    st, inf, ob, ek, oc, dn, n_act = undo_stack.pop()
    sim['state'], sim['info'], sim['obs'] = st, inf, ob
    sim['env_key'], sim['outcome'], sim['done'] = ek, oc, dn
    del all_states[n_act + 1:]
    del all_obs[n_act + 1:]
    del all_actions[n_act:]
    redraw()


def proximity_metrics():
    hist = np.asarray(all_states)  # (T+1, 2, 6)
    d = np.linalg.norm(hist[:, 1, :2] - hist[:, 0, :2], axis=1) - HUMAN_RADIUS - env.robot_radius
    dt = env_params['robot_dt']
    intrusion = {thr: float(np.sum(dt * np.maximum(0.0, thr - d))) for thr in (0.5, 1.0, 1.5)}
    return float(d.min()), int(d.argmin()), intrusion


def do_save():
    os.makedirs(OUT_DIR, exist_ok=True)
    hist = np.asarray(all_states)
    robot_yaws = jnp.asarray(hist[:, -1, 4])
    lidar = vmap(lambda o, ry: _lidar_from_obs(o, ry))(jnp.asarray(all_obs), robot_yaws)
    data = {
        'label': TRAJECTORY_NAME,
        'all_states': hist,
        'all_actions': np.asarray(all_actions),
        'all_lidar': np.asarray(lidar),
        'robot_goal': np.asarray(sim['info']['robot_goal']),
        'humans_goal': np.asarray(sim['info']['humans_goal']),
        'humans_parameters': np.asarray(sim['info']['humans_parameters']),
        'static_obstacles': np.asarray(sim['info']['static_obstacles']),
        'current_scenario': -1,
        'robot_dt': env_params['robot_dt'],
        'humans_dt': env_params['humans_dt'],
        'kinematics': env_params['kinematics'],
        'n_humans': env_params['n_humans'],
        'lidar_num_rays': env.lidar_num_rays,
        'lidar_angular_range': float(env.lidar_angular_range),
        'robot_radius': env.robot_radius,
        'outcome': {k: bool(v) for k, v in sim['outcome'].items()},
    }
    path = os.path.join(OUT_DIR, f"{TRAJECTORY_NAME}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    min_d, min_step, intrusion = proximity_metrics()
    print(f"\nSaved trajectory to: {path}")
    print(f"  steps={len(all_actions)}  outcome={data['outcome']}")
    print(f"  min surface-to-surface distance (robot,human) = {min_d:.3f} m at step {min_step}")
    print(f"  intrusion integral  sum(dt*max(0, thr-d))  = {intrusion}\n")


def do_replay():
    hist = np.asarray(all_states)
    robot_yaws = jnp.asarray(hist[:, -1, 4])
    lidar = vmap(lambda o, ry: _lidar_from_obs(o, ry))(jnp.asarray(all_obs), robot_yaws)
    animate_trajectory(
        hist,
        np.asarray(sim['info']['humans_parameters'][:, 0]),
        env.robot_radius,
        'hsfm',
        np.asarray(sim['info']['robot_goal']),
        -1,
        static_obstacles=np.asarray(sim['info']['static_obstacles'][-1]),
        robot_dt=env_params['robot_dt'],
        lidar_measurements=np.asarray(lidar),
        kinematics='unicycle',
    )


def on_key(event):
    k = event.key
    if k in ('up', 'w'):
        cmd['v'] += DV
    elif k in ('down', 's'):
        cmd['v'] -= DV
    elif k in ('left', 'a'):
        cmd['w'] += DW
    elif k in ('right', 'd'):
        cmd['w'] -= DW
    elif k == 'x':
        cmd['v'] = 0.0
    elif k == 'c':
        cmd['w'] = 0.0
    elif k == ' ':
        do_step()
        return
    elif k in ('z', 'backspace'):
        do_undo()
        return
    elif k == 'p':
        do_save()
        return
    elif k == 'v':
        do_replay()
        return
    elif k == 'h':
        print(__doc__)
        return
    else:
        return
    clamp_cmd()
    redraw()


fig.canvas.mpl_connect('key_press_event', on_key)
redraw()
plt.show()
