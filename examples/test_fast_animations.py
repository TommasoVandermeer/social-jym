"""
Check that the new *_fast animate methods of the policies draw the same frames of the old animate methods, and measure their speed.

Usage:
    MPLBACKEND=Agg python examples/test_fast_animations.py                 # Compare old and new frames for all cases
    MPLBACKEND=Agg python examples/test_fast_animations.py --benchmark     # Also measure ms/frame of old and new animations
    python examples/test_fast_animations.py --show dir_safe                # Open the new (fast) animation of one case
    python examples/test_fast_animations.py --show dir_safe --old          # Open the old animation of one case

For each case a short episode is simulated (with real policy outputs where cheap, plausible random ones otherwise), then the old
and the new animate methods are called on the same data. Both are intercepted before showing anything: the frame function of the
old FuncAnimation and the new TrajectoryAnimator are used to render some frames with Agg, which are compared pixel by pixel.
"""
import argparse
import os
import pickle
import sys
import tempfile
import time
import numpy as np
from jax import random, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.patches import Ellipse
from matplotlib.testing.compare import calculate_rms

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1 as SocialReward
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as LaserDummyReward
from socialjym.policies.cadrl import CADRL
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.dwa import DWA
from socialjym.policies.mppi import MPPI
from socialjym.policies.dra_mppi import DRAMPPI
from socialjym.policies.jessi import JESSI
from socialjym.policies.jessi_mppi import JESSI_MPPI
import socialjym.policies.cadrl as cadrl_module
import socialjym.policies.dwa as dwa_module
import socialjym.policies.mppi as mppi_module
import socialjym.policies.jessi as jessi_module
from socialjym.utils.animation import states_to_scene
from socialjym.utils.animation.animator import TrajectoryAnimator
from socialjym.utils.animation.layers import EllipsesLayer

N_STEPS = 12 # Steps simulated for each case
RANDOM_SEED = 0
# Pixel comparison tolerances (RGBA buffers, 0-255)
MAX_RMS = 1.
MAX_DIFFERENT_PIXELS_PERCENTAGE = 0.5
DIFFERENT_PIXEL_THRESHOLD = 32
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

### EPISODES GENERATION

def socialnav_env(**kwargs):
    env_params = {
        'n_humans': 5,
        'n_obstacles': 5,
        'robot_radius': 0.3,
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': True,
        'scenario': 'hybrid_scenario',
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]),
        'ccso_n_static_humans': 0,
        'reward_function': SocialReward(kinematics='unicycle'),
        'kinematics': 'unicycle',
    }
    env_params.update(kwargs)
    return SocialNav(**env_params)

def lasernav_env(**kwargs):
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
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]),
        'ccso_n_static_humans': 0,
        'reward_function': LaserReward(robot_radius=0.3),
        'kinematics': 'unicycle',
    }
    env_params.update(kwargs)
    return LaserNav(**env_params)

def simulate(env, act, n_steps=N_STEPS, seed=RANDOM_SEED):
    """
    Simulate up to n_steps with act(obs, info, step) -> (action, outputs_dict). Returns a dict with the stacked data of the steps
    (states, observations, goals, static obstacles, humans radii, legs if any, actions and policy outputs), aligned as in the examples.
    """
    reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed)
    state, _, obs, info, outcome = env.reset(reset_key)
    data = {k: [] for k in ['states', 'observations', 'goals', 'static_obstacles', 'humans_radii', 'actions', 'legs_states', 'legs_radii']}
    outputs = []
    for step in range(n_steps):
        action, out = act(obs, info, step)
        data['states'].append(state)
        data['observations'].append(obs)
        data['goals'].append(info['robot_goal'])
        data['static_obstacles'].append(info['static_obstacles'][-1])
        data['humans_radii'].append(info['humans_parameters'][:,0])
        if 'humans_leg_state' in info:
            data['legs_states'].append(info['humans_leg_state'])
            data['legs_radii'].append(info['humans_leg_parameters'][:,-1])
        data['actions'].append(action)
        outputs.append(out)
        if isinstance(env, LaserNav):
            state, obs, info, _, outcome, (_, env_key) = env.step(state, info, action, test=True, env_key=env_key)
        else:
            state, obs, info, _, outcome, _ = env.step(state, info, action, test=True)
        if not outcome["nothing"]:
            break
    data = {k: jnp.stack(v) if len(v) > 0 else None for k, v in data.items()}
    if len(outputs[0]) > 0:
        data.update(tree_map(lambda *x: jnp.stack(x), *outputs))
    return data

def towards_goal_action(obs_robot_pose, goal, v_max=1., wheels_distance=0.7):
    """Simple unicycle action towards the goal (used to move the robot when policy outputs are random)."""
    diff = goal - obs_robot_pose[:2]
    heading_error = jnp.arctan2(jnp.sin(jnp.arctan2(diff[1], diff[0]) - obs_robot_pose[2]), jnp.cos(jnp.arctan2(diff[1], diff[0]) - obs_robot_pose[2]))
    w_max = 2 * v_max / wheels_distance
    w = jnp.clip(heading_error / 0.25, -w_max, w_max)
    v = v_max * (1 - jnp.abs(w) / w_max)
    return jnp.array([v, w])

def random_hcgs(key, n_steps, n_humans, n_detectable, horizon=None):
    """Random (but plausible) bivariate Gaussians and weights, with some NaN means if horizon is given (like DRA-MPPI)."""
    lead = (n_steps,) if horizon is None else (n_steps, horizon)
    k1, k2, k3, k4 = random.split(key, 4)
    distrs = {
        'means': random.uniform(k1, lead + (n_detectable, 2), minval=-6, maxval=6),
        'logsigmas': random.uniform(k2, lead + (n_detectable, 2), minval=-2, maxval=0),
        'correlation': random.uniform(k3, lead + (n_detectable,), minval=-0.7, maxval=0.7),
    }
    weights = random.uniform(k4, lead + (n_detectable,))
    return distrs, weights

### CASES
# Each case returns (policy, old_method_name, args, kwargs, module_of_old_FuncAnimation). The new method is old_method_name + "_fast".

def case_cadrl():
    env = socialnav_env()
    policy = CADRL(SocialReward(kinematics='unicycle'), kinematics='unicycle')
    policy.lidar_angles_robot_frame = jnp.linspace(-jnp.pi, jnp.pi, 60) # CADRL has no LiDAR: needed by the old method to draw lidar_scans
    data = simulate(env, lambda obs, info, step: (towards_goal_action(obs[-1,jnp.array([0,1,5])], info['robot_goal']), {}))
    n = len(data['actions'])
    key = random.PRNGKey(1)
    pos_distrs, weights = random_hcgs(key, n, env.n_humans, 6)
    vel_distrs, _ = random_hcgs(random.split(key)[0], n, env.n_humans, 6)
    gx, gy = jnp.meshgrid(jnp.linspace(-9, 9, 10), jnp.linspace(-9, 9, 10), indexing='ij')
    # animate_socialnav_trajectory does not expose lidar_scans, so this case calls animate_trajectory directly
    robot_poses, humans_poses, humans_velocities = states_to_scene(data['states'], env.humans_policy)
    args = (env.robot_radius, robot_poses, data['actions'], data['goals'], humans_poses, humans_velocities, data['humans_radii'])
    kwargs = dict(
        robot_actions_values=random.uniform(key, (n, len(policy.action_space))),
        perception_distrs={'pos_distrs': pos_distrs, 'vel_distrs': vel_distrs, 'weights': weights},
        lidar_scans=random.uniform(key, (n, 60), minval=1, maxval=8),
        static_obstacles=data['static_obstacles'],
        occupancy_grids=random.bernoulli(key, 0.3, (n, 10, 10)),
        grid_cells=jnp.stack((gx, gy), axis=-1),
        grid_cells_size=2.,
    )
    return policy, 'animate_trajectory', args, kwargs, cadrl_module

def case_dir_safe():
    env = socialnav_env()
    policy = DIRSAFE(reward_function=env.reward_function)
    state, _, obs, info, _ = env.reset(random.PRNGKey(RANDOM_SEED))
    actor_params, _ = policy.init_nns(random.PRNGKey(0), obs, info)
    def act(obs, info, step):
        action, _, _, _, action_distr = policy.act(random.PRNGKey(step), obs, info, actor_params, sample=False)
        return action, {'action_distrs': action_distr}
    data = simulate(env, act)
    args = (data['states'], data['actions'], data['goals'], data['humans_radii'], env)
    kwargs = dict(static_obstacles=data['static_obstacles'], action_distrs=data['action_distrs'])
    return policy, 'animate_socialnav_trajectory', args, kwargs, cadrl_module

def _case_dwa(use_box_action_space):
    env = lasernav_env(reward_function=LaserDummyReward(robot_radius=0.3))
    policy = DWA(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
        actions_discretization=9 if use_box_action_space else 16,
        use_box_action_space=use_box_action_space,
    )
    def act(obs, info, step):
        action, actions_cost = policy.act(obs, info)
        return action, {'actions_costs': actions_cost}
    data = simulate(env, act)
    args = (data['states'], data['observations'], data['actions'], data['actions_costs'], data['goals'], data['static_obstacles'], data['humans_radii'], env)
    return policy, 'animate_lasernav_trajectory', args, {}, dwa_module

def case_dwa():
    return _case_dwa(False)

def case_dwa_box():
    return _case_dwa(True)

def case_mppi():
    env = lasernav_env(reward_function=LaserDummyReward(robot_radius=0.3))
    policy = MPPI(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
        num_samples=300,
    )
    carry = {'u_mean': policy.init_u_mean(), 'key': random.PRNGKey(0)}
    def act(obs, info, step):
        action, u_mean, trajectories, costs, carry['key'] = policy.act(obs, info, carry['u_mean'], carry['key'])
        carry['u_mean'] = u_mean
        return action, {'u_means': u_mean, 'trajectories': trajectories, 'trajectories_costs': costs}
    data = simulate(env, act)
    args = (data['states'], data['observations'], data['actions'], data['u_means'], data['trajectories'], data['trajectories_costs'], data['goals'], data['static_obstacles'], data['humans_radii'], env)
    return policy, 'animate_lasernav_trajectory', args, {}, mppi_module

def _dra_mppi_random_outputs(policy, data, key, n_humans):
    """Random MPPI outputs (sampled trajectories with some costs above the threshold, u_means) and humans predicted distributions with NaNs."""
    n = len(data['actions'])
    k1, k2, k3, k4 = random.split(key, 4)
    robot_poses = jnp.column_stack((data['states'][:,-1,:2], data['states'][:,-1,4]))
    controls = jnp.clip(random.normal(k1, (n, 50, policy.horizon, 2)) * jnp.array([0.3, 1.]) + jnp.array([0.6, 0.]), jnp.array([0., -2.8]), jnp.array([1., 2.8]))
    trajectories = vmap(lambda pose, us: vmap(lambda u: policy._rollout_u_means(pose[None], u[None])[0])(us))(robot_poses, controls)
    costs = jnp.where(random.uniform(k2, (n, 50)) < 0.3, 2_000_000., random.uniform(k2, (n, 50)))
    u_means = controls[:,0]
    humans_distributions, _ = random_hcgs(k3, n, n_humans, n_humans, horizon=policy.horizon)
    nan_humans = random.uniform(k4, (n, 1, n_humans)) < 0.3
    humans_distributions['means'] = jnp.where(nan_humans[...,None], jnp.nan, humans_distributions['means'])
    return u_means, trajectories, costs, humans_distributions

def case_dra_mppi_socialnav():
    env = socialnav_env()
    policy = DRAMPPI()
    data = simulate(env, lambda obs, info, step: (towards_goal_action(obs[-1,jnp.array([0,1,5])], info['robot_goal']), {}))
    u_means, trajectories, costs, humans_distributions = _dra_mppi_random_outputs(policy, data, random.PRNGKey(2), env.n_humans)
    args = (data['states'], data['actions'], u_means, trajectories, costs, data['goals'], data['static_obstacles'], data['humans_radii'], humans_distributions, env)
    return policy, 'animate_socialnav_trajectory', args, {}, mppi_module

def case_dra_mppi_lasernav():
    env = lasernav_env()
    policy = DRAMPPI(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
    )
    data = simulate(env, lambda obs, info, step: (towards_goal_action(obs[0,:3], info['robot_goal']), {}))
    u_means, trajectories, costs, humans_distributions = _dra_mppi_random_outputs(policy, data, random.PRNGKey(3), 10)
    n = len(data['actions'])
    pos_distrs, weights = random_hcgs(random.PRNGKey(4), n, env.n_humans, 10)
    vel_distrs, _ = random_hcgs(random.PRNGKey(5), n, env.n_humans, 10)
    perception_distributions = {'pos_distrs': pos_distrs, 'vel_distrs': vel_distrs, 'weights': weights}
    args = (data['states'], data['observations'], data['actions'], u_means, trajectories, costs, data['goals'], data['static_obstacles'], data['humans_radii'], humans_distributions, perception_distributions, env)
    return policy, 'animate_lasernav_trajectory', args, {}, mppi_module

def _jessi_network_params(policy):
    path = os.path.join(EXAMPLES_DIR, 'jessi_multitask_rl_out_32.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            network_params, _, _ = pickle.load(f)
        return network_params
    return policy.init_nns(random.PRNGKey(0))[2]

def _jessi_policy(policy_class, env, **kwargs):
    return policy_class(
        lidar_num_rays=env.lidar_num_rays,
        lidar_angular_range=env.lidar_angular_range,
        lidar_max_dist=env.lidar_max_dist,
        n_stack=env.n_stack,
        embedding_dim=32,
        **kwargs,
    )

def case_jessi(real_robot=False):
    env = lasernav_env(lidar_num_rays=200, leg_dynamics=not real_robot, scenario='circular_crossing_with_static_obstacles', ccso_n_static_humans=3, lidar_noise=True)
    policy = _jessi_policy(JESSI, env)
    network_params = _jessi_network_params(policy)
    def act(obs, info, step):
        action, _, _, _, _, _, perception_distr, actor_distr, _, spat_attn, temp_attn, human_attn = policy.act(random.PRNGKey(0), obs, info, network_params, sample=False)
        return action, {'humans_distrs': perception_distr, 'actor_distrs': actor_distr, 'spatial_attentions': spat_attn[0], 'temporal_attentions': temp_attn[0], 'human_attentions': human_attn[0]}
    data = simulate(env, act)
    kwargs = dict(
        observations=data['observations'],
        actions=data['actions'],
        actor_distrs=data['actor_distrs'],
        humans_distrs=data['humans_distrs'],
        goals=data['goals'],
        spatial_attentions=data['spatial_attentions'],
        temporal_attentions=data['temporal_attentions'],
        human_attentions=data['human_attentions'],
    )
    if not real_robot:
        kwargs.update(
            states=data['states'],
            humans_leg_states=data['legs_states'],
            static_obstacles=data['static_obstacles'],
            humans_radii=data['humans_radii'],
            humans_leg_radii=data['legs_radii'],
        )
    return policy, 'animate_lasernav_trajectory', (env,), kwargs, jessi_module

def case_jessi_real_robot():
    return case_jessi(real_robot=True)

def case_jessi_mppi():
    env = lasernav_env(lidar_num_rays=200)
    policy = _jessi_policy(JESSI_MPPI, env, num_samples=50)
    perception_params, actor_critic_params = policy.split_nns_params(_jessi_network_params(policy))
    def act(obs, info, step):
        action, u_mean, trajectories, costs, perception_distr, humans_distr_seq, actor_distr, _, spatial_attn, temporal_attn, human_attn, _ = policy.act(
            random.PRNGKey(step), obs, info, perception_params, actor_critic_params,
        )
        return action, {
            'u_means': u_mean, 'trajectories': trajectories, 'trajectories_costs': costs, 'humans_distrs': perception_distr,
            'future_humans_distrs': humans_distr_seq, 'actor_distrs': actor_distr, 'spatial_attentions': spatial_attn[0],
            'temporal_attentions': temporal_attn[0], 'human_attentions': human_attn[0],
        }
    data = simulate(env, act)
    kwargs = dict(
        states=data['states'],
        observations=data['observations'],
        actions=data['actions'],
        actor_distrs=data['actor_distrs'],
        humans_distrs=data['humans_distrs'],
        goals=data['goals'],
        static_obstacles=data['static_obstacles'],
        humans_radii=data['humans_radii'],
        spatial_attentions=data['spatial_attentions'],
        temporal_attentions=data['temporal_attentions'],
        human_attentions=data['human_attentions'],
        trajectories=data['trajectories'],
        trajectories_costs=data['trajectories_costs'],
        control_sequences=data['u_means'],
        future_humans_distrs=data['future_humans_distrs'],
    )
    return policy, 'animate_lasernav_trajectory', (env,), kwargs, jessi_module

CASES = {
    'cadrl': case_cadrl,
    'dir_safe': case_dir_safe,
    'dwa': case_dwa,
    'dwa_box': case_dwa_box,
    'mppi': case_mppi,
    'dra_mppi_socialnav': case_dra_mppi_socialnav,
    'dra_mppi_lasernav': case_dra_mppi_lasernav,
    'jessi': case_jessi,
    'jessi_real_robot': case_jessi_real_robot,
    'jessi_mppi': case_jessi_mppi,
}

### OLD AND NEW ANIMATIONS INTERCEPTION

class _CapturedFuncAnimation:
    """Replaces FuncAnimation in the module of the old animate method: keeps the figure and the frame function, shows nothing."""
    captured = None
    def __init__(self, fig, func, **kwargs):
        _CapturedFuncAnimation.captured = (fig, func, kwargs.get('frames'))
    def save(self, *args, **kwargs):
        pass

def capture_old_animation(policy, method_name, args, kwargs, module):
    original_func_animation, original_show = module.FuncAnimation, plt.show
    module.FuncAnimation, plt.show = _CapturedFuncAnimation, (lambda *a, **k: None)
    try:
        getattr(policy, method_name)(*args, **kwargs)
    finally:
        module.FuncAnimation, plt.show = original_func_animation, original_show
    return _CapturedFuncAnimation.captured

def capture_new_animation(policy, method_name, args, kwargs):
    captured = {}
    original_run = TrajectoryAnimator.run
    TrajectoryAnimator.run = lambda self, *a, **k: captured.setdefault('animator', self)
    try:
        getattr(policy, method_name + '_fast')(*args, **kwargs)
    finally:
        TrajectoryAnimator.run = original_run
    return captured['animator']

def render_old_frame(fig, func, frame):
    func(frame)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()

### DATA COMPARISON (on the artists where the per-frame logic was vectorized: action space scatters, trajectories, ellipses)

def _shown_scatters(ax):
    return sorted([c for c in ax.collections if isinstance(c, PathCollection) and c.get_visible() and len(c.get_offsets()) > 0], key=lambda c: c.get_zorder())

def compare_frame_data(old_fig, animator, frame):
    """Compare the data of the old artists (just drawn at frame) with the new ones (just updated to frame). Returns a list of errors."""
    errors = []
    old_axs = old_fig.axes
    new_axs = animator.sim_axs + [animator.action_ax]
    # Action space scatters (sampled actions, actions values/costs, action distribution): positions, colormapped values and color limits
    old_scatters, new_scatters = _shown_scatters(old_axs[-1]), _shown_scatters(new_axs[-1])
    if len(old_scatters) != len(new_scatters):
        errors.append(f"action space: {len(old_scatters)} scatters vs {len(new_scatters)}")
    for old_sc, new_sc in zip(old_scatters, new_scatters):
        if not np.allclose(old_sc.get_offsets(), new_sc.get_offsets(), atol=1e-5):
            errors.append(f"action space: different scatter points (zorder {old_sc.get_zorder()})")
        elif old_sc.get_array() is not None and not (np.allclose(old_sc.get_array(), new_sc.get_array(), rtol=1e-4) and np.allclose(old_sc.get_clim(), new_sc.get_clim(), rtol=1e-4)):
            errors.append(f"action space: different scatter values or color limits (zorder {old_sc.get_zorder()})")
    for i, (old_ax, new_ax) in enumerate(zip(old_axs[:-1], new_axs[:-1])):
        # Sampled trajectories (zorder 10): old Line2Ds vs new LineCollection
        old_lines = [l for l in old_ax.lines if l.get_zorder() == 10]
        new_collections = [c for c in new_ax.collections if isinstance(c, LineCollection) and c.get_zorder() == 10]
        new_segments = [seg for c in new_collections for seg in c.get_segments()]
        if len(old_lines) != len(new_segments):
            errors.append(f"axes {i}: {len(old_lines)} trajectories vs {len(new_segments)}")
        elif len(old_lines) > 0:
            new_colors = np.concatenate([np.broadcast_to(c.get_colors(), (len(c.get_segments()), 4)) for c in new_collections])
            new_widths = np.concatenate([np.broadcast_to(c.get_linewidths(), (len(c.get_segments()),)) for c in new_collections])
            for j, line in enumerate(old_lines):
                if not np.allclose(line.get_xydata(), new_segments[j], atol=1e-4):
                    errors.append(f"axes {i}: different trajectory {j}")
                    break
                if not (np.allclose(mcolors.to_rgba(line.get_color(), line.get_alpha()), new_colors[j]) and np.isclose(line.get_linewidth(), new_widths[j])):
                    errors.append(f"axes {i}: different style of trajectory {j}")
                    break
        # Gaussian ellipses: old Ellipse patches vs new EllipseCollections (angles modulo 180 degrees)
        old_ellipses = np.array([[*e.get_center(), e.get_width(), e.get_height(), e.get_angle()] for e in old_ax.patches if type(e) is Ellipse]).reshape(-1, 5)
        new_ellipses = []
        for layer in animator.layers:
            if isinstance(layer, EllipsesLayer) and layer.ax is new_ax:
                shown = np.ones(layer.widths.shape[1], bool) if layer.mask is None else layer.mask[frame]
                new_ellipses.append(np.column_stack((layer.centers[frame][shown], layer.widths[frame][shown], layer.heights[frame][shown], layer.angles[frame][shown])))
        new_ellipses = np.concatenate(new_ellipses) if new_ellipses else np.zeros((0, 5))
        if len(old_ellipses) != len(new_ellipses):
            errors.append(f"axes {i}: {len(old_ellipses)} ellipses vs {len(new_ellipses)}")
        elif len(old_ellipses) > 0:
            angles_error = np.abs((old_ellipses[:,4] - new_ellipses[:,4] + 90) % 180 - 90)
            almost_circles = np.isclose(old_ellipses[:,2], old_ellipses[:,3], rtol=1e-3) # Their angle is not defined
            if not np.allclose(old_ellipses[:,:4], new_ellipses[:,:4], atol=1e-4) or np.any(angles_error[~almost_circles] > 1e-2):
                errors.append(f"axes {i}: different ellipses")
    return errors

### MAIN

def compare_case(name, benchmark, out_dir):
    policy, method_name, args, kwargs, module = CASES[name]()
    old_fig, old_func, n_frames = capture_old_animation(policy, method_name, args, kwargs, module)
    animator = capture_new_animation(policy, method_name, args, kwargs)
    assert n_frames == animator.n_frames, f"Different number of frames: old {n_frames}, new {animator.n_frames}"
    passed = True
    for frame in range(n_frames):
        old_func(frame)
        animator.update(frame)
        errors = compare_frame_data(old_fig, animator, frame)
        if errors:
            print(f"  frame {frame:3d}: FAIL data " + "; ".join(errors))
            passed = False
    print(f"  data of all {n_frames} frames: {'OK' if passed else 'FAIL'}")
    for frame in sorted({0, n_frames // 2, n_frames - 1}):
        old_image = render_old_frame(old_fig, old_func, frame)
        new_image = animator.render_frame(frame)
        if old_image.shape != new_image.shape:
            print(f"  frame {frame:3d}: FAIL different image shapes {old_image.shape} vs {new_image.shape}")
            passed = False
            continue
        rms = calculate_rms(old_image[...,:3].astype(np.int16), new_image[...,:3].astype(np.int16)) # int16: calculate_rms subtracts before converting to float
        difference = np.abs(old_image.astype(int) - new_image.astype(int)).max(axis=-1)
        different_pixels = 100 * np.mean(difference > DIFFERENT_PIXEL_THRESHOLD)
        frame_passed = (rms <= MAX_RMS) and (different_pixels <= MAX_DIFFERENT_PIXELS_PERCENTAGE)
        passed &= frame_passed
        print(f"  frame {frame:3d}: RMS {rms:6.3f} - different pixels {different_pixels:6.3f}% {'OK' if frame_passed else 'FAIL'}")
        if not frame_passed:
            prefix = os.path.join(out_dir, f"{name}_frame{frame}")
            plt.imsave(prefix + "_old.png", old_image)
            plt.imsave(prefix + "_new.png", new_image)
            plt.imsave(prefix + "_diff.png", difference, cmap='gray')
            print(f"    saved {prefix}_{{old,new,diff}}.png")
    if benchmark:
        def _time(fn):
            fn(0) # Warm up
            times = []
            for frame in range(1, n_frames):
                start = time.perf_counter()
                fn(frame)
                times.append(time.perf_counter() - start)
            return 1000 * np.mean(times)
        old_ms = _time(lambda f: render_old_frame(old_fig, old_func, f))
        new_ms = _time(animator.render_frame)
        blit_ms = _time(lambda f: blit_frame(animator, f))
        print(f"  old: {old_ms:7.1f} ms/frame - new (full draw): {new_ms:6.1f} ms/frame - new (blit): {blit_ms:6.1f} ms/frame - speedup {old_ms/blit_ms:5.1f}x (real time needs {1000*policy.dt:.0f} ms/frame)")
    plt.close(old_fig)
    plt.close(animator.fig)
    return passed

def blit_frame(animator, frame):
    """What the interactive playback does at each frame with blitting: restore the background and redraw only the data artists."""
    canvas = animator.fig.canvas
    if not hasattr(animator, '_benchmark_backgrounds'):
        artists = animator.update(frame)
        for artist in artists:
            artist.set_animated(True)
        canvas.draw()
        animator._benchmark_backgrounds = {ax: canvas.copy_from_bbox(ax.bbox) for ax in animator.fig.axes}
    for ax, background in animator._benchmark_backgrounds.items():
        canvas.restore_region(background)
    for artist in sorted(animator.update(frame), key=lambda a: a.get_zorder()):
        artist.axes.draw_artist(artist)
    for ax in animator._benchmark_backgrounds:
        canvas.blit(ax.bbox)

def show_case(name, old):
    policy, method_name, args, kwargs, _ = CASES[name]()
    getattr(policy, method_name if old else method_name + '_fast')(*args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cases', nargs='*', default=list(CASES.keys()), help=f"cases to run (default: all): {', '.join(CASES.keys())}")
    parser.add_argument('--benchmark', action='store_true', help="also measure the time per frame of old and new animations")
    parser.add_argument('--show', metavar='CASE', choices=CASES.keys(), help="open the animation of one case instead of comparing")
    parser.add_argument('--old', action='store_true', help="with --show, open the old animation")
    cli_args = parser.parse_args()
    if cli_args.show is not None:
        show_case(cli_args.show, cli_args.old)
        sys.exit(0)
    assert matplotlib.get_backend().lower() == 'agg', "Run the comparison with MPLBACKEND=Agg"
    out_dir = tempfile.mkdtemp(prefix='fast_animations_')
    results = {}
    for name in cli_args.cases:
        print(f"\n### {name}")
        results[name] = compare_case(name, cli_args.benchmark, out_dir)
    print("\n### SUMMARY")
    for name, passed in results.items():
        print(f"  {name:20s} {'OK' if passed else 'FAIL'}")
    if not all(results.values()):
        print(f"Images of the failed frames in {out_dir}")
        sys.exit(1)
    os.rmdir(out_dir)
