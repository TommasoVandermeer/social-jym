import jax.numpy as jnp
from jax import lax, jit, random, vmap, debug
from jax_tqdm import loop_tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pickle as pkl
import os
from datetime import date

from socialjym.envs.base_env import BaseEnv, SCENARIOS, HUMAN_POLICIES
from socialjym.policies.base_policy import BasePolicy

@jit
def epsilon_scaling_decay(epsilon_start:float, epsilon_end:float, current_episode:int, decay_rate:float) -> float:
    epsilon = lax.cond(current_episode < decay_rate, lambda x: epsilon_start + (epsilon_end - epsilon_start) / decay_rate * x, lambda x: epsilon_end, current_episode)
    return epsilon

@jit
def is_multiple(number:float, dividend:float, tolerance:float=1e-7) -> bool:
    """
    Checks if a number (also a float) is a multiple of another number within a given tolerance error.
    """
    mod = number % dividend
    return jnp.any(jnp.array([abs(mod) <= tolerance,abs(dividend - mod) <= tolerance]))

@jit
def point_to_line_distance(point:jnp.ndarray, line_start:jnp.ndarray, line_end:jnp.ndarray) -> float:
    """
    Computes the distance between a point and a line defined by two points.

    args:
    - point: jnp.ndarray, shape=(2,), dtype=jnp.float32. The point to compute the distance from.
    - line_start: jnp.ndarray, shape=(2,), dtype=jnp.float32. The starting point of the line.
    - line_end: jnp.ndarray, shape=(2,), dtype=jnp.float32. The ending point of the line.

    output:
    - distance: float. The distance between the point and the line.
    """
    x = point[0]
    y = point[1]
    x1 = line_start[0]
    y1 = line_start[1]
    x2 = line_end[0]
    y2 = line_end[1]
    dx = x2 - x1
    dy = y2 - y1

    u = lax.cond(
        jnp.all(jnp.array([dx == 0, dy == 0])), 
        lambda _: 0., 
        lambda _: jnp.squeeze(((x - x1) * dx + (y - y1) * dy) / jnp.linalg.norm(line_end - line_start)**2), 
        None)

    # Clamp u to [0,1]
    u = lax.cond(u < 0, lambda x: 0., lambda x: x, u)
    u = lax.cond(u > 1, lambda x: 1., lambda x: x, u)

    closest_point = jnp.array([x1 + u * dx, y1 + u * dy])
    closest_distance = jnp.linalg.norm(closest_point - point)

    return closest_distance

@jit
def batch_point_to_line_distance(points:jnp.ndarray, line_starts:jnp.ndarray, line_ends:jnp.ndarray) -> jnp.ndarray:
    return vmap(point_to_line_distance, in_axes=(0, 0, 0))(points, line_starts, line_ends)

def plot_state(
        ax:Axes, 
        time:float, 
        full_state:jnp.ndarray, 
        humans_policy:str, 
        scenario:int, 
        humans_radiuses:np.ndarray, 
        robot_radius:float, 
        circle_radius=7, 
        traffic_height=3, 
        traffic_length=14, 
        crowding_square_side=14,
        plot_time=True,
        kinematics:str='holonomic') -> None:
    """
    Plots a given single state of the environment.

    args:
    - ax: matplotlib.axes.Axes. The axes to plot the state.
    - time: float. The time of the state to be plotted.
    - full_state: shape=(n_humans+1, 6), dtype=jnp.float32
        The state of the environment to be plotted. The last row of full_state corresponds to the robot state.
    - humans_policy: str, one of ['orca', 'sfm', 'hsfm']
    - humans_radiuses: np.ndarray, shape=(n_humans,), dtype=np.float32
    - circle_radius: float
    - robot_radius: float

    output:
    - None
    """
    colors = list(mcolors.TABLEAU_COLORS.values())
    num = int(time) if (time).is_integer() else (time)
    if scenario == SCENARIOS.index('circular_crossing'): ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
    elif scenario == SCENARIOS.index('parallel_traffic'): ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_height-3,traffic_height+3])
    elif scenario == SCENARIOS.index('perpendicular_traffic'): ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_length/2,traffic_length/2])
    elif scenario == SCENARIOS.index('robot_crowding'): ax.set(xlabel='X',ylabel='Y',xlim=[-crowding_square_side/2-1.5,crowding_square_side/2+1.5],ylim=[-crowding_square_side/2-1.5,crowding_square_side/2+1.5])
    # Humans
    for h in range(len(full_state)-1): 
        if humans_policy == 'hsfm': 
            head = plt.Circle((full_state[h,0] + np.cos(full_state[h,4]) * humans_radiuses[h], full_state[h,1] + np.sin(full_state[h,4]) * humans_radiuses[h]), 0.1, color=colors[h%len(colors)], zorder=1)
            ax.add_patch(head)
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor=colors[h%len(colors)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        if plot_time: ax.text(full_state[h,0],full_state[h,1], f"{num}", color=colors[h%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
        else: ax.text(full_state[h,0],full_state[h,1], f"{h}", color=colors[h%len(colors)], va="center", ha="center", size=10, zorder=1, weight='bold')
    # Robot
    if kinematics == 'unicycle':
        head = plt.Circle((full_state[-1,0] + np.cos(full_state[-1,4]) * robot_radius, full_state[-1,1] + np.sin(full_state[-1,4]) * robot_radius), 0.1, color='black', zorder=1)
        ax.add_patch(head)
    circle = plt.Circle((full_state[-1,0],full_state[-1,1]), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
    ax.add_patch(circle)
    # Time/Label
    if plot_time: 
        ax.text(full_state[-1,0],full_state[-1,1], f"{num}", color="black", va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=3, weight='bold')
    else: 
        ax.text(full_state[-1,0],full_state[-1,1], f"R", color="black", va="center", ha="center", size=10, zorder=3, weight='bold')

def plot_trajectory(ax:Axes, agents_positions:jnp.ndarray, humans_goal:jnp.ndarray, robot_goal:jnp.ndarray):
    colors = list(mcolors.TABLEAU_COLORS.values())
    n_agents = len(agents_positions[0])
    for h in range(n_agents): 
        ax.plot(agents_positions[:,h,0], agents_positions[:,h,1], color=colors[h%len(colors)] if h < n_agents - 1 else "red", linewidth=1, zorder=0)
        if h < n_agents - 1: ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=colors[h%len(colors)], zorder=2)
        else: ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)

def plot_lidar_measurements(ax:Axes, lidar_measurements:jnp.ndarray, robot_state:jnp.ndarray, robot_radius:float):
    for i in range(len(lidar_measurements)):
        ax.plot(
            [robot_state[0], robot_state[0] + lidar_measurements[i,0] * jnp.cos(lidar_measurements[i,1])], 
            [robot_state[1], robot_state[1] + lidar_measurements[i,0] * jnp.sin(lidar_measurements[i,1])], 
            color="black", 
            linewidth=0.5, 
            zorder=0)

def test_k_trials(
    k: int, 
    random_seed: int, 
    env: BaseEnv, 
    policy: BasePolicy, 
    model_params: dict, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    personal_space:float=0.5,
    custom_episodes:dict=None) -> tuple:
    """
    This function tests a policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - policy: BasePolicy. The policy to test.
    - model_params: dict. The parameters of the policy model.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """

    # TODO: Impelement unicycle kinematics case

    # Since jax does not allow to loop over a dict, we have to decompose it in singular jax numpy arrays
    if custom_episodes is not None:
        assert len(custom_episodes) == k, "The number of custom episodes must be equal to the number of trials."
        custom_trials = True
        custom_states = jnp.array([custom_episodes[i]["full_state"] for i in range(k)])
        custom_robot_goals = jnp.array([custom_episodes[i]["robot_goal"] for i in range(k)])
        custom_humans_goals = jnp.array([custom_episodes[i]["humans_goal"] for i in range(k)])
        custom_static_obstacles = jnp.array([custom_episodes[i]["static_obstacles"] for i in range(k)])
        custom_scenario = jnp.array([custom_episodes[i]["scenario"] for i in range(k)])
        custom_humans_radius = jnp.array([custom_episodes[i]["humans_radius"] for i in range(k)])
        custom_humans_speed = jnp.array([custom_episodes[i]["humans_speed"] for i in range(k)])
    else:
        custom_trials = False
        # Dummy variables
        custom_states = jnp.empty((k, env.n_humans+1, 6))
        custom_robot_goals = jnp.empty((k, 2))
        custom_humans_goals = jnp.empty((k, env.n_humans, 2))
        custom_static_obstacles = jnp.empty((k, 1, 1, 2, 2))
        custom_scenario = jnp.empty((k,), dtype=int)
        custom_humans_radius = jnp.empty((k, env.n_humans))
        custom_humans_speed = jnp.empty((k, env.n_humans))

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):
         
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards = while_val
            # Make a step in the environment
            action, policy_key, _ = policy.act(policy_key, obs, info, model_params, 0.)
            state, obs, info, reward, outcome = env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            all_rewards = all_rewards.at[steps].set(reward)
            # Update step counter
            steps += 1
            return state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards

        ## Retrieve data from the tuple
        seed, metrics = for_val
        policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
        ## Reset the environment
        state, reset_key, obs, info = lax.cond(
            custom_trials, 
            lambda x: env.reset_custom_episode(
                x,
                {"full_state": custom_states[i], 
                 "robot_goal": custom_robot_goals[i], 
                 "humans_goal": custom_humans_goals[i], 
                 "static_obstacles": custom_static_obstacles[i], 
                 "scenario": custom_scenario[i], 
                 "humans_radius": custom_humans_radius[i], 
                 "humans_speed": custom_humans_speed[i]}),
            lambda x: env.reset(x), 
            reset_key)
        # state, reset_key, obs, info = env.reset(reset_key)
        initial_robot_position = state[-1,:2]
        robot_goal = info["robot_goal"]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        all_rewards = jnp.empty((int(time_limit/env.robot_dt)+1,))
        init_outcome = {"nothing": True, "success": False, "failure": False, "timeout": False}
        while_val_init = (state, obs, info, init_outcome, policy_key, 0, all_actions, all_states, all_rewards)
        _, _, _, outcome, policy_key, episode_steps, all_actions, all_states, all_rewards = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
        ## Update metrics
        metrics["successes"] = lax.cond(outcome["success"], lambda x: x + 1, lambda x: x, metrics["successes"])
        metrics["collisions"] = lax.cond(outcome["failure"], lambda x: x + 1, lambda x: x, metrics["collisions"])
        metrics["timeouts"] = lax.cond(outcome["timeout"], lambda x: x + 1, lambda x: x, metrics["timeouts"])
        @jit
        def _compute_state_value_for_body(j:int, t:int, value:float):
            value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * all_rewards[j]
            return value 
        metrics["returns"] = metrics["returns"].at[i].set(lax.fori_loop(0, episode_steps, lambda rr, val: _compute_state_value_for_body(rr, 0, val), 0.))
        path_length = lax.fori_loop(0, episode_steps-1, lambda p, val: val + jnp.linalg.norm(all_states[p+1, -1, :2] - all_states[p, -1, :2]), 0.)
        metrics["episodic_spl"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.linalg.norm(robot_goal-initial_robot_position)/path_length), lambda x: x.at[i].set(0.), metrics["episodic_spl"])
        # Metrics computed only if the episode is successful
        metrics["path_length"] = lax.cond(outcome["success"], lambda x: x.at[i].set(path_length), lambda x: x.at[i].set(jnp.nan), metrics["path_length"])
        metrics["times_to_goal"] = lax.cond(outcome["success"], lambda x: x.at[i].set((episode_steps-1) * env.robot_dt), lambda x: x.at[i].set(jnp.nan), metrics["times_to_goal"])
        speeds = lax.fori_loop(
            0, 
            int(time_limit/env.robot_dt)+1, 
            lambda s, x: lax.cond(
                s < episode_steps,
                lambda y: y.at[s].set(all_actions[s]),
                lambda y: y.at[s].set(jnp.array([jnp.nan,jnp.nan])), 
                x),
            jnp.empty((int(time_limit/env.robot_dt)+1, 2)))
        metrics["average_speed"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.nanmean(jnp.linalg.norm(speeds, axis=1))), lambda x: x.at[i].set(jnp.nan), metrics["average_speed"])
        accelerations = lax.fori_loop(
            0,
            int(time_limit/env.robot_dt)+1,
            lambda a, x: lax.cond(
                a < episode_steps-1,
                lambda y: y.at[a].set((speeds[a+1] - speeds[a]) / env.robot_dt),
                lambda y: y.at[a].set(jnp.array([jnp.nan,jnp.nan])),
                x),
            jnp.empty((int(time_limit/env.robot_dt)+1, 2)))
        metrics["average_acceleration"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.nanmean(jnp.linalg.norm(accelerations, axis=1))), lambda x: x.at[i].set(jnp.nan), metrics["average_acceleration"])
        jerks = lax.fori_loop(
            0,
            int(time_limit/env.robot_dt)+1,
            lambda j, x: lax.cond(
                j < episode_steps-2,
                lambda y: y.at[j].set((accelerations[j+1] - accelerations[j]) / env.robot_dt),
                lambda y: y.at[j].set(jnp.array([jnp.nan,jnp.nan])),
                x),
            jnp.empty((int(time_limit/env.robot_dt)+1, 2)))
        metrics["average_jerk"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.nanmean(jnp.linalg.norm(jerks, axis=1))), lambda x: x.at[i].set(jnp.nan), metrics["average_jerk"])
        min_distances = lax.fori_loop(
            0,
            int(time_limit/env.robot_dt)+1,
            lambda m, x: lax.cond(
                m < episode_steps,
                lambda y: y.at[m].set(jnp.min(jnp.linalg.norm(all_states[m, :-1, :2] - all_states[m, -1, :2], axis=1) - info["humans_parameters"][:,0]) - env.robot_radius),
                lambda y: y.at[m].set(jnp.nan),
                x),
            jnp.empty((int(time_limit/env.robot_dt)+1,)))
        metrics["min_distance"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.nanmin(min_distances)), lambda x: x.at[i].set(jnp.nan), metrics["min_distance"])
        space_compliances = lax.fori_loop(
            0,
            int(time_limit/env.robot_dt)+1,
            lambda s, x: lax.cond(
                s < episode_steps,
                lambda y: y.at[s].set(min_distances[s] > personal_space),
                lambda y: y.at[s].set(jnp.nan),
                x),
            jnp.empty((int(time_limit/env.robot_dt)+1,)))
        metrics["space_compliance"] = lax.cond(outcome["success"], lambda x: x.at[i].set(jnp.nanmean(space_compliances)), lambda x: x.at[i].set(jnp.nan), metrics["space_compliance"])
        seed += 1
        return seed, metrics
    
    # Initialize metrics
    metrics = {
        "successes": 0, 
        "collisions": 0, 
        "timeouts": 0, 
        "returns": jnp.empty((k,)),
        "times_to_goal": jnp.empty((k,)),
        "average_speed": jnp.empty((k,)),
        "average_acceleration": jnp.empty((k,)),
        "average_jerk": jnp.empty((k,)),
        "min_distance": jnp.empty((k,)),
        "space_compliance": jnp.empty((k,)),
        "episodic_spl": jnp.empty((k,)),
        "path_length": jnp.empty((k,))}
    # Execute k tests
    print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    _, metrics = lax.fori_loop(0, k, _fori_body, (random_seed, metrics))
    # Print results
    print("RESULTS")
    print(f"Success rate: {round(metrics['successes']/k,2):.2f}")
    print(f"Collision rate: {round(metrics['collisions']/k,2):.2f}")
    print(f"Timeout rate: {round(metrics['timeouts']/k,2):.2f}")
    print(f"Average return: {round(jnp.mean(metrics['returns']),2):.2f}")
    print(f"SPL: {round(jnp.mean(metrics['episodic_spl']),2):.2f}")
    print(f"Average time to goal: {round(jnp.nanmean(metrics['times_to_goal']),2):.2f} s")
    print(f"Average path length: {round(jnp.nanmean(metrics['path_length']),2):.2f} m")
    print(f"Average speed: {round(jnp.nanmean(metrics['average_speed']),2):.2f} m/s")
    print(f"Average acceleration: {round(jnp.nanmean(metrics['average_acceleration']),2):.2f} m/s^2")
    print(f"Average jerk: {round(jnp.nanmean(metrics['average_jerk']),2):.2f} m/s^3")
    print(f"Average space compliance: {round(jnp.nanmean(metrics['space_compliance']),2):.2f}")
    print(f"Average minimum distance to humans: {round(jnp.nanmean(metrics['min_distance']),2):.2f} m")
    return metrics
                
def animate_trajectory(
    states:jnp.ndarray, 
    humans_radiuses:np.ndarray, 
    robot_radius:float, 
    humans_policy:str,
    robot_goal:np.ndarray,
    scenario:int,
    robot_dt:float=0.25,
    lidar_measurements=None,
    kinematics:str='holonomic',
    ) -> None:

    # TODO: Add a progress bar,
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.78, top=0.90, bottom=0.05)
    ax.set_aspect('equal')
    ax.set(xlim=[-10,10],ylim=[-10,10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def animate(frame):
        ax.clear()
        ax.set_title(f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))} - Humans policy: {humans_policy.upper()}", weight='bold')
        ax.legend(
            handles=[
                Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Robot'), 
                Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans'),
                Line2D([0], [0], color='white', marker='*', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Goal')],
            bbox_to_anchor=(0.99, 0.5), loc='center left')
        ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)
        plot_state(ax, frame*robot_dt, states[frame], humans_policy, scenario, humans_radiuses, robot_radius, plot_time=False, kinematics=kinematics)
        if lidar_measurements is not None:
            plot_lidar_measurements(ax, lidar_measurements[frame], states[frame][-1], robot_radius)

    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=len(states))
    
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused

    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()

def save_policy_params(
    policy_name:str,
    policy_params: dict, 
    train_env_params:dict, 
    reward_params:dict,
    hyperparameters:dict, 
    path:str,
    filename=None
) -> None:

    if os.path.exists(path) == False:
        os.makedirs(path)

    if filename is None:
        today = date.today().strftime('%d_%m_%Y')
        filename = f"{policy_name}_nh{train_env_params['n_humans']}_hp{train_env_params['humans_policy']}_s{train_env_params['scenario']}_r{reward_params['type'][-1]}_{today}.pkl"
    else:
        filename = f"{filename}.pkl"
        
    with open(os.path.join(path, filename), 'wb') as f:
        pkl.dump({
            "policy_name": policy_name,
            "policy_params": policy_params,
            "train_env_params": {k: train_env_params[k] for k in set(list(train_env_params.keys())) - set(['reward_function'])},
            "reward_params": reward_params,
            "hyperparameters": hyperparameters}, f)

def load_socialjym_policy(
    path:str, 
) -> dict:
    with open(os.path.join(path), 'rb') as f:
        trained_policy = pkl.load(f)
        policy_params = trained_policy["policy_params"]
        return policy_params

def load_crowdnav_policy(
        policy_name:str,
        path:str) -> tuple:
    
    import torch

    ## Get Value Network Parameters (weights and biases)
    vnet_params = {}
    if policy_name == "cadrl":
        l = 0
        for k, v in torch.load(path, weights_only=True).items():
            if l%2 == 0:
                vnet_params[f"mlp/~/linear_{l//2}"] = {}
                vnet_params[f"mlp/~/linear_{l//2}"]["w"] = jnp.array(v.detach().cpu().numpy().T)
            else:
                vnet_params[f"mlp/~/linear_{l//2}"]["b"] = jnp.array(v.detach().cpu().numpy().T)
            l += 1
    elif policy_name == "sarl":
        l = 0
        for k, v in torch.load(path, weights_only=True).items():
            if l%2 == 0:
                if l == 0: vnet_params["value_network/~/mlp1/~/linear_0"] = {}
                if l == 2: vnet_params["value_network/~/mlp1/~/linear_1"] = {}
                if l == 4: vnet_params["value_network/~/mlp2/~/linear_0"] = {}
                if l == 6: vnet_params["value_network/~/mlp2/~/linear_1"] = {}
                if l == 8: vnet_params["value_network/~/attention/~/linear_0"] = {}
                if l == 10: vnet_params["value_network/~/attention/~/linear_1"] = {}
                if l == 12: vnet_params["value_network/~/attention/~/linear_2"] = {}
                if l == 14: vnet_params["value_network/~/mlp3/~/linear_0"] = {}
                if l == 16: vnet_params["value_network/~/mlp3/~/linear_1"] = {}
                if l == 18: vnet_params["value_network/~/mlp3/~/linear_2"] = {}
                if l == 20: vnet_params["value_network/~/mlp3/~/linear_3"] = {}
                vnet_params[list(vnet_params.keys())[l//2]]["w"] = jnp.array(v.detach().cpu().numpy().T)
            else:
                vnet_params[list(vnet_params.keys())[l//2]]["b"] = jnp.array(v.detach().cpu().numpy().T)
            l += 1
    return vnet_params
