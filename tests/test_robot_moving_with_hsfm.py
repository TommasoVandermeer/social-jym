import jax.numpy as jnp
from jax import lax, jit, random, vmap, debug
import matplotlib.pyplot as plt

from jhsfm.hsfm import get_linear_velocity
from jhsfm.hsfm import single_update
from jhsfm.utils import get_standard_humans_parameters
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import animate_trajectory
from socialjym.envs.base_env import HUMAN_POLICIES, ROBOT_KINEMATICS, SCENARIOS, wrap_angle

humans_policy = 'sfm'
robot_dt = 0.25
robot_vmax = 1.0
robot_wheels_distance = 0.7
key = random.PRNGKey(0)
n_obstacles = 3
n_humans = 5
env_params = {
    'robot_radius': 0.3,
    'n_humans': n_obstacles + n_humans,
    'n_obstacles': 0, # n_obstacles is not used in this scenario
    'robot_dt': robot_dt,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing_with_static_obstacles',
    'humans_policy': humans_policy,
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
    'ccso_n_static_humans': n_obstacles,
}
env = SocialNav(**env_params)

assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "This test is only for unicycle robots."

@jit
def _while_body(while_val:tuple):
    # Retrieve data from the tuple
    prev_state, state, obs, info, outcome, steps, all_actions, all_states, all_robot_goals = while_val
    temp = jnp.copy(state)
    ### COMPUTE ROBOT ACTION ###
    if env.humans_policy == HUMAN_POLICIES.index('hsfm'):
        # Setup humans parameters
        parameters = jnp.vstack((info["humans_parameters"], jnp.array([env.robot_radius, 80., robot_vmax, *get_standard_humans_parameters(1)[0,3:]])))
        # Setup robot state for HSFM
        humans_lin_vel = vmap(get_linear_velocity, in_axes=(0, 0))(state[:-1,4], state[:-1,2:4])
        feed_state = jnp.copy(state)
    elif env.humans_policy == HUMAN_POLICIES.index('sfm') or env.humans_policy == HUMAN_POLICIES.index('orca'):
        # Setup humans parameters
        parameters = get_standard_humans_parameters(env.n_humans+1)
        parameters = parameters.at[-1,0].set(env.robot_radius) # Set robot radius
        parameters = parameters.at[-1,2].set(robot_vmax) # Set robot max speed
        parameters = parameters.at[:-1,0].set(info["humans_parameters"][:,0]) # Set humans radius
        parameters = parameters.at[:-1,2].set(info["humans_parameters"][:,2]) # Set humans max_speed
        # Convert SFM/ORCA state to HSFM state
        humans_lin_vel = state[:-1,2:4]
        feed_state = jnp.copy(state)
        humans_theta = jnp.arctan2(humans_lin_vel[:,1], humans_lin_vel[:,0])
        previous_humans_lin_vel = prev_state[:-1,2:4]
        previous_humans_theta = jnp.arctan2(previous_humans_lin_vel[:,1], previous_humans_lin_vel[:,0])
        @jit
        def _get_body_and_ang_velocity(lin_vel, theta, previous_theta):
            rotational_matrix = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
            angular_velocity = wrap_angle(theta - previous_theta) / env.humans_dt
            return rotational_matrix @ lin_vel, angular_velocity
        humans_body_vel, humans_angular_vel = vmap(_get_body_and_ang_velocity, in_axes=(0,0,0))(
            humans_lin_vel, 
            humans_theta, 
            previous_humans_theta
        )
        feed_state = feed_state.at[:-1,2:4].set(humans_body_vel)
        feed_state = feed_state.at[:-1,4].set(humans_theta)
        feed_state = feed_state.at[:-1,5].set(humans_angular_vel)
    # Set robot feed state
    linear_velocity = (state[-1,:2]-prev_state[-1,0:2])/env.robot_dt
    robot_theta = state[-1,4] 
    rotational_matrix = jnp.array([[jnp.cos(robot_theta), jnp.sin(robot_theta)], [-jnp.sin(robot_theta), jnp.cos(robot_theta)]])
    body_velocity = rotational_matrix @ linear_velocity
    feed_state = feed_state.at[-1,2:4].set(body_velocity)
    angular_velocity = wrap_angle(robot_theta - prev_state[-1,4]) / env.robot_dt
    feed_state = feed_state.at[-1,5].set(angular_velocity)
    # Step the robot in HSFM environment (doing it with substeps to avoid instabilities)
    @jit
    def _substep(i, feed_state):
        new_robot_state = single_update(
            -1,
            feed_state, 
            info["robot_goal"], 
            parameters,
            info["static_obstacles"][-1],
            env.humans_dt
        )
        feed_state = feed_state.at[:-1,0].set(feed_state[:-1,0] + humans_lin_vel[:,0] * env.humans_dt)
        feed_state = feed_state.at[:-1,1].set(feed_state[:-1,1] + humans_lin_vel[:,1] * env.humans_dt)
        feed_state = feed_state.at[-1].set(new_robot_state)
        return feed_state
    new_feed_state = lax.fori_loop(0, int(env.robot_dt/env.humans_dt), _substep, feed_state)
    new_robot_state = new_feed_state[-1]
    # Compute action
    new_velocity = (new_robot_state[:2] - state[-1,:2]) / env.robot_dt
    action = jnp.array([jnp.linalg.norm(new_velocity), wrap_angle(new_robot_state[4] - state[-1,4]) / env.robot_dt])
    ### STEP THE ENVIRONMENT ###
    state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
    ### Update prev_state
    prev_state = jnp.copy(temp)
    ### Save data
    all_actions = all_actions.at[steps].set(action)
    all_states = all_states.at[steps].set(state)
    all_robot_goals = all_robot_goals.at[steps].set(info['robot_goal'])
    ### Update step counter
    steps += 1
    return prev_state, state, obs, info, outcome, steps, all_actions, all_states, all_robot_goals

### Reset environment
state, reset_key, obs, info, init_outcome = env.reset(key)
### Episode loop
all_actions = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
all_states = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, env.n_humans+1, 6))
all_robot_goals = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
while_val_init = (state, state, obs, info, init_outcome, 0, all_actions, all_states, all_robot_goals)
_, _, _, end_info, outcome, episode_steps, all_actions, all_states, all_robot_goals = lax.while_loop(lambda x: x[4]["nothing"] == True, _while_body, while_val_init)

### Animate the trajectory
animate_trajectory(
    all_states[:episode_steps], 
    info['humans_parameters'][:,0], 
    env.robot_radius, 
    env_params['humans_policy'],
    all_robot_goals[:episode_steps],
    SCENARIOS.index('circular_crossing_with_static_obstacles'),
    robot_dt=env_params['robot_dt'],
    static_obstacles=info['static_obstacles'],
    kinematics='unicycle',
    vmax=1.,
    wheels_distance=0.7,
    figsize= (11, 6.6),
)

### Plot triangular unicycle action space an scatter the actions taken by the robot
# Compute percentage of actions taken outside the action space
actions_taken = all_actions[:episode_steps]
actions_taken = actions_taken.at[:].set(jnp.round(actions_taken, 2))
out_of_boundary_conditions = \
    (jnp.abs(actions_taken[:,1]) - (2 * (robot_vmax - actions_taken[:,0]) / robot_wheels_distance) > 1e-2) | \
    (actions_taken[:,0] < 0) | \
    (actions_taken[:,0] > robot_vmax)
print(f'Actions outside: {actions_taken[jnp.where(out_of_boundary_conditions)]}')
outside = jnp.sum(jnp.where(out_of_boundary_conditions, 1, 0))
print(f'Actions taken outside the action space: {outside} out of {episode_steps} ({100*outside/episode_steps:.2f}%)')
# Plot
fig, ax = plt.subplots(figsize=(6,6))
w_max = 2 * robot_vmax / robot_wheels_distance
triangle = jnp.array([[0,w_max], [robot_vmax,0], [0,-w_max], [0,w_max]])
ax.plot(triangle[:,0], triangle[:,1], 'k-')
ax.fill(triangle[:,0], triangle[:,1], color='lightgrey', alpha=0.5)
ax.scatter(all_actions[:episode_steps,0], all_actions[:episode_steps,1], c='red', s=25)
ax.set_xlabel(r'Linear velocity $v$ [m/s]')
ax.set_ylabel(r'Angular velocity $\omega$ [rad/s]')
ax.set_title('Robot action space and actions taken')
ax.set_xlim([jnp.min(jnp.append(actions_taken[:,0], 0.))-0.1, jnp.max(jnp.append(actions_taken[:,0], robot_vmax))+0.1])
ax.set_ylim([jnp.min(jnp.append(actions_taken[:,1], -w_max))-0.1, jnp.max(jnp.append(actions_taken[:,1], w_max))+0.1])
ax.grid()
plt.show()