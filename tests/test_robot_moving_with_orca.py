import jax.numpy as jnp
from jax import lax, jit, random, vmap, debug

from jhsfm.hsfm import get_linear_velocity
from jorca.orca import single_update
from jorca.utils import get_standard_humans_parameters
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import animate_trajectory
from socialjym.envs.base_env import HUMAN_POLICIES, ROBOT_KINEMATICS, SCENARIOS, wrap_angle

humans_policy = 'hsfm'
robot_vmax = 1.0
key = random.PRNGKey(42)
n_obstacles = 3
n_humans = 2
env_params = {
    'robot_radius': 0.3,
    'n_humans': n_obstacles + n_humans,
    'n_obstacles': 0, # n_obstacles is not used in this scenario
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing_with_static_obstacles',
    'humans_policy': humans_policy,
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
    'ccso_n_static_humans': n_obstacles,
}
env = SocialNav(**env_params)

@jit
def _while_body(while_val:tuple):
    # Retrieve data from the tuple
    prev_state, state, obs, info, outcome, steps, all_actions, all_states, all_robot_goals = while_val
    temp = jnp.copy(state)
    # Make a step in the environment
    if env.humans_policy == HUMAN_POLICIES.index('hsfm'):
        # Setup humans parameters
        parameters = get_standard_humans_parameters(env.n_humans+1)
        parameters = parameters.at[-1,0].set(env.robot_radius) # Set robot radius
        parameters = parameters.at[-1,2].set(robot_vmax) # Set robot max speed
        parameters = parameters.at[-1,-1].set(0.02) # Set a safety margin for the robot
        parameters = parameters.at[:-1,0].set(info["humans_parameters"][:,0]) # Set humans radius
        parameters = parameters.at[:-1,2].set(info["humans_parameters"][:,2]) # Set humans max_speed
        # Convert HSFM state to ORCA state
        humans_lin_vel = vmap(get_linear_velocity, in_axes=(0, 0))(state[:-1,4], state[:-1,2:4])
        feed_state = jnp.copy(state[:,:4])
        feed_state = feed_state.at[:-1,2:4].set(humans_lin_vel)
        feed_state = feed_state.at[-1,2:4].set((state[-1,0:2]-prev_state[-1,0:2])/env.robot_dt)
        # Step the robot in ORCA environment
        new_robot_state = single_update(
            -1,
            feed_state, 
            info["robot_goal"], 
            parameters,
            info["static_obstacles"][-1],
            env.robot_dt
        )
        # Compute action
        if env.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            action = jnp.array([jnp.linalg.norm(new_robot_state[2:4]), wrap_angle(jnp.atan2(new_robot_state[3], new_robot_state[2]) - state[-1,4]) / env.robot_dt])
        elif env.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            action = state[-1,2:4]
    elif env.humans_policy == HUMAN_POLICIES.index('sfm'):
        pass
    elif env.humans_policy == HUMAN_POLICIES.index('orca'):
        pass
    debug.print("Robot state {x}, robot action {y}", x=state[-1], y=action)
    state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
    # Update prev_state
    prev_state = jnp.copy(temp)
    # Save data
    all_actions = all_actions.at[steps].set(action)
    all_states = all_states.at[steps].set(state)
    all_robot_goals = all_robot_goals.at[steps].set(info['robot_goal'])
    # Update step counter
    steps += 1
    return prev_state, state, obs, info, outcome, steps, all_actions, all_states, all_robot_goals

# Reset environment
state, reset_key, obs, info, init_outcome = env.reset(key)
## Episode loop
all_actions = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
all_states = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, env.n_humans+1, 6))
all_robot_goals = jnp.empty((int(env.reward_function.time_limit/env.robot_dt)+1, 2))
while_val_init = (state, state, obs, info, init_outcome, 0, all_actions, all_states, all_robot_goals)
_, _, _, end_info, outcome, episode_steps, all_actions, all_states, all_robot_goals = lax.while_loop(lambda x: x[4]["nothing"] == True, _while_body, while_val_init)

## Animate the trajectory
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