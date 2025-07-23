import jax.numpy as jnp
from jax import random, lax
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.soappo import SOAPPO
from socialjym.utils.aux_functions import animate_trajectory

### Hyperparameters
policy = 'dir-safe' # 'dir-safe' or 'dwa'
trial = 15
n_humans = 20
time_limit = 100.
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

# Load custom episodes
with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans}_humans.pkl'), 'rb') as f:
    custom_episodes = pickle.load(f)

### Initialize environment
test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans,
    'n_obstacles': len(custom_episodes['static_obstacles'][trial,0]),
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
if policy == 'dir-safe':
    ## Initialize robot policy
    policy = SOAPPO(reward_function, v_max=1., dt=0.25)
    with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
        actor_params = pickle.load(f)['actor_params']
    ## Reset the environment
    state, _, obs, info, outcome = test_env.reset_custom_episode(
        random.PRNGKey(0), # Not used, but required by the function
        {
            "full_state": custom_episodes["full_state"][trial],
            "robot_goal": custom_episodes["robot_goals"][trial,0],
            "humans_goal": custom_episodes["humans_goal"][trial],
            "static_obstacles": custom_episodes["static_obstacles"][trial],
            "scenario": -1,
            "humans_radius": custom_episodes["humans_radius"][trial],
            "humans_speed": custom_episodes["humans_speed"][trial],
        }
    )
    ## Run the episode
    all_states = jnp.array([state])
    all_robot_goals = jnp.array([info['robot_goal']])
    while outcome['nothing']:
        # Update robot goal
        info["robot_goal"], info["robot_goal_index"] = lax.cond(
            (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= test_env.robot_radius*2) & # Waypoint reached threshold is set to be higher
            (info['robot_goal_index'] < len(custom_episodes["robot_goals"][trial])-1) & # Check if current goal is not the last one
            (~(jnp.any(jnp.isnan(custom_episodes["robot_goals"][trial,info['robot_goal_index']+1])))), # Check if next goal is not NaN
            lambda _: (custom_episodes["robot_goals"][trial,info['robot_goal_index']+1], info['robot_goal_index']+1),
            lambda x: x,
            (info["robot_goal"], info["robot_goal_index"])
        )
        # Update humans goal
        info["humans_goal"] = lax.fori_loop(
            0, 
            test_env.n_humans, 
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
        # Step the environment
        action, _, _, _, _ = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
        state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
        # Save the state
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
elif policy == 'dwa':
    pass
else:
    raise ValueError(f'Policy {policy} not available.')

### Animate trajectory
animate_trajectory(
    all_states, 
    info['humans_parameters'][:,0], 
    test_env.robot_radius, 
    test_env_params['humans_policy'],
    all_robot_goals,
    None, # Custom scenario
    robot_dt=test_env_params['robot_dt'],
    static_obstacles=custom_episodes['static_obstacles'][trial,0],
    kinematics='unicycle',
    vmax=1.,
    wheels_distance=policy.wheels_distance,
    figsize= (11, 6.6),
)