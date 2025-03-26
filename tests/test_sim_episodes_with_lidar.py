from jax import jit, random, vmap, lax, debug
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import animate_trajectory, load_socialjym_policy

### Hyperparameters
random_seed = 0 
n_episodes = 50
kinematics = "unicycle"
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = True,
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
    )
env_params = {
    'robot_radius': 0.3,
    'n_humans': 18,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing_with_static_obstacles',
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}

### Initialize and reset environment
env = SocialNav(**env_params)
### Initialize robot policy
vnet_params = load_socialjym_policy(
    os.path.join(
        os.path.expanduser("~"),
        "Repos/social-jym/trained_policies/socialjym_policies/sarl_k1_nh5_hp2_s4_r1_20_11_2024.pkl"
    )
)
policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
### Simulate some episodes
@loop_tqdm(n_episodes)
@jit
def _simulate_episodes_with_lidar(i:int, for_val:tuple):
    @jit
    def _while_body(while_val:tuple):
        # Retrieve data from the tuple
        state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards = while_val
        action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
        # Save data
        all_actions = all_actions.at[steps].set(action)
        all_states = all_states.at[steps].set(state)
        all_rewards = all_rewards.at[steps].set(reward)
        # Update step counter
        steps += 1
        return state, obs, info, outcome, policy_key, steps, all_actions, all_states, all_rewards
    ## Retrieve data from the tuple
    seed, output_data = for_val
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
    ## Reset the environment
    state, reset_key, obs, info, init_outcome = env.reset(reset_key)
    initial_robot_position = state[-1,:2]
    robot_goal = info["robot_goal"]
    ## Episode loop
    all_actions = jnp.empty((int(reward_function.time_limit/env.robot_dt)+1, 2))
    all_states = jnp.empty((int(reward_function.time_limit/env.robot_dt)+1, env.n_humans+1, 6))
    all_rewards = jnp.empty((int(reward_function.time_limit/env.robot_dt)+1,))
    while_val_init = (state, obs, info, init_outcome, policy_key, 0, all_actions, all_states, all_rewards)
    _, _, _, outcome, policy_key, episode_steps, all_actions, all_states, all_rewards = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
    seed += 1
    return seed, output_data

    