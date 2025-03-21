from jax import random, vmap, debug
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl_ppo import SARLPPO
from socialjym.policies.sarl_a2c import SARLA2C
from socialjym.utils.aux_functions import animate_trajectory, load_crowdnav_policy, test_k_trials, load_socialjym_policy

### Hyperparameters
random_seed = 0 
n_episodes = 50
kinematics = "unicycle"
distribution = "dirichlet-bernoulli"
tests_n_humans = [5, 10, 15, 20, 25]
n_trials = 100
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = 1,
        time_penalty_reward = 0,
        high_rotation_penalty_reward = 0,
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
    )
env_params = {
    'robot_radius': 0.3,
    'n_humans': 15,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0, 1, 2, 3, 4, 5]),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy
policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics, distribution=distribution)
with open(os.path.join(os.path.dirname(__file__), 'il_out.pkl'), 'rb') as f:
    rl_out = pickle.load(f)
    actor_params = rl_out['actor_params']

# ### Test policy
# for n_humans in tests_n_humans:
#     test_env_params = {
#         'robot_radius': 0.3,
#         'n_humans': n_humans,
#         'robot_dt': 0.25,
#         'humans_dt': 0.01,
#         'robot_visible': True,
#         'scenario': env_params['scenario'],
#         'hybrid_scenario_subset': env_params['hybrid_scenario_subset'],
#         'humans_policy': env_params['humans_policy'],
#         'circle_radius': 7,
#         'reward_function': reward_function,
#         'kinematics': env_params['kinematics'],
#     }
#     test_env = SocialNav(**test_env_params)
#     metrics_after_rl = test_k_trials(
#         n_trials, 
#         random_seed, 
#         test_env, 
#         policy, 
#         actor_params, 
#         reward_function.time_limit)

### Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    episode_start_time = time.time()
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _, _, distr = policy.act(policy_key, obs, info, actor_params, False)
        print(distr)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True) 
        all_states = np.vstack((all_states, [state]))

    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'],
        kinematics=env_params['kinematics'])
    