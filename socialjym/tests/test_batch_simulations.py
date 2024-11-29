from jax import random, debug, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import load_socialjym_policy

# Hyperparameters
random_seed = 1
n_simulations = 5
kinematics = 'unicycle'
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
reward_function = Reward1(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/cadrl_k1_nh1_hp1_s4_r1_20_11_2024.pkl"))

# Simulate some episodes
policy_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)
reset_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)
states, reset_keys, obses, infos = env.batch_reset(reset_keys)
outcomes = {
    'nothing': jnp.ones(n_simulations, dtype=bool),
    'failure': jnp.zeros(n_simulations, dtype=bool),
    'success': jnp.zeros(n_simulations, dtype=bool),
    'timeout': jnp.zeros(n_simulations, dtype=bool),
}
while jnp.any(outcomes['nothing']):
    # Evaluate ongoing episodes
    go_idxs = jnp.where(outcomes['nothing'] == True)[0]
    # Filter for ongoing episodes
    go_states = states[go_idxs]
    go_obses = obses[go_idxs]
    go_policy_keys = policy_keys[go_idxs]
    go_infos = tree_map(lambda x: x[go_idxs], infos)
    # Compute actions
    go_actions, go_policy_keys, _ = policy.batch_act(go_policy_keys, go_obses, go_infos, vnet_params, 0.)
    # Make environment step
    go_states, go_obses, go_infos, go_rewards, go_outcomes = env.batch_step(go_states, go_infos, go_actions, test=True)
    # Update overall data
    states = states.at[go_idxs].set(go_states)
    obses = obses.at[go_idxs].set(go_obses)
    infos = tree_map(lambda x, y: x.at[go_idxs].set(y), infos, go_infos)
    outcomes = tree_map(lambda x, y: x.at[go_idxs].set(y), outcomes, go_outcomes)
print(outcomes)