import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import jit, lax, vmap, random
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import animate_trajectory
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl_ppo import SARLPPO

# Hyperparameters
random_seed = 1
n_simulations = 10
steps_to_simulate = 100
kinematics = 'unicycle'
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'progress_to_goal_reward': True,
    'time_penalty_reward': False,
    'high_rotation_penalty_reward': False,
    'progress_to_goal_weight': 0.03,
}
reward_function = Reward2(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
actor_params = policy.actor.init(random_seed, jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))

# Initialize random keys
policy_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)
reset_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)

# Reset environment
states, reset_keys, obses, infos, outcomes = env.batch_reset(reset_keys)

# Initialize data to save
all_states = jnp.empty((n_simulations, steps_to_simulate, env.n_humans+1, 6), dtype=jnp.float32)
all_obses = jnp.empty((n_simulations, steps_to_simulate, env.n_humans+1, 6), dtype=jnp.float32)
all_infos = {
            "humans_goal": jnp.zeros((n_simulations, steps_to_simulate, env.n_humans, 2), dtype=jnp.float32),
            "robot_goal": jnp.zeros((n_simulations, steps_to_simulate, 2), dtype=jnp.float32),
            "humans_parameters": jnp.zeros((n_simulations, steps_to_simulate, env.n_humans, 19), dtype=jnp.float32),
            "static_obstacles": jnp.zeros((n_simulations, steps_to_simulate, 1, 1, 2, 2), dtype=jnp.float32),
            "time": jnp.zeros((n_simulations, steps_to_simulate), dtype=jnp.float32),
            "current_scenario": jnp.zeros((n_simulations, steps_to_simulate), dtype=int),
            "humans_delay": jnp.zeros((n_simulations, steps_to_simulate, env.n_humans), dtype=jnp.float32),
        }
all_rewards = jnp.empty((n_simulations, steps_to_simulate), dtype=jnp.float32)
all_outcomes = {
    'nothing': jnp.zeros((n_simulations, steps_to_simulate), dtype=bool),
    'failure': jnp.zeros((n_simulations, steps_to_simulate), dtype=bool),
    'success': jnp.zeros((n_simulations, steps_to_simulate), dtype=bool),
    'timeout': jnp.zeros((n_simulations, steps_to_simulate), dtype=bool),
}
all_dones = jnp.zeros((n_simulations, steps_to_simulate), dtype=bool)

# Simulate some steps
@jit
def _fori_body(i:int, val:tuple):
    states, reset_keys, obses, infos, outcomes, policy_keys, reset_keys, all_states, all_obses, all_infos, all_rewards, all_outcomes, all_dones = val
    # Compute actions
    actions, policy_keys, _, _, _ = policy.batch_act(policy_keys, obses, infos, actor_params, 1.)
    # Step env
    states, obses, infos, rewards, outcomes, reset_keys = env.batch_step(states, infos, actions, reset_keys, False, True)
    # Compute dones
    all_dones = all_dones.at[:,i].set(outcomes['failure'] | outcomes['success'] | outcomes['timeout'])
    # Save data
    all_states = all_states.at[:,i].set(states)
    all_obses = all_obses.at[:,i].set(obses)
    all_infos = tree_map(lambda x, y: x.at[:,i].set(y), all_infos, infos)
    all_rewards = all_rewards.at[:,i].set(rewards)
    all_outcomes = tree_map(lambda x, y: x.at[:,i].set(y), all_outcomes, outcomes)
    return states, reset_keys, obses, infos, outcomes, policy_keys, reset_keys, all_states, all_obses, all_infos, all_rewards, all_outcomes, all_dones

_, _, _, _, _, _, _, all_states, all_obses, all_infos, all_rewards, all_outcomes, all_dones = lax.fori_loop(
    0, 
    steps_to_simulate, 
    _fori_body, 
    (states, reset_keys, obses, infos, outcomes, policy_keys, reset_keys, all_states, all_obses, all_infos, all_rewards, all_outcomes, all_dones)
)

# Animate episodes
for i, states in enumerate(all_states):
    ## Animate trajectory
    animate_trajectory(
        states, 
        all_infos['humans_parameters'][i,0,:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        all_infos['robot_goal'][i,0],
        all_infos['current_scenario'][i,0],
        robot_dt=env_params['robot_dt'],
        kinematics=env_params['kinematics'])