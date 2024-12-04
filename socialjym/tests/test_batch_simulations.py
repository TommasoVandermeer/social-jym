from jax import random, debug, vmap, jit, lax
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
from socialjym.utils.aux_functions import load_socialjym_policy, test_k_trials

# Hyperparameters
random_seed = 1
n_simulations = 1000
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
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/sarl_k1_nh5_hp2_s4_r1_20_11_2024.pkl"))

# Simulate some episodes
policy_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)
reset_keys = vmap(random.PRNGKey)(jnp.arange(n_simulations, dtype=int) + random_seed)
states, reset_keys, obses, infos = env.batch_reset(reset_keys)
output = {
    'all_outcomes': {
        'nothing': jnp.zeros((n_simulations, int(env.reward_function.time_limit/env.robot_dt)), dtype=bool),
        'failure': jnp.zeros((n_simulations, int(env.reward_function.time_limit/env.robot_dt)), dtype=bool),
        'success': jnp.zeros((n_simulations, int(env.reward_function.time_limit/env.robot_dt)), dtype=bool),
        'timeout': jnp.zeros((n_simulations, int(env.reward_function.time_limit/env.robot_dt)), dtype=bool),
    },
    'all_states': jnp.empty((n_simulations, int(env.reward_function.time_limit/env.robot_dt), env.n_humans+1, 6), dtype=jnp.float32),
    'all_obses': jnp.empty((n_simulations, int(env.reward_function.time_limit/env.robot_dt),  env.n_humans+1, 6), dtype=jnp.float32),
    'all_rewards': jnp.empty((n_simulations, int(env.reward_function.time_limit/env.robot_dt)), dtype=jnp.float32),
    'all_actions': jnp.empty((n_simulations, int(env.reward_function.time_limit/env.robot_dt), 2), dtype=jnp.float32),
}

@jit
def _while_body(val:tuple):
    output, states, obses, policy_keys, infos, steps, dones, end_steps = val
    # Compute actions
    actions, policy_keys, _ = policy.batch_act(policy_keys, obses, infos, vnet_params, 0.)
    # Make environment step
    states, obses, infos, rewards, outcomes = env.batch_step(states, infos, actions, test=True)
    # Save data
    output['all_outcomes']['nothing'] = output['all_outcomes']['nothing'].at[:,steps].set(outcomes['nothing'])
    output['all_outcomes']['failure'] = output['all_outcomes']['failure'].at[:,steps].set(outcomes['failure'])
    output['all_outcomes']['success'] = output['all_outcomes']['success'].at[:,steps].set(outcomes['success'])
    output['all_outcomes']['timeout'] = output['all_outcomes']['timeout'].at[:,steps].set(outcomes['timeout'])
    output['all_states'] = output['all_states'].at[:,steps].set(states)
    output['all_obses'] = output['all_obses'].at[:,steps].set(obses)
    output['all_rewards'] = output['all_rewards'].at[:,steps].set(rewards)
    output['all_actions'] = output['all_actions'].at[:,steps].set(actions)
    # Update dones
    dones = jnp.logical_or(jnp.logical_not(outcomes['nothing']),dones)
    # Increment step counter and update end_steps
    steps += 1
    @jit
    def _update_end_steps(done:bool, end_step:int, step:int):
        return lax.cond(
            jnp.logical_and(done, end_step > step), 
            lambda _: step,
            lambda x: x, 
            end_step)
    end_steps = vmap(_update_end_steps, in_axes=(0,0,None))(dones, end_steps, steps)
    return output, states, obses, policy_keys, infos, steps, dones, end_steps

# Warm-up
_ = lax.while_loop(
    lambda x: jnp.logical_not(jnp.all(x[-2])),
    _while_body,
    (output, states, obses, policy_keys, infos, 0, jnp.ones(n_simulations, dtype=bool), jnp.ones(n_simulations, dtype=int)*int(env.reward_function.time_limit/env.robot_dt)))
_ = test_k_trials(1, 0, env, policy, vnet_params, reward_function.time_limit, print_avg_metrics=False)

# Parallel simulations
print("\n\nPARALLEL SIMULATIONS")
init_time = time.time()
output, _, _, _, _, max_steps, dones, end_steps = lax.while_loop(
    lambda x: jnp.logical_not(jnp.all(x[-2])),
    _while_body,
    (output, states, obses, policy_keys, infos, 0, jnp.zeros(n_simulations, dtype=bool), jnp.ones(n_simulations, dtype=int)*int(env.reward_function.time_limit/env.robot_dt)))
end_time = time.time()
print("Execution time: ", end_time-init_time, " seconds")
print("Successes: ", jnp.sum(output['all_outcomes']['success'][jnp.arange(n_simulations),end_steps-1]))
print("Failures: ", jnp.sum(output['all_outcomes']['failure'][jnp.arange(n_simulations),end_steps-1]))
print("Timeouts: ", jnp.sum(output['all_outcomes']['timeout'][jnp.arange(n_simulations),end_steps-1]))

# Serial simulations
print("\n\nSERIAL SIMULATIONS")
init_time = time.time()
metrics = test_k_trials(n_simulations, 0, env, policy, vnet_params, reward_function.time_limit, print_avg_metrics=False)
end_time = time.time()
print("Execution time: ", end_time-init_time, " seconds")