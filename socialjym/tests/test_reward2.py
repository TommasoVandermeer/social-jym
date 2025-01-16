import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import matplotlib.pyplot as plt

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import animate_trajectory, plot_state, plot_trajectory
from socialjym.policies.cadrl import CADRL
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

kinematics = 'unicycle'
random_seed = 0
n_episodes = 10

# Define reward parameters
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'target_reached_reward': False,
    'collision_penalty_reward': False,
    'discomfort_penalty_reward': False,
    'progress_to_goal_reward': False,
    'time_penalty_reward': False,
    'high_rotation_penalty_reward': False,
}
reward_function = Reward2(**reward_params)
print("Reward type: ", reward_function.type)

# Initialize and reset environment
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
env = SocialNav(**env_params)

# Initialize robot policy and vnet params
policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
initial_vnet_params = policy.model.init(random.key(random_seed), jnp.zeros((env_params["n_humans"],policy.vnet_input_size)))

### Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, initial_vnet_params, 0.)
        state, obs, info, reward, outcome = env.step(state,info,action,test=True) 
        all_states = np.vstack((all_states, [state]))

    ## Plot episode trajectory
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
        plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    # plot last state
    plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    plt.show()

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