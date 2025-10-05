from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl_star import SARLStar
from socialjym.utils.aux_functions import animate_trajectory, load_socialjym_policy

# Hyperparameters
random_seed = 1
n_episodes = 50
kinematics = 'unicycle'
reward_function = Reward2(
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
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'n_obstacles': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5,6,7], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
    'grid_map_computation': True, # Enable grid map computation for global planning
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARLStar(
    reward_function, 
    env.get_grid_size(), 
    planner="A*", 
    v_max=1.0, 
    dt=0.25, 
    kinematics='unicycle', 
    wheels_distance=0.7
)
policy_params = load_socialjym_policy(os.path.join(os.path.dirname(__file__), 'best_sarl.pkl'))
# plt.plot(policy.action_space[:,0], policy.action_space[:,1],'.')
# plt.show()

# Warm up the environment and policy - Dummy step and act to jit compile the functions 
# (this way, computation time will only reflect execution and not compilation)
state, _, _, info, _ = env.reset(random.key(0))
_, obs, _, _, _, _ = env.step(state,info,jnp.zeros((2,)))
_ = policy.act(random.key(0), obs, info, policy_params, 0.1)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    episode_start_time = time.time()
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = jnp.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, policy_params, 0.)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
        print(f"Return in steps [0,{info['step']}):", info["return"], f" - time : {info['time']}")
        all_states = jnp.vstack((all_states, [state]))
    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'],
        kinematics=kinematics,
        static_obstacles=info['static_obstacles'][-1],
    )