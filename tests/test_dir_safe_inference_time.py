from jax import random, lax
import jax.numpy as jnp
import numpy as np
import time
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.policies.dir_safe import DIRSAFE

### Hyperparameters
random_seed = 0 
n_inferences = 1000
n_humans_large_scenario = 30
v_max = 1

### Initialize environment with small scenarios
env_params = {
    'robot_radius': 0.3,
    'n_humans': 3,
    'n_obstacles': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'perpendicular_traffic',
    'hybrid_scenario_subset': jnp.array([0, 1, 2, 3, 4, 6]), # All scenarios but circular_crossing_with_static_obstacles
    'humans_policy': 'hsfm',
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
}
env = SocialNav(**env_params)

### Initialize robot policy
policy = DIRSAFE(DummyReward(kinematics='unicycle'), v_max=v_max, dt=env_params['robot_dt'])
_, _, obs, info, _ = env.reset(random.PRNGKey(0))
actor_params, critic_params = policy.init_nns(random.PRNGKey(0), obs, info)

### Inferences on small scenarios
reset_key = random.PRNGKey(random_seed)
state, reset_key, obs, info, outcome = env.reset(reset_key)
all_times = []
for i in range(n_inferences):
    time_before_inference = time.time()
    action, _, _, _, distr = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
    all_times.append(time.time() - time_before_inference)
    state, obs, info, reward, outcome, reset_key = env.step(state,info,action,test=True,reset_key=reset_key,reset_if_done=True) 
print(f"Average inference time over {len(all_times)} inferences (small scenario {len(info['static_obstacles'][0])} obstacles, {env_params['n_humans']} humans)): {np.mean(all_times)*1000:.6f} ms ± {np.std(all_times)*1000:.6f} ms")

### Inferences on larger scenarios
## Load custom episodes
with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans_large_scenario}_humans.pkl'), 'rb') as f:
    custom_episodes = pickle.load(f)
## Initialize environment with larger scenarios
trial = 0
stacked_obstacles = jnp.stack([custom_episodes["static_obstacles"][trial,-1] for _ in range(n_humans_large_scenario+1)], axis=0)  # shape: (n_agents, n_obstacles, 4, 2, 2)
test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans_large_scenario,
    'n_obstacles': len(stacked_obstacles[0]),
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': None, # Custom scenario
    'humans_policy': 'hsfm',
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 0,
}
test_env = SocialNav(**test_env_params)
## Reset the environment
state, _, obs, info, outcome = test_env.reset_custom_episode(
    random.PRNGKey(0), # Not used, but required by the function
    {
        "full_state": custom_episodes["full_state"][trial],
        "robot_goal": custom_episodes["robot_goals"][trial,0],
        "humans_goal": custom_episodes["humans_goal"][trial],
        "static_obstacles": stacked_obstacles,
        "scenario": -1,
        "humans_radius": custom_episodes["humans_radius"][trial],
        "humans_speed": custom_episodes["humans_speed"][trial],
    }
)
all_times = []
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
    time_before_inference = time.time()
    action, _, _, _, distr = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
    all_times.append(time.time() - time_before_inference)
    state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)

print(f"Average inference time over {len(all_times)} inferences (large scenario {len(stacked_obstacles[0])} obstacles, {n_humans_large_scenario} humans): {np.mean(all_times)*1000:.6f} ms ± {np.std(all_times)*1000:.6f} ms")