import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from functools import partial
from jax import random, lax, jit, vmap
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import \
    animate_trajectory, \
    initialize_metrics_dict, \
    print_average_metrics, \
    compute_episode_metrics, \
    load_socialjym_policy, \
    test_k_trials_dwa, \
    test_k_trials, \
    test_k_trials_sfm, \
    test_k_trials_hsfm

### Hyperparameters
n_trials = 100
random_seed = 0
n_humans = [2,4,6]
n_obstacles = [2,4,6]
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
robot_vmax = 1.0
robot_wheels_distance = 0.7

### Initialize DIRSAFE policy
dirsafe = DIRSAFE(reward_function, v_max=robot_vmax, dt=0.25, wheels_distance=robot_wheels_distance)
with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
    dirsafe_params = pickle.load(f)['actor_params']
### Initialize SARL policy
sarl = SARL(reward_function, v_max=robot_vmax, dt=0.25, kinematics='unicycle', wheels_distance=robot_wheels_distance)
sarl_params = load_socialjym_policy(os.path.join(os.path.dirname(__file__), 'best_sarl.pkl'))

### Initialize output data structure
policies = ['DIR-SAFE', 'DWA', 'SARL', 'SFM', 'HSFM']
n_policies = len(policies)
metrics_dims = (n_policies,len(n_humans),len(n_obstacles))
all_metrics = initialize_metrics_dict(n_trials, dims=metrics_dims)

# ### Visualize one episode
# episode = 500
# n_humans = 4
# n_obstacles = 5
# ## Initialize environment
# test_env_params = {
#     'robot_radius': 0.3,
#     'n_humans': n_obstacles + n_humans,
#     'n_obstacles': 0, # n_obstacles is not used in this scenario
#     'robot_dt': 0.25,
#     'humans_dt': 0.01,
#     'robot_visible': True,
#     'scenario': 'circular_crossing_with_static_obstacles',
#     'humans_policy': 'hsfm',
#     'reward_function': reward_function,
#     'kinematics': 'unicycle',
#     'ccso_n_static_humans': n_obstacles,
# }
# test_env = SocialNav(**test_env_params)
# ## Reset the environment
# state, _, obs, info, outcome = test_env.reset(random.PRNGKey(episode))
# ## Compute obstacles if in "circular_crossing_with_static_obstacles" scenario (squares circumscribing humans disks)
# static_humans_positions = state[0:test_env_params['ccso_n_static_humans'],0:2]
# static_humans_radii = info['humans_parameters'][0:test_env_params['ccso_n_static_humans'],0]
# static_obstacles = jnp.array([dirsafe.batch_compute_disk_circumscribing_n_agon(static_humans_positions, static_humans_radii, n_edges=10)])
# nan_obstacles = jnp.full((test_env_params['n_humans'],) + static_obstacles.shape[1:], jnp.nan)
# static_obstacles = jnp.vstack((nan_obstacles, static_obstacles))
# ## Run the episode
# all_states = jnp.array([state])
# all_robot_goals = jnp.array([info['robot_goal']])
# while outcome['nothing']:
#     # Overwrite obstacles if in "circular_crossing_with_static_obstacles" scenario
#     aux_info = info.copy()
#     aux_info['static_obstacles'] = static_obstacles # Set obstacles as squares circumscribing static humans
#     aux_obs = obs[test_env_params['ccso_n_static_humans']:, :] # Remove static humans from observations (so they are not considered as humans by the policy, but only as obstacles)
#     # Step the environment
#     action, _, _, _, _ = dirsafe.act(random.PRNGKey(0), aux_obs, aux_info, actor_params, sample=False)
#     state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
#     # Save the state
#     all_states = jnp.vstack((all_states, jnp.array([state])))
#     all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
# ## Animate trajectory
# animate_trajectory(
#     all_states, 
#     info['humans_parameters'][:,0], 
#     test_env.robot_radius, 
#     test_env_params['humans_policy'],
#     all_robot_goals,
#     SCENARIOS.index('circular_crossing_with_static_obstacles'),
#     robot_dt=test_env_params['robot_dt'],
#     static_obstacles=static_obstacles[-1],
#     kinematics='unicycle',
#     vmax=1.,
#     wheels_distance=0.7,
#     figsize= (11, 6.6),
# )

### Test function (we create a dedicated test function for DIR-SAFE because we need to handle static obstacles not as humans, conversely to SARL)
def test_dir_safe_on_ccso(
    n_trials: int, 
    random_seed: int, 
    env: SocialNav, 
    dirsafe: DIRSAFE, 
    actor_params: dict, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    personal_space:float=0.5,
):
    @loop_tqdm(n_trials)
    @jit
    def _episode_loop(i:int, for_val:tuple):   
        @jit
        def _step_loop(while_val:tuple):
            state, obs, info, outcome, steps, static_obstacles, all_actions, all_states = while_val
            # Overwrite obstacles if in "circular_crossing_with_static_obstacles" scenario
            aux_info = info.copy()
            aux_info['static_obstacles'] = static_obstacles # Set obstacles as squares circumscribing static humans
            aux_obs = obs[test_env_params['ccso_n_static_humans']:, :] # Remove static humans from observations (so they are not considered as humans by the policy, but only as obstacles)
            # Step the environment
            action, _, _, _, _ = dirsafe.act(random.PRNGKey(0), aux_obs, aux_info, actor_params, sample=False)
            state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            # Update step counter
            steps += 1
            return state, obs, info, outcome, steps, static_obstacles, all_actions, all_states
        
        ## Retrieve data from the tuple
        seed, metrics = for_val
        ## Reset the environment
        reset_key = random.PRNGKey(random_seed + seed)
        state, reset_key, obs, info, init_outcome = env.reset(reset_key)
        ## Save initial robot position
        initial_robot_position = state[-1,:2]
        ## Compute static obstacles correspondig to static humans (decagon circumscribing humans disks)
        static_humans_positions = state[0:env.ccso_n_static_humans,0:2]
        static_humans_radii = info['humans_parameters'][0:env.ccso_n_static_humans,0]
        static_obstacles = jnp.array([dirsafe.batch_compute_disk_circumscribing_n_agon(static_humans_positions, static_humans_radii, n_edges=10)])
        nan_obstacles = jnp.full((env.n_humans,) + static_obstacles.shape[1:], jnp.nan)
        static_obstacles = jnp.vstack((nan_obstacles, static_obstacles))
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val_init = (state, obs, info, init_outcome, 0, static_obstacles, all_actions, all_states)
        _, _, end_info, outcome, episode_steps, static_obstacles, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _step_loop, while_val_init)
        ## Update metrics
        metrics = compute_episode_metrics(
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': dirsafe.v_max, 'wheels_distance': dirsafe.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics

    ## Check that the environment scenario is correct
    assert env.scenario == SCENARIOS.index("circular_crossing_with_static_obstacles"), "This function is designed to work with the 'circular_crossing_with_static_obstacles' scenario only"
    assert dirsafe.name == 'DIRSAFE', "This function is designed to work with the 'DIR-SAFE' policy only"    
    ## Initialize metrics
    metrics = initialize_metrics_dict(n_trials)
    ## Execute n_trials tests
    print(f"\nExecuting {n_trials} tests with {env.n_humans - env.ccso_n_static_humans} dynamic humans and {env.ccso_n_static_humans} static humans...")
    _, metrics = lax.fori_loop(0, n_trials, _episode_loop, (random_seed, metrics))
    ## Print average results
    print_average_metrics(n_trials, metrics)
    return metrics

### Execute tests
for i, nh in enumerate(n_humans):
    for j, no in enumerate(n_obstacles):
        ## Initialize environment
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': no + nh,
            'n_obstacles': 0, # n_obstacles is not used in this scenario
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': 'circular_crossing_with_static_obstacles',
            'humans_policy': 'hsfm',
            'reward_function': reward_function,
            'kinematics': 'unicycle',
            'ccso_n_static_humans': no,
        }
        test_env = SocialNav(**test_env_params)
        ## DIR-SAFE tests
        print("\nDIR-SAFE Tests")
        metrics_dir_safe = test_dir_safe_on_ccso(n_trials, random_seed, test_env, dirsafe, dirsafe_params, time_limit=50.)
        ## DWA Tests
        print("\nDWA Tests")
        metrics_dwa = test_k_trials_dwa(n_trials, random_seed, test_env, time_limit=50, robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
        ## SARL Tests
        print("\nSARL Tests")
        metrics_sarl = test_k_trials(n_trials, random_seed, test_env, sarl, sarl_params, time_limit=50.)
        ## SFM Tests
        print("\nSFM Tests")
        metrics_sfm = test_k_trials_sfm(n_trials, random_seed, test_env, time_limit=50., robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
        ## HSFM Tests
        print("\nHSFM Tests")
        metrics_hsfm = test_k_trials_hsfm(n_trials, random_seed, test_env, time_limit=50., robot_vmax=robot_vmax, robot_wheels_distance=robot_wheels_distance)
        ### Store results
        all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_dir_safe)
        all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_dwa)
        all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_sarl)
        all_metrics = tree_map(lambda x, y: x.at[3,i,j].set(y), all_metrics, metrics_sfm)
        all_metrics = tree_map(lambda x, y: x.at[4,i,j].set(y), all_metrics, metrics_hsfm)

### Save results
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_results.pkl')):
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_results.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
### Load results
else:
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_benchmark_results.pkl'), 'rb') as f:
        all_metrics = pickle.load(f)