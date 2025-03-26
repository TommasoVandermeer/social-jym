from jax import jit, random, debug, lax
from jax_tqdm import loop_tqdm
from jax.tree_util import tree_map
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import ROBOT_KINEMATICS
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import animate_trajectory, load_socialjym_policy

### Hyperparameters
filter_timeouts = True
filter_failures = True
random_seed = 0 
n_steps = 10_000
kinematics = "unicycle"
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = False,
        discomfort_distance=0.2,
        progress_to_goal_weight=0.03,
    )
env_params = {
    'robot_radius': 0.3,
    'n_humans': 10,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'circular_crossing',
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
    'lidar_angular_range':jnp.pi,
    'lidar_max_dist':10.,
    'lidar_num_rays':60,
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
### Initialize output data dictionary
output_data = {
    "actions": jnp.zeros((n_steps,2)),
    "states": jnp.zeros((n_steps,env_params["n_humans"]+1,6)),
    "rewards": jnp.zeros((n_steps,)),
    "outcomes": {
        "nothing": jnp.zeros((n_steps,), dtype=bool),
        "failure": jnp.zeros((n_steps,), dtype=bool),
        "success": jnp.zeros((n_steps,), dtype=bool),
        "timeout": jnp.zeros((n_steps,), dtype=bool),
    },
    "dones": jnp.zeros((n_steps,), dtype=bool),
    "lidar_measurements": jnp.zeros((n_steps,env.lidar_num_rays)),
}
### Simulate n_steps steps
print(f"Simulating {n_steps} steps...")
@loop_tqdm(n_steps)
@jit
def _simulate_steps_with_lidar(i:int, for_val:tuple):
    ## Retrieve data from the tuple
    output_data, state, obs, info, reset_key = for_val
    ## Simulate one step
    if policy.kinematics == ROBOT_KINEMATICS.index("holonomic"):
        lidar_measurements = env.get_lidar_measurements(obs[-1,:2], jnp.atan2(*jnp.flip(obs[-1,2:4])), obs[:-1,:2], info["humans_parameters"][:,0])
    if policy.kinematics == ROBOT_KINEMATICS.index("unicycle"):
        lidar_measurements = env.get_lidar_measurements(obs[-1,:2], obs[-1,4], obs[:-1,:2], info["humans_parameters"][:,0])
    action, _, _ = policy.act(random.PRNGKey(0), obs, info, vnet_params, epsilon=0.)
    state, obs, info, reward, outcome, _ = env.step(
        state,
        info,
        action,test=True,
        reset_if_done=True,
        reset_key=reset_key
    )
    # Save data
    step_data = {
        "actions": action,
        "states": state,
        "rewards": reward,
        "outcomes": outcome,
        "dones": ~outcome["nothing"],
        "lidar_measurements": lidar_measurements[:,0],
    }
    output_data = tree_map(lambda x, y: x.at[i].set(y), output_data, step_data)
    return output_data, state, obs, info, reset_key
# Initialize first episode
state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
# Step loop
output_data, _, _, _, _ = lax.fori_loop(
    0,
    n_steps,
    _simulate_steps_with_lidar,
    (output_data, state, obs, info, reset_key)
)
# Filter last uncompleted episode
last_done_idx = jnp.where(output_data["dones"])[0][-1]
output_data = tree_map(lambda x: x[:last_done_idx+1], output_data)
# Print final stats
print(f"\nDone! {jnp.sum(output_data['dones'])} episodes have been simulated for a total of {len(output_data['dones'])} steps.")
print(f"Success rate: {jnp.sum(output_data['outcomes']['success'])/jnp.sum(output_data['dones']):.2f} - successes: {jnp.sum(output_data['outcomes']['success'])}")
print(f"Failure rate: {jnp.sum(output_data['outcomes']['failure'])/jnp.sum(output_data['dones']):.2f} - failures: {jnp.sum(output_data['outcomes']['failure'])}")
print(f"Timeout rate: {jnp.sum(output_data['outcomes']['timeout'])/jnp.sum(output_data['dones']):.2f} - timeouts: {jnp.sum(output_data['outcomes']['timeout'])}")
# Filter data
# TODO: Control indexing and make this more efficient
if filter_timeouts and jnp.any(output_data["outcomes"]["timeout"]):
    print(f"\nFiltering timeouts...")
    while jnp.any(output_data["outcomes"]["timeout"]):
        last_timeout_idx = jnp.where(output_data["outcomes"]["timeout"])[0][-1]
        last_done_before_timeout_idx = jnp.where(output_data["dones"][:last_timeout_idx])[0][-1]
        indexes_to_filter = jnp.arange(last_done_before_timeout_idx + 1, last_timeout_idx + 1)
        output_data = tree_map(lambda x: jnp.delete(x, indexes_to_filter, axis=0), output_data)
    print(f"\nDone! {jnp.sum(output_data['dones'])} episodes have been simulated for a total of {len(output_data['dones'])} steps.")
    print(f"successes: {jnp.sum(output_data['outcomes']['success'])}")
    print(f"failures: {jnp.sum(output_data['outcomes']['failure'])}")
    print(f"timeouts: {jnp.sum(output_data['outcomes']['timeout'])}")
if filter_failures and jnp.any(output_data["outcomes"]["failure"]):
    print(f"\nFiltering failures...")
    while jnp.any(output_data["outcomes"]["failure"]):
        last_failure_idx = jnp.where(output_data["outcomes"]["failure"])[0][-1]
        last_done_before_failure_idx = jnp.where(output_data["dones"][:last_failure_idx])[0][-1]
        indexes_to_filter = jnp.arange(last_done_before_failure_idx + 1, last_failure_idx + 1)
        output_data = tree_map(lambda x: jnp.delete(x, indexes_to_filter, axis=0), output_data)
    print(f"\nDone! {jnp.sum(output_data['dones'])} episodes have been simulated for a total of {len(output_data['dones'])} steps.")
    print(f"successes: {jnp.sum(output_data['outcomes']['success'])}")
    print(f"failures: {jnp.sum(output_data['outcomes']['failure'])}")
    print(f"timeouts: {jnp.sum(output_data['outcomes']['timeout'])}")