from jax import jit, random, debug, lax, vmap
from jax_tqdm import loop_tqdm
from jax.tree_util import tree_map
import jax.numpy as jnp
import pickle
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import ROBOT_KINEMATICS, wrap_angle
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import load_socialjym_policy, animate_trajectory

### Hyperparameters
## Simulation parameters
random_seed = 0
filter_uncompleted_episodes = True
filter_timeouts = True
filter_failures = True 
n_steps_per_setting = 100_000 # Approximately 100 steps = 1 episode.
## Testing settings parameters
scenario = "hybrid_scenario"
n_humans = [3,5,10,15]
humans_policy = ["sfm","hsfm"]
## Robot parameters
robot_radius = 0.3
robot_dt = 0.25
robot_visible = False
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 60
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


### Create output data dir
if not os.path.exists(os.path.join(os.path.dirname(__file__),"data")):
    os.makedirs(os.path.join(os.path.dirname(__file__),"data"))
### Initialize robot policy
vnet_params = load_socialjym_policy(
    os.path.join(
        os.path.expanduser("~"),
        "Repos/social-jym/trained_policies/socialjym_policies/sarl_k1_nh5_hp2_s4_r1_20_11_2024.pkl"
    )
)
policy = SARL(reward_function, dt=robot_dt, kinematics=kinematics)
### Initialize output data dictionary
total_n_steps = n_steps_per_setting * len(n_humans) * len(humans_policy)
output_data = {
    "actions": jnp.zeros((total_n_steps,2)),
    "rewards": jnp.zeros((total_n_steps,)),
    "outcomes": {
        "nothing": jnp.zeros((total_n_steps,), dtype=bool),
        "failure": jnp.zeros((total_n_steps,), dtype=bool),
        "success": jnp.zeros((total_n_steps,), dtype=bool),
        "timeout": jnp.zeros((total_n_steps,), dtype=bool),
    },
    "dones": jnp.zeros((total_n_steps,), dtype=bool),
    "lidar_measurements": jnp.zeros((total_n_steps,lidar_num_rays)),
    "robot_goals": jnp.zeros((total_n_steps,2)),
    "robot_positions": jnp.zeros((total_n_steps,2)),
    "robot_orientations": jnp.zeros((total_n_steps,)),
}
### Simulate n_steps_per_setting steps for each setting
global_idx = 0
for h, h_policy in enumerate(humans_policy):
    for n, n_h in enumerate(n_humans):
        ## Initialize and reset environment
        env_params = {
            'robot_radius': 0.3,
            'n_humans': n_h,
            'robot_dt': robot_dt,
            'robot_radius': robot_radius, 
            'humans_dt': 0.01,
            'robot_visible': robot_visible,
            'scenario': scenario,
            'humans_policy': h_policy,
            'reward_function': reward_function,
            'kinematics': kinematics,
            'lidar_angular_range':lidar_angular_range,
            'lidar_max_dist':lidar_max_dist,
            'lidar_num_rays':lidar_num_rays,
        }
        env = SocialNav(**env_params)
        print("\n################")
        print(f"Simulating for {n_steps_per_setting} steps with {n_h} {h_policy}-driven humans...")
        @loop_tqdm(n_steps_per_setting)
        @jit
        def _simulate_steps_with_lidar(i:int, for_val:tuple):
            ## Retrieve data from the tuple
            setting_data, aux_data, state, obs, info, reset_key = for_val
            ## Simulate one step
            if policy.kinematics == ROBOT_KINEMATICS.index("holonomic"):
                robot_orientation = jnp.arctan2(*jnp.flip(obs[-1,2:4]))
                lidar_measurements = env.get_lidar_measurements(obs[-1,:2], jnp.atan2(*jnp.flip(obs[-1,2:4])), obs[:-1,:2], info["humans_parameters"][:,0])
            elif policy.kinematics == ROBOT_KINEMATICS.index("unicycle"):
                robot_orientation = obs[-1,5]
            lidar_measurements = env.get_lidar_measurements(obs[-1,:2], robot_orientation, obs[:-1,:2], info["humans_parameters"][:,0])
            action, _, _ = policy.act(random.PRNGKey(0), obs, info, vnet_params, epsilon=0.)
            final_state, final_obs, final_info, reward, outcome, final_reset_key = env.step(
                state,
                info,
                action,
                test=False,
                reset_if_done=True,
                reset_key=reset_key
            )
            # Save output data
            step_out_data = {
                "actions": action,
                "rewards": reward,
                "outcomes": outcome,
                "dones": ~outcome["nothing"],
                "lidar_measurements": lidar_measurements[:,0],
                "robot_goals": info["robot_goal"],
                "robot_positions": obs[-1,:2],
                "robot_orientations": robot_orientation,
            }
            setting_data = tree_map(lambda x, y: x.at[i].set(y), setting_data, step_out_data)
            # Save aux data
            step_aux_data = {
                "states": state,
                "lidar_measurements": lidar_measurements,
            }
            aux_data = tree_map(lambda x, y: x.at[i].set(y), aux_data, step_aux_data)
            return setting_data, aux_data, final_state, final_obs, final_info, final_reset_key
        # Initialize first episode
        state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed + global_idx))
        # Initialize setting data
        setting_data = {
            "actions": jnp.zeros((n_steps_per_setting,2)),
            "rewards": jnp.zeros((n_steps_per_setting,)),
            "outcomes": {
                "nothing": jnp.zeros((n_steps_per_setting,), dtype=bool),
                "failure": jnp.zeros((n_steps_per_setting,), dtype=bool),
                "success": jnp.zeros((n_steps_per_setting,), dtype=bool),
                "timeout": jnp.zeros((n_steps_per_setting,), dtype=bool),
            },
            "dones": jnp.zeros((n_steps_per_setting,), dtype=bool),
            "lidar_measurements": jnp.zeros((n_steps_per_setting,lidar_num_rays)),
            "robot_goals": jnp.zeros((n_steps_per_setting,2)),
            "robot_positions": jnp.zeros((n_steps_per_setting,2)),
            "robot_orientations": jnp.zeros((n_steps_per_setting,)),
        }
        # Initialize auxiliary data
        aux_data = {
            "states": jnp.zeros((n_steps_per_setting,env_params["n_humans"]+1,6)),
            "lidar_measurements": jnp.zeros((n_steps_per_setting,env.lidar_num_rays,2)),
        }
        # Step loop
        setting_data, aux_data, _, _, _, _ = lax.fori_loop(
            0,
            n_steps_per_setting,
            _simulate_steps_with_lidar,
            (setting_data, aux_data, state, obs, info, reset_key)
        )
        # Print pre-filtered final stats
        print(f"Done! {jnp.sum(setting_data['dones'])} episodes have been simulated for a total of {len(setting_data['dones'])} steps.")
        print(f"Success rate: {jnp.sum(setting_data['outcomes']['success'])/jnp.sum(setting_data['dones']):.2f} - successes: {jnp.sum(setting_data['outcomes']['success'])}")
        print(f"Failure rate: {jnp.sum(setting_data['outcomes']['failure'])/jnp.sum(setting_data['dones']):.2f} - failures: {jnp.sum(setting_data['outcomes']['failure'])}")
        print(f"Timeout rate: {jnp.sum(setting_data['outcomes']['timeout'])/jnp.sum(setting_data['dones']):.2f} - timeouts: {jnp.sum(setting_data['outcomes']['timeout'])}")
        # Filter data
        if filter_uncompleted_episodes:
            print(f"\nFiltering uncompleted episode...")
            if not jnp.any(setting_data["dones"]):
                print(f"\nWARNING: there are no completed episodes in the simulation. And you want to filter uncompleted episodes, resulting in empty data.\nInterrupting...")
                exit()
            last_done_idx = jnp.where(setting_data["dones"])[0][-1]
            setting_data = tree_map(lambda x: x[:last_done_idx+1], setting_data)
        if filter_failures and jnp.any(setting_data["outcomes"]["failure"]):
            print(f"\nFiltering failures...")
            failure_idxs = jnp.where(setting_data["outcomes"]["failure"])[0]
            mask = jnp.ones((len(setting_data["dones"]),), dtype=bool)
            for failure_idx in failure_idxs:
                dones_before_failure = jnp.where(setting_data["dones"][:failure_idx])[0]
                last_done_before_failure_idx = dones_before_failure[-1] if dones_before_failure.shape[0] != 0 else -1 # The first episode might have failed, meaning there are no dones before the failure
                mask = mask.at[last_done_before_failure_idx + 1:failure_idx + 1].set(False)
            setting_data = tree_map(lambda x: x[mask], setting_data)
            print(f"Done!\n{jnp.sum(setting_data['dones'])} episodes remaining for a total of {len(setting_data['dones'])} steps.")
            print(f"successes: {jnp.sum(setting_data['outcomes']['success'])}")
            print(f"failures: {jnp.sum(setting_data['outcomes']['failure'])}")
            print(f"timeouts: {jnp.sum(setting_data['outcomes']['timeout'])}")
        if filter_timeouts and filter_failures and (not jnp.any(setting_data["outcomes"]["success"])):
            print(f"\nWARNING: there are no successful episodes in the simulation. And you want to filter timeouts and failures, resulting in empty data.\nInterrupting...")
            exit()
        if filter_timeouts and jnp.any(setting_data["outcomes"]["timeout"]):
            print(f"\nFiltering timeouts...")
            timeout_idxs = jnp.where(setting_data["outcomes"]["timeout"])[0]
            mask = jnp.ones((len(setting_data["dones"]),), dtype=bool)
            for timeout_idx in timeout_idxs:
                dones_before_timeout = jnp.where(setting_data["dones"][:timeout_idx])[0]
                last_done_before_timeout_idx = dones_before_timeout[-1] if dones_before_timeout.shape[0] != 0 else -1 # The first episode might have timed out, meaning there are no dones before the timeout
                mask = mask.at[last_done_before_timeout_idx + 1:timeout_idx + 1].set(False)
            setting_data = tree_map(lambda x: x[mask], setting_data)
            print(f"Done!\n{jnp.sum(setting_data['dones'])} episodes remaining for a total of {len(setting_data['dones'])} steps.")
            print(f"successes: {jnp.sum(setting_data['outcomes']['success'])}")
            print(f"failures: {jnp.sum(setting_data['outcomes']['failure'])}")
            print(f"timeouts: {jnp.sum(setting_data['outcomes']['timeout'])}")
        # Save setting data in output data
        output_data = tree_map(lambda x, y: x.at[global_idx:global_idx+len(setting_data["dones"])].set(y), output_data, setting_data)
        ## Create animation of first episode (just for debugging)
        # animate_trajectory(
        #     aux_data["states"][:jnp.where(setting_data["dones"])[0][0]+1],
        #     info['humans_parameters'][:,0],
        #     env.robot_radius,
        #     env_params['humans_policy'],
        #     info['robot_goal'],
        #     info['current_scenario'],
        #     robot_dt=env_params['robot_dt'],
        #     lidar_measurements=aux_data["lidar_measurements"][:jnp.where(setting_data["dones"])[0][0]+1],
        #     kinematics=kinematics
        # )
        # Increment global index
        global_idx += len(setting_data["dones"])
### Filter empty entries
output_data = tree_map(lambda x: x[:global_idx], output_data)
print("\n################")
print(f"All settings completed!\n{jnp.sum(output_data['dones'])} episodes have been simulated for a total of {len(output_data['dones'])} steps.")
print(f"Success rate: {jnp.sum(output_data['outcomes']['success'])/jnp.sum(output_data['dones']):.2f} - successes: {jnp.sum(output_data['outcomes']['success'])}")
print(f"Failure rate: {jnp.sum(output_data['outcomes']['failure'])/jnp.sum(output_data['dones']):.2f} - failures: {jnp.sum(output_data['outcomes']['failure'])}")
print(f"Timeout rate: {jnp.sum(output_data['outcomes']['timeout'])/jnp.sum(output_data['dones']):.2f} - timeouts: {jnp.sum(output_data['outcomes']['timeout'])}")
### Save and load filtered data
with open(os.path.join(os.path.dirname(__file__),"data","filtered_data.pkl"), "wb") as f:
    pickle.dump(output_data, f)
with open(os.path.join(os.path.dirname(__file__),"data","filtered_data.pkl"), "rb") as f:
    output_data = pickle.load(f)
### Process data for LaserNav env experience standard
n_experiences = len(output_data["actions"])
experience_data = {
    "actions": output_data["actions"],
    "returns": jnp.zeros((n_experiences,)),
    "inputs": jnp.zeros((n_experiences, lidar_num_rays + 5)), # [d_goal,robot_radius,theta,vx,vy,lidar_measurements] It is robot-centric and goal-oriented
}
## Save inputs
@jit
def _compute_input(lidar_measurements, robot_position, robot_goal, robot_orientation, action):
    # Compute vx, vy and theta
    if policy.kinematics == ROBOT_KINEMATICS.index("holonomic"):
        vx, vy = action
        theta = 0.
    elif policy.kinematics == ROBOT_KINEMATICS.index("unicycle"):
        vx, vy = action[0] * jnp.cos(robot_orientation), action[0] * jnp.sin(robot_orientation)
        theta = wrap_angle(robot_orientation - jnp.atan2(*jnp.flip(robot_goal - robot_position)))
    return jnp.array([jnp.linalg.norm(robot_goal - robot_position), robot_radius, theta, vx, vy, *lidar_measurements])
experience_data["inputs"] = experience_data["inputs"].at[:].set(vmap(_compute_input, in_axes=(0,0,0,0,0))(
    output_data["lidar_measurements"],
    output_data["robot_positions"],
    output_data["robot_goals"],
    output_data["robot_orientations"],
    output_data["actions"],
))
## Compute returns
# First create the beginnings array indicating when each episode starts
beginnings = jnp.insert(output_data["dones"], 0, True, axis=0)
print("\nComputing returns...")
@jit
def _compute_returns(t:int, val:tuple):
    rt = n_experiences - t - 1
    returns, rewards, beginnings = val
    returns = returns.at[rt].set(rewards[rt] + pow(policy.gamma, policy.dt * policy.v_max) * returns[rt+1] * (1 - beginnings[rt+1]))
    return returns, rewards, beginnings
experience_data["returns"], _, _ = lax.fori_loop(
    0,
    n_experiences,
    _compute_returns,
    (experience_data["returns"], output_data["rewards"], beginnings),
)
print("Done!")
print("Rewards at dones: (for debugging purposes) \n", experience_data["returns"][jnp.where(output_data["dones"])[0]]) # Printed for debugging purposes
### Save processed data
with open(os.path.join(os.path.dirname(__file__),"data","processed_data.pkl"), "wb") as f:
    pickle.dump(experience_data, f)