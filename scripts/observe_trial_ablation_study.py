from jax import random, vmap
import jax.numpy as jnp
import numpy as np
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import load_socialjym_policy, animate_trajectory, decimal_to_binary
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

# SETTINGS
scenario = 'parallel_traffic'
n_humans = 25
humans_policy = 'hsfm'
reward_1_decimal_type = 0 # This is the baseline
reward_2_decimal_type = 7 # This is the subject of the analysis
first_policy_params = "sarl_after_RL_hsfm_unicycle_reward_0_hybrid_scenario_10_01_2025.pkl"
second_policy_params = "sarl_after_RL_hsfm_unicycle_reward_7_hybrid_scenario_11_01_2025.pkl"
criterion = "path_length"
maximize = True # If False, it minimizes the criterion
base_seed = 33_000

# Ablation study parameters
test_n_humans = [5,15,25]
n_test_trials = 1000
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
train_scenario = 'hybrid_scenario'
train_hybrid_scenario_subset = jnp.array([1,2,3,4], dtype=jnp.int32) # Exclude normal circular crossing
test_scenarios = ['parallel_traffic', 'perpendicular_traffic', 'robot_crowding', 'delayed_circular_crossing']
scenarios_labels = ["PaT", "PeT", "RC", "DCC"]
# Reward terms parameters
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.035 # High rotation penalty weight
w_bound = 1. # Rotation bound

# Load results
with open(os.path.join(os.path.dirname(__file__), "metrics_after_il_ablation_study.pkl"), "rb") as f:
    all_metrics_after_il = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "metrics_after_rl_ablation_study.pkl"), "rb") as f:
    all_metrics_after_rl = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "training_data_ablation_study.pkl"), "rb") as f:
    training_data = pickle.load(f)

##########################
# all_metrics_after_il_ablation_study is in the form: reward x scenario x n_humans x n_trials
# metrics_after_rl_ablation_study is in the form: reward x scenario x n_humans x n_trials
##########################

### Print episode with highest time to goal for PaT scenario
if maximize:
    episode_1 = jnp.nanargmax(all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:])
else:
    episode_1 = jnp.nanargmin(all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:])
print(
    f"\n## Highest {criterion} on {scenario} scenario for Reward {reward_2_decimal_type}:\n", 
    f"Episode: {episode_1}\n",
    f"{criterion} for baseline reward: {all_metrics_after_rl[criterion][reward_1_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),episode_1]}\n",
    f"{criterion} for subject reward: {all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),episode_1]}\n")

### Print episode with highest difference between full and base reward on time to goal for PaT scenario
if maximize:
    episode_2 = jnp.nanargmax(jnp.abs(
        all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:] - 
        all_metrics_after_rl[criterion][reward_1_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:]))
else:
    episode_2 = jnp.nanargmin(jnp.abs(
        all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:] - 
        all_metrics_after_rl[criterion][reward_1_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),:]))
print(
    f"## Highest difference between Rewards {reward_1_decimal_type} and {reward_2_decimal_type} on {criterion} for {scenario} scenario:\n", 
    f"Episode: {episode_2}\n",
    f"{criterion} for baseline reward: {all_metrics_after_rl[criterion][reward_1_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),episode_2]}\n",
    f"{criterion} for subject reward: {all_metrics_after_rl[criterion][reward_2_decimal_type,test_scenarios.index(scenario),test_n_humans.index(n_humans),episode_2]}\n")
episodes = [episode_1, episode_2]

### Now we replicate the two experiments above to investigate the difference between the two trained policies
## Baseline reward
binary_baseline = decimal_to_binary(reward_1_decimal_type, int(2**len(reward_terms)))
baseline_reward = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = binary_baseline[0],
        time_penalty_reward = binary_baseline[1],
        high_rotation_penalty_reward = binary_baseline[2],
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
        time_penalty=wt,
        angular_speed_bound=w_bound,
        angular_speed_penalty_weight=wr
    )
# Subject reward
binary_subject = decimal_to_binary(reward_2_decimal_type, int(2**len(reward_terms)))
subject_reward = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = binary_subject[0],
        time_penalty_reward = binary_subject[1],
        high_rotation_penalty_reward = binary_subject[2],
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
        time_penalty=wt,
        angular_speed_bound=w_bound,
        angular_speed_penalty_weight=wr
    )
rewards = [baseline_reward, subject_reward]
# Episode loop
for j, episode in enumerate(episodes):
    for i, reward in enumerate(rewards):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': scenario,
            'hybrid_scenario_subset': jnp.array([1,2,3,4], dtype=jnp.int32),
            'humans_policy': humans_policy,
            'circle_radius': 7,
            'reward_function': reward,
            'kinematics': 'unicycle',
        }
        test_env = SocialNav(**test_env_params)
        # Initialize robot policy
        policy = SARL(test_env.reward_function, dt=0.25, kinematics='unicycle')
        # Load policy parameters
        if bool(i):
            vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/", second_policy_params))
        else:
            vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/", first_policy_params))
        print(f"## Episode {j} with Reward {i} ##")
        policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + base_seed + episode) # We don't care if we generate two identical keys, they operate differently
        state, reset_key, obs, info, outcome = test_env.reset(reset_key)
        all_states = np.array([state])
        while outcome["nothing"]:
            action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
            state, obs, info, reward, outcome = test_env.step(state,info,action,test=True)
            all_states = np.vstack((all_states, [state]))
        ## Animate trajectory
        animate_trajectory(
            all_states, 
            info['humans_parameters'][:,0], 
            test_env.robot_radius, 
            test_env_params['humans_policy'],
            info['robot_goal'],
            info['current_scenario'],
            robot_dt=test_env_params['robot_dt'],
            kinematics='unicycle')

