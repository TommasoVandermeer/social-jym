from jax import random, vmap, lax
import jax.numpy as jnp

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as Reward
from socialjym.policies.dwa import DWA

# Hyperparameters
random_seed = 0
n_episodes = 100
kinematics = 'unicycle'
lidar_n_stack_to_use = 1
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': jnp.pi * 2,
    'lidar_max_dist': 10.,
    'n_humans': 5,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,      
    'robot_visible': True,
    'scenario': 'parallel_traffic', 
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic
    'ccso_n_static_humans': 0,
    'reward_function': Reward(robot_radius=0.3),
    'kinematics': kinematics,
    'lidar_noise': False,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = DWA(
    lidar_num_rays=env.lidar_num_rays,
    lidar_angular_range=env.lidar_angular_range,
    lidar_max_dist=env.lidar_max_dist,
    n_stack=env.n_stack,
    lidar_n_stack_to_use=lidar_n_stack_to_use,
    predict_time_horizon=2,
    actions_discretization=16,
)

# # Execute tests
# metrics = policy.evaluate(
#     n_episodes,
#     random_seed,
#     env,
# )

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    step = 0
    max_steps = int(env.reward_function.time_limit/env.robot_dt)+1
    all_states = jnp.array([state])
    all_observations = jnp.array([obs])
    all_robot_goals = jnp.array([info['robot_goal']])
    all_static_obstacles = jnp.array([info['static_obstacles'][-1]])
    all_humans_radii = jnp.array([info['humans_parameters'][:,0]])
    all_actions = jnp.zeros((max_steps, 2))
    all_actions_costs = jnp.zeros((max_steps,len(policy.action_space)))
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, cost, actions_cost = policy.act(obs, info)
        # print(f"Episode {i}, Step {step}, Action: {action}, Cost: {cost}")
        # Step the environment
        state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)
        # Save data for animation
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_observations = jnp.vstack((all_observations, jnp.array([obs])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
        all_static_obstacles = jnp.vstack((all_static_obstacles, jnp.array([info['static_obstacles'][-1]])))
        all_humans_radii = jnp.vstack((all_humans_radii, jnp.array([info['humans_parameters'][:,0]])))
        all_actions = all_actions.at[step].set(action)
        all_actions_costs = all_actions_costs.at[step].set(actions_cost)
        # Increment step
        step += 1
    all_actions = all_actions[:step]
    all_actions_costs = all_actions_costs[:step]
    print("\nOutcome: ", [k for k, v in outcome.items() if v][0])
    policy.animate_lasernav_trajectory(
        all_states[:-1],
        all_observations[:-1],
        all_actions,
        all_actions_costs,
        all_robot_goals[:-1],
        all_static_obstacles[:-1],
        all_humans_radii[:-1],
        env,
    )