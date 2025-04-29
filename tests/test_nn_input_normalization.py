import haiku as hk
from jax import random, vmap, jit
import jax.numpy as jnp
from jax.tree_util import tree_map

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory

# Define a simple function with BatchNorm
def forward_fn(x, update_stats):
    bn = hk.BatchNorm(create_scale=False, create_offset=False, decay_rate=0.9, eps=1e-3)
    return bn(x, update_stats)
# Transform the function into a Haiku module
model = hk.transform_with_state(forward_fn)

# Hyperparameters
random_seed = 1
n_steps = 2_000
clip_obs_bound = 5.
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
    'n_humans': 6,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'circular_crossing_with_static_obstacles',
    'hybrid_scenario_subset': jnp.array([0,1], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}
env = SocialNav(**env_params)
policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
initial_vnet_params = policy.model.init(random.key(random_seed), jnp.zeros((env_params["n_humans"],policy.vnet_input_size)))

# Episode
policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed) # We don't care if we generate two identical keys, they operate differently
state, reset_key, obs, info, outcome = env.reset(reset_key)
vnet_input = policy.batch_compute_vnet_input(obs[-1], obs[:-1], info)
norm_params, norm_state = model.init(random.PRNGKey(0), vnet_input, update_stats=True)
all_normalized_inputs = jnp.zeros((n_steps,)+vnet_input.shape)
all_inputs = jnp.zeros((n_steps,)+vnet_input.shape)
for step in range(n_steps):
    action, policy_key, vnet_input = policy.act(policy_key, obs, info, initial_vnet_params, 0.)
    state, obs, info, reward, outcome, reset_key = env.step(state,info,action,test=True,reset_if_done=True,reset_key=reset_key)
    # Apply the model (normalize the input)
    norm_vnet_input, norm_state = model.apply(
        norm_params,
        norm_state,
        None,
        vnet_input,
        True,  # update_stats=False
    )
    clipped_norm_vnet_input = jnp.clip(norm_vnet_input, -clip_obs_bound, clip_obs_bound)
    # Save data
    all_normalized_inputs = all_normalized_inputs.at[step].set(clipped_norm_vnet_input)
    all_inputs = all_inputs.at[step].set(vnet_input)

# Print results
print("Initial network state:")
print(norm_state)
print("Original initial Input:")
print(all_inputs[0])
print("Normalized initial Input:")
print(all_normalized_inputs[0])
print("Final network state:")
print(norm_state)
print("Original final Input:")
print(all_inputs[-1])
print("Normalized final Input:")
print(all_normalized_inputs[-1])

print("\nOriginal inputs stats: ")
print(f"infs: {jnp.isinf(all_inputs).sum()} - nans: {jnp.isnan(all_inputs).sum()} - max: {jnp.max(all_inputs)} - min: {jnp.min(all_inputs)} - mean: {jnp.mean(all_inputs)} - std: {jnp.std(all_inputs)}")
print("Normalized inputs stats: ")
print(f"infs: {jnp.isinf(all_normalized_inputs).sum()} - nans: {jnp.isnan(all_normalized_inputs).sum()} - max: {jnp.max(all_normalized_inputs)} - min: {jnp.min(all_normalized_inputs)} - mean: {jnp.mean(all_normalized_inputs)} - std: {jnp.std(all_normalized_inputs)}")
