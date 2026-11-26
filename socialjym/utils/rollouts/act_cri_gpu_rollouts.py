import optax
import pickle
from functools import partial
from jax import jit, lax, random, vmap, default_backend
from jax.tree_util import tree_map
import jax.numpy as jnp
from tqdm import tqdm

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.base_act_cri_buffer import BaseACBuffer


def _select_envs(mask, on_true, on_false):
    # Per-env selection between two pytrees whose leaves have the env axis first
    return tree_map(lambda t, f: jnp.where(jnp.reshape(mask, (-1,) + (1,) * (t.ndim - 1)), t, f), on_true, on_false)

@partial(jit, static_argnames=("policy", "env", "max_steps"))
def simulate_il_episodes(
    states,
    obses,
    infos,
    outcomes,
    policy:BasePolicy,
    env:BaseEnv,
    max_steps:int,
):
    """
    Runs one imitation learning episode per env in parallel, until all of them are over. Finished envs are frozen.
    Returns the time-major history (inputs, rewards, actions), the length of each episode, and the final outcomes.
    """
    n_envs = states.shape[0]
    def _cond(carry):
        t, _, _, _, outcomes, _, _ = carry
        return (t < max_steps) & jnp.any(outcomes["nothing"])
    def _body(carry):
        t, states, obses, infos, outcomes, lengths, history = carry
        active = outcomes["nothing"]
        inputs = vmap(policy.batch_compute_vnet_input, in_axes=(0,0,0))(obses[:,-1], obses[:,:-1], infos)
        new_states, new_obses, new_infos, (rewards, _), new_outcomes = vmap(env.imitation_learning_step)(states, infos)
        history = {
            "inputs": history["inputs"].at[t].set(inputs),
            "rewards": history["rewards"].at[t].set(rewards),
            "actions": history["actions"].at[t].set(new_obses[:,-1,2:4]),
        }
        states, obses, infos, outcomes = _select_envs(active, (new_states, new_obses, new_infos, new_outcomes), (states, obses, infos, outcomes))
        return t + 1, states, obses, infos, outcomes, lengths + active, history
    history = {
        "inputs": jnp.zeros((max_steps, n_envs, env.n_humans, policy.vnet_input_size)),
        "rewards": jnp.zeros((max_steps, n_envs)),
        "actions": jnp.zeros((max_steps, n_envs, 2)),
    }
    init = (0, states, obses, infos, outcomes, jnp.zeros((n_envs,), dtype=jnp.int32), history)
    _, _, _, _, outcomes, lengths, history = lax.while_loop(_cond, _body, init)
    return history, lengths, outcomes

@partial(jit, static_argnames=("policy",))
def compute_il_targets(
    rewards,
    lengths,
    policy:BasePolicy,
):
    """
    Discounted return from each step k of each episode: sum_{j=k}^{L-1} gamma^((j-k) dt v_max) r_j.
    The terms are summed in the same order as act_cri_rollouts.py. Rewards are time-major, output is (k, env).
    """
    max_steps, n_envs = rewards.shape
    ks = jnp.arange(max_steps)
    def _body(targets, j):
        terms = jnp.power(policy.reward_function.gamma, (j - ks)[:,None] * policy.dt * policy.v_max) * rewards[j][None,:]
        mask = (j >= ks)[:,None] & (j < lengths)[None,:]
        return targets + jnp.where(mask, terms, 0.), None
    targets, _ = lax.scan(_body, jnp.zeros((max_steps, n_envs)), jnp.arange(max_steps))
    return targets

@partial(jit, static_argnames=("replay_buffer",))
def add_il_experiences(
    buffer_state,
    buffer_size,
    history,
    targets,
    lengths,
    success,
    replay_buffer:BaseACBuffer,
):
    """
    Adds the experiences of successful episodes to the ring buffer, in the same order as act_cri_rollouts.py
    (episode by episode, step by step). Only the experiences that are not overwritten within this call are written,
    so that the scatter has no duplicate indices.
    """
    max_steps, n_envs = targets.shape
    capacity = replay_buffer.buffer_size
    keep = (success[:,None] & (jnp.arange(max_steps)[None,:] < lengths[:,None])).reshape(-1).astype(jnp.int32)
    positions = buffer_size + jnp.cumsum(keep) - keep
    n_added = jnp.sum(keep)
    write = (keep == 1) & (positions >= buffer_size + n_added - capacity)
    idxs = jnp.where(write, positions % capacity, capacity) # Out of bounds indices are dropped
    def env_major(x): # (max_steps, n_envs, ...) -> (n_envs * max_steps, ...)
        return jnp.reshape(jnp.swapaxes(x, 0, 1), (n_envs * max_steps, *x.shape[2:]))
    buffer_state = {
        **buffer_state,
        "inputs": buffer_state["inputs"].at[idxs].set(env_major(history["inputs"]), mode="drop"),
        "critic_targets": buffer_state["critic_targets"].at[idxs].set(env_major(targets), mode="drop"),
        "sample_actions": buffer_state["sample_actions"].at[idxs].set(env_major(history["actions"]), mode="drop"),
    }
    return buffer_state, buffer_size + n_added

@partial(jit, static_argnames=("policy", "env", "replay_buffer", "max_steps", "stored_steps"))
def collect_il_chunk(
    episode_ids,
    n_episodes,
    random_seed,
    buffer_state,
    buffer_size,
    custom_data,
    policy:BasePolicy,
    env:BaseEnv,
    replay_buffer:BaseACBuffer,
    max_steps:int,
    stored_steps:int,
):
    """
    Simulates the IL episodes with the given ids in parallel and adds the successful ones to the buffer.
    Ids >= n_episodes are padding (last chunk) and are discarded. Returns the updated buffer and the episodes returns.
    """
    keys = vmap(random.PRNGKey)(random_seed + episode_ids)
    if custom_data is None:
        states, _, obses, infos, outcomes = env.batch_reset(keys)
    else:
        episodes = tree_map(lambda x: x[jnp.minimum(episode_ids, n_episodes - 1)], custom_data)
        states, _, obses, infos, outcomes = vmap(env.reset_custom_episode)(keys, episodes)
    # For imitation learning, set the humans' safety space to 0.1
    infos["humans_parameters"] = infos["humans_parameters"].at[:,:,-1].set(0.1)
    states, obses, infos, outcomes = tree_map(lambda x: lax.convert_element_type(x, x.dtype), (states, obses, infos, outcomes))
    history, lengths, outcomes = simulate_il_episodes(states, obses, infos, outcomes, policy, env, max_steps)
    # act_cri_rollouts.py stores at most stored_steps steps per episode: data of later steps (an episode can last one
    # more step, the one where the timeout is detected) is dropped, and reading it returns the last stored step.
    history = tree_map(lambda x: x.at[stored_steps:].set(x[stored_steps-1]), history)
    targets = compute_il_targets(history["rewards"], lengths, policy)
    buffer_state, buffer_size = add_il_experiences(
        buffer_state, buffer_size, history, targets, lengths, outcomes["success"] & (episode_ids < n_episodes), replay_buffer
    )
    return buffer_state, buffer_size, targets[0]

@partial(jit, static_argnames=("policy", "actor_optimizer", "critic_optimizer", "replay_buffer", "optimization_steps", "shuffle"))
def train_il_epoch(
    buffer_key,
    buffer_state,
    buffer_size,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    policy:BasePolicy,
    actor_optimizer:optax.GradientTransformation,
    critic_optimizer:optax.GradientTransformation,
    replay_buffer:BaseACBuffer,
    optimization_steps:int,
    shuffle:bool,
):
    # Shuffle the buffer if this is full (otherwise it is not possible without making a mess)
    if shuffle:
        buffer_state, buffer_key = replay_buffer.shuffle(buffer_state, buffer_key)
    def _batch_step(carry, k):
        actor_params, critic_params, actor_opt_state, critic_opt_state = carry
        experiences = replay_buffer.iterate(buffer_state, buffer_size, k)
        critic_params, actor_params, critic_opt_state, actor_opt_state, critic_loss, actor_loss, _ = policy.update_il(
            critic_params,
            actor_params,
            actor_optimizer,
            actor_opt_state,
            critic_optimizer,
            critic_opt_state,
            experiences,
        )
        return (actor_params, critic_params, actor_opt_state, critic_opt_state), (actor_loss, critic_loss)
    (actor_params, critic_params, actor_opt_state, critic_opt_state), (actor_losses, critic_losses) = lax.scan(
        _batch_step,
        (actor_params, critic_params, actor_opt_state, critic_opt_state),
        jnp.arange(optimization_steps),
    )
    return buffer_key, buffer_state, actor_params, critic_params, actor_opt_state, critic_opt_state, actor_losses, critic_losses

def actor_critic_il_rollout(
        initial_actor_params: dict,
        initial_critic_params: dict,
        train_episodes: int,
        random_seed: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        buffer_state: dict,
        current_buffer_size: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseACBuffer,
        buffer_capacity: int,
        num_epochs: int,
        batch_size: int,
        custom_episodes: str = None,
        n_parallel_envs: int = 1000,
) -> dict:
    """
    Parallel GPU version of act_cri_rollouts.actor_critic_il_rollout (same inputs and outputs, plus n_parallel_envs).
    Episodes are simulated n_parallel_envs at a time instead of one by one. Episode i is reset with the same key as
    in act_cri_rollouts.py, so the collected experiences, returns, and buffer are the same.
    """
    policies = ["SARL-PPO", "DIRSAFE"]
    assert policy.name in policies, "This function is only compatible with the Actor-Critic policies."
    if default_backend() != "gpu":
        print(f"WARNING: JAX default backend is '{default_backend()}', IL will not run on GPU.")

    # Load custom episodes if provided
    if custom_episodes is not None:
        with open(custom_episodes, 'rb') as f:
            custom_episode_data = pickle.load(f)
        train_episodes = len(custom_episode_data)
        print(f"Custom episodes loaded: {train_episodes}")
        custom_data = {
            "full_state": jnp.array([custom_episode_data[i]["full_state"] for i in range(train_episodes)]),
            "robot_goal": jnp.array([custom_episode_data[i]["robot_goal"] for i in range(train_episodes)]),
            "humans_goal": jnp.array([custom_episode_data[i]["humans_goal"] for i in range(train_episodes)]),
            "static_obstacles": jnp.array([custom_episode_data[i]["static_obstacles"] for i in range(train_episodes)]),
            "scenario": jnp.array([custom_episode_data[i]["scenario"] for i in range(train_episodes)]),
            "humans_radius": jnp.array([custom_episode_data[i]["humans_radius"] for i in range(train_episodes)]),
            "humans_speed": jnp.array([custom_episode_data[i]["humans_speed"] for i in range(train_episodes)]),
            "humans_delay": jnp.zeros((train_episodes, env.n_humans)),
        }
    else:
        custom_data = None

    # Simulate the episodes in parallel chunks
    n_parallel_envs = min(n_parallel_envs, train_episodes)
    n_chunks = -(-train_episodes // n_parallel_envs)
    stored_steps = int(env.reward_function.time_limit/env.robot_dt)
    max_steps = stored_steps + 3 # Upper bound of the episode length (the timeout is detected after stored_steps steps)
    buffer_size = jnp.asarray(current_buffer_size, dtype=jnp.int32)
    chunk_returns = []
    print(f"Simulating {train_episodes} IL episodes, {n_parallel_envs} in parallel...")
    for chunk in tqdm(range(n_chunks)):
        episode_ids = jnp.arange(chunk * n_parallel_envs, (chunk + 1) * n_parallel_envs, dtype=jnp.int32)
        buffer_state, buffer_size, returns = collect_il_chunk(
            episode_ids, train_episodes, random_seed, buffer_state, buffer_size, custom_data,
            policy, env, replay_buffer, max_steps, stored_steps,
        )
        buffer_size.block_until_ready() # SYNC (keeps the progress bar honest)
        chunk_returns.append(returns)
    returns = jnp.concatenate(chunk_returns)[:train_episodes]
    current_buffer_size = buffer_size
    actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_capacity]))

    # Normalizing networks inputs (if necessary)
    if policy.normalize_and_clip_obs:
        norm_state = initial_actor_params["norm_state"]
        inputs = buffer_state["inputs"][:actual_buffer_size].reshape((actual_buffer_size * env.n_humans, policy.vnet_input_size))
        updated_norm_state = policy.update_norm_state(
            inputs,
            norm_state,
        )
        normalized_inputs = policy.normalize_and_clip_inputs(
            inputs,
            updated_norm_state,
        )
        normalized_inputs = normalized_inputs.reshape((actual_buffer_size, env.n_humans, policy.vnet_input_size))
        buffer_state["inputs"] = buffer_state["inputs"].at[:actual_buffer_size].set(normalized_inputs)
        initial_actor_params["norm_state"] = updated_norm_state

    # Initialize the optimizer
    if policy.normalize_and_clip_obs:
        actor_optimizer_state = actor_optimizer.init(initial_actor_params["actor_params"])
    else:
        actor_optimizer_state = actor_optimizer.init(initial_actor_params)
    critic_optimizer_state = critic_optimizer.init(initial_critic_params)
    # Initialize the buffer key
    buffer_key = random.PRNGKey(0) # We do not care to control the buffer shuffle

    # Update model parameters
    print(f"Buffer size after IL: {int(actual_buffer_size)}")
    optimization_steps = int(actual_buffer_size) // batch_size
    buffer_full = bool(actual_buffer_size == buffer_capacity)
    if buffer_full:
        print("Pre-shuffling the buffer...")
        buffer_state, buffer_key = replay_buffer.shuffle(buffer_state, buffer_key, 100)
    actor_params = initial_actor_params["actor_params"] if policy.normalize_and_clip_obs else initial_actor_params
    critic_params = initial_critic_params
    actor_params, critic_params, actor_optimizer_state, critic_optimizer_state = tree_map(
        lambda x: lax.convert_element_type(jnp.asarray(x), jnp.asarray(x).dtype),
        (actor_params, critic_params, actor_optimizer_state, critic_optimizer_state),
    )
    act_losses, cri_losses = [jnp.zeros((0, optimization_steps))], [jnp.zeros((0, optimization_steps))]
    print(f"Optimizing model on generated experiences for {num_epochs} epochs...")
    for _ in tqdm(range(num_epochs)):
        buffer_key, buffer_state, actor_params, critic_params, actor_optimizer_state, critic_optimizer_state, epoch_act_losses, epoch_cri_losses = train_il_epoch(
            buffer_key, buffer_state, actual_buffer_size, actor_params, critic_params, actor_optimizer_state, critic_optimizer_state,
            policy, actor_optimizer, critic_optimizer, replay_buffer, optimization_steps, buffer_full,
        )
        epoch_act_losses.block_until_ready() # SYNC (keeps the progress bar honest)
        act_losses.append(epoch_act_losses[None,:])
        cri_losses.append(epoch_cri_losses[None,:])
    act_losses, cri_losses = jnp.concatenate(act_losses), jnp.concatenate(cri_losses)
    output_dict = {
        "actor_params": {"actor_params": actor_params, "norm_state": initial_actor_params["norm_state"]} if policy.normalize_and_clip_obs else actor_params,
        "critic_params": critic_params,
        "actor_optimizer_state": actor_optimizer_state,
        "critic_optimizer_state": critic_optimizer_state,
        "buffer_state": buffer_state,
        "current_buffer_size": current_buffer_size,
        "buffer_key": buffer_key,
        "actor_losses": jnp.mean(act_losses, axis=1),
        "critic_losses": jnp.mean(cri_losses, axis=1),
        "returns": returns}
    return output_dict
