import haiku as hk
import optax
import pickle
from typing import Callable
from functools import partial
from jax import jit, lax, random, vmap, default_backend
from jax.tree_util import tree_map
import jax.numpy as jnp
from tqdm import tqdm

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.base_vnet_replay_buffer import BaseVNetReplayBuffer
from socialjym.utils.rollouts.act_cri_gpu_rollouts import simulate_il_episodes, compute_il_targets


def _select_envs(mask, on_true, on_false):
    # Per-env selection between two pytrees whose leaves have the env axis first
    return tree_map(lambda t, f: jnp.where(jnp.reshape(mask, (-1,) + (1,) * (t.ndim - 1)), t, f), on_true, on_false)

def _strip_weak_types(tree):
    return tree_map(lambda x: lax.convert_element_type(jnp.asarray(x), jnp.asarray(x).dtype), tree)

def _load_custom_episodes(custom_episodes, env):
    with open(custom_episodes, 'rb') as f:
        custom_episode_data = pickle.load(f)
    n = len(custom_episode_data)
    print(f"Custom episodes loaded: {n}")
    custom_data = {
        "full_state": jnp.array([custom_episode_data[i]["full_state"] for i in range(n)]),
        "robot_goal": jnp.array([custom_episode_data[i]["robot_goal"] for i in range(n)]),
        "humans_goal": jnp.array([custom_episode_data[i]["humans_goal"] for i in range(n)]),
        "static_obstacles": jnp.array([custom_episode_data[i]["static_obstacles"] for i in range(n)]),
        "scenario": jnp.array([custom_episode_data[i]["scenario"] for i in range(n)]),
        "humans_radius": jnp.array([custom_episode_data[i]["humans_radius"] for i in range(n)]),
        "humans_speed": jnp.array([custom_episode_data[i]["humans_speed"] for i in range(n)]),
        "humans_delay": jnp.zeros((n, env.n_humans)),
    }
    return n, custom_data

def _reset_episodes(env, keys, episode_ids, n_episodes, custom_data):
    # Reset one env per episode id (custom episodes if provided), with weak types stripped
    if custom_data is None:
        states, _, obses, infos, outcomes = env.batch_reset(keys)
    else:
        episodes = tree_map(lambda x: x[jnp.minimum(episode_ids, n_episodes - 1)], custom_data)
        states, _, obses, infos, outcomes = vmap(env.reset_custom_episode)(keys, episodes)
    return _strip_weak_types((states, obses, infos, outcomes))

def _add_experiences(buffer_state, buffer_size, vnet_inputs, targets, keep, replay_buffer):
    """
    Adds (vnet_inputs[n], targets[n]) for all n where keep[n], in order, to the ring buffer (same indices as
    replay_buffer.add). Only the experiences that are not overwritten within this call are written (no duplicate indices).
    """
    capacity = replay_buffer.buffer_size
    keep = keep.astype(jnp.int32)
    positions = buffer_size + jnp.cumsum(keep) - keep
    n_added = jnp.sum(keep)
    write = (keep == 1) & (positions >= buffer_size + n_added - capacity)
    idxs = jnp.where(write, positions % capacity, capacity) # Out of bounds indices are dropped
    buffer_state = {
        **buffer_state,
        "vnet_inputs": buffer_state["vnet_inputs"].at[idxs].set(jnp.reshape(vnet_inputs, (-1, *buffer_state["vnet_inputs"].shape[1:])), mode="drop"),
        "targets": buffer_state["targets"].at[idxs].set(jnp.reshape(targets, (-1, *buffer_state["targets"].shape[1:])), mode="drop"),
    }
    return buffer_state, buffer_size + n_added

def _env_major(x):
    # (max_steps, n_envs, ...) -> (n_envs * max_steps, ...)
    return jnp.reshape(jnp.swapaxes(x, 0, 1), (x.shape[0] * x.shape[1], *x.shape[2:]))

### RL

@partial(jit, static_argnames=("policy", "env", "max_steps"))
def simulate_rl_episodes(
    model_params,
    epsilons,
    policy_keys,
    states,
    obses,
    infos,
    outcomes,
    policy:BasePolicy,
    env:BaseEnv,
    max_steps:int,
):
    """
    Runs one epsilon-greedy RL episode per env in parallel (same model parameters for all envs), until all of them
    are over. Finished envs are frozen. Returns the time-major history (vnet inputs, rewards, dones), the length of
    each episode, and the final outcomes. History entries after the end of an episode are zero.
    """
    n_envs = states.shape[0]
    def _cond(carry):
        t, _, _, _, outcomes, _, _, _ = carry
        return (t < max_steps) & jnp.any(outcomes["nothing"])
    def _body(carry):
        t, states, obses, infos, outcomes, keys, lengths, history = carry
        active = outcomes["nothing"]
        actions, new_keys, vnet_inputs, _ = vmap(policy.act, in_axes=(0,0,0,None,0))(keys, obses, infos, model_params, epsilons)
        new_states, new_obses, new_infos, (rewards, _), new_outcomes, _ = vmap(env.step)(states, infos, actions)
        history = {
            "vnet_inputs": history["vnet_inputs"].at[t].set(_select_envs(active, vnet_inputs, jnp.zeros_like(vnet_inputs))),
            "rewards": history["rewards"].at[t].set(jnp.where(active, rewards, 0.)),
            "dones": history["dones"].at[t].set(jnp.where(active, ~(new_outcomes["nothing"]), False)),
        }
        states, obses, infos, outcomes, keys = _select_envs(
            active, (new_states, new_obses, new_infos, new_outcomes, new_keys), (states, obses, infos, outcomes, keys)
        )
        return t + 1, states, obses, infos, outcomes, keys, lengths + active, history
    history = {
        "vnet_inputs": jnp.zeros((max_steps, n_envs, env.n_humans, policy.vnet_input_size)),
        "rewards": jnp.zeros((max_steps, n_envs)),
        "dones": jnp.zeros((max_steps, n_envs)),
    }
    init = (0, states, obses, infos, outcomes, policy_keys, jnp.zeros((n_envs,), dtype=jnp.int32), history)
    _, _, _, _, outcomes, _, lengths, history = lax.while_loop(_cond, _body, init)
    return history, lengths, outcomes

@partial(jit, static_argnames=(
    "policy", "env", "model", "optimizer", "replay_buffer", "epsilon_decay_fn", "epsilon_start", "epsilon_end", "decay_rate",
    "max_steps", "stored_steps", "buffer_size", "num_batches", "target_update_interval", "exploration_episodes",
))
def vnet_rl_wave(
    episode_ids,
    n_episodes,
    random_seed,
    training_state,
    custom_data,
    policy:BasePolicy,
    env:BaseEnv,
    model:hk.Transformed,
    optimizer:optax.GradientTransformation,
    replay_buffer:BaseVNetReplayBuffer,
    epsilon_decay_fn:Callable,
    epsilon_start:float,
    epsilon_end:float,
    decay_rate:float,
    max_steps:int,
    stored_steps:int,
    buffer_size:int,
    num_batches:int,
    target_update_interval:int,
    exploration_episodes:int,
):
    """
    Collects the episodes with the given ids in parallel with the current model, then processes them one by one in
    order exactly as vnet_rollouts.vnet_rl_rollout does after each episode (buffer update, shuffle, num_batches model
    updates, target network update). Ids >= n_episodes are padding (last wave) and are skipped.
    """
    model_params = training_state[0]
    # Policy, buffer and reset keys are all PRNGKey(random_seed + i), as in vnet_rollouts.py
    keys = vmap(random.PRNGKey)(random_seed + episode_ids)
    states, obses, infos, outcomes = _reset_episodes(env, keys, episode_ids, n_episodes, custom_data)
    epsilons = vmap(lambda i: lax.cond(
        i >= exploration_episodes,
        lambda x: epsilon_decay_fn(*x),
        lambda _: 1.0,
        (epsilon_start, epsilon_end, i - exploration_episodes, decay_rate)))(episode_ids)
    history, lengths, outcomes = simulate_rl_episodes(model_params, epsilons, keys, states, obses, infos, outcomes, policy, env, max_steps)
    # vnet_rollouts.py stores at most stored_steps steps per episode: data of later steps is dropped, and reading it
    # returns the last stored step (out of bounds indices are clamped).
    history = tree_map(lambda x: x.at[stored_steps:].set(x[stored_steps-1]), history)
    discount = pow(policy.reward_function.gamma, policy.dt * policy.v_max)

    def _process_episode(training_state, e):
        def _process(training_state):
            model_params, target_model_params, optimizer_state, buffer_state, current_buffer_size, losses, returns = training_state
            i = episode_ids[e]
            vnet_inputs, rewards, dones, length = history["vnet_inputs"][:,e], history["rewards"][:,e], history["dones"][:,e], lengths[e]
            # Compute episode return
            def _return_body(value, j):
                return value + jnp.where(j < length, pow(policy.reward_function.gamma, j * policy.dt * policy.v_max) * rewards[j], 0.), None
            cumulative_episode_reward, _ = lax.scan(_return_body, jnp.zeros(()), jnp.arange(max_steps))
            # Add only experiences of successful or failed episodes (targets computed with the current target network)
            def _add(args):
                buffer_state, current_buffer_size = args
                next_vnet_inputs = jnp.concatenate([vnet_inputs[1:], vnet_inputs[-1:]])
                next_values = jnp.reshape(vmap(model.apply, in_axes=(None,None,0))(target_model_params, None, next_vnet_inputs), (max_steps,))
                targets = rewards + (1 - dones) * discount * next_values
                return _add_experiences(buffer_state, current_buffer_size, vnet_inputs, targets, jnp.arange(max_steps) < length, replay_buffer)
            buffer_state, current_buffer_size = lax.cond(
                outcomes["success"][e] | outcomes["failure"][e],
                _add,
                lambda x: x,
                (buffer_state, current_buffer_size))
            # Shuffle the buffer if this is full (otherwise it is not possible without making a mess) and the number of experiences used is equal to buffer size
            actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_size]))
            buffer_state, _ = lax.cond(
                (actual_buffer_size == buffer_size) & (((i * num_batches * replay_buffer.batch_size) % buffer_size) == 0),
                lambda x: replay_buffer.shuffle(*x),
                lambda x: x,
                (buffer_state, keys[e]))
            # Update model parameters (iterate over the buffer in num_batches)
            def _update(args):
                model_params, target_model_params, optimizer_state, losses, returns = args
                def _batch_step(carry, j):
                    model_params, optimizer_state = carry
                    experiences_batch = replay_buffer.iterate(
                        buffer_state,
                        actual_buffer_size,
                        (((((i-exploration_episodes) * num_batches * replay_buffer.batch_size) % buffer_size) / replay_buffer.batch_size) + j))
                    model_params, optimizer_state, loss = policy.update(model_params, optimizer, optimizer_state, experiences_batch)
                    return (model_params, optimizer_state), loss
                (model_params, optimizer_state), all_losses = lax.scan(_batch_step, (model_params, optimizer_state), jnp.arange(num_batches))
                # Update target network
                target_model_params = lax.cond((i-exploration_episodes) % target_update_interval == 0, lambda x: model_params, lambda x: x, target_model_params)
                losses = losses.at[i-exploration_episodes].set(jnp.mean(all_losses))
                returns = returns.at[i-exploration_episodes].set(cumulative_episode_reward)
                return model_params, target_model_params, optimizer_state, losses, returns
            model_params, target_model_params, optimizer_state, losses, returns = lax.cond(
                i >= exploration_episodes,
                _update,
                lambda x: x,
                (model_params, target_model_params, optimizer_state, losses, returns))
            return model_params, target_model_params, optimizer_state, buffer_state, current_buffer_size, losses, returns
        return lax.cond(episode_ids[e] < n_episodes, _process, lambda x: x, training_state), None

    training_state, _ = lax.scan(_process_episode, training_state, jnp.arange(episode_ids.shape[0]))
    return training_state

def vnet_rl_rollout(
        initial_vnet_params: dict,
        train_episodes: int, # Number of episodes to train on, exploration episodes excluded
        random_seed: int,
        model: hk.Transformed,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        buffer_size: int,
        current_buffer_size: int,
        num_batches: int,
        epsilon_decay_fn: Callable,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
        target_update_interval: int = 50,
        custom_episodes: str = None,
        exploration_episodes: int = 0, # Number of episodes to explore before starting training
        debugging: bool = False,
        n_parallel_envs: int = 50,
) -> dict:
    """
    GPU version of vnet_rollouts.vnet_rl_rollout (same inputs and outputs, plus n_parallel_envs).
    Episodes are collected n_parallel_envs at a time with the same model parameters, then each of them goes through the
    same per-episode buffer update, shuffle, model updates, and target network update as in vnet_rollouts.py.
    With n_parallel_envs=1 the algorithm is the same as vnet_rollouts.py. With n_parallel_envs > 1 an episode is
    collected with the model parameters at the start of its group of episodes (up to n_parallel_envs-1 episodes old).
    The time per episode is dominated by the num_batches sequential model updates already with n_parallel_envs=50.
    """
    # If policy is CADRL, the number of humans in the training environment must be 1
    if policy.name == "CADRL": assert env.n_humans == 1, "CADRL policy only supports training with one human."
    # For a correct shuffling of the buffer, the latter size must be a multiple of the number of batches times the batch size
    assert buffer_size % (num_batches * replay_buffer.batch_size) == 0, "The buffer size must be a multiple of the number of batches times the batch size."
    if default_backend() != "gpu":
        print(f"WARNING: JAX default backend is '{default_backend()}', RL will not run on GPU.")

    # Load custom episodes if provided
    if custom_episodes is not None:
        total_episodes, custom_data = _load_custom_episodes(custom_episodes, env)
        train_episodes = total_episodes - exploration_episodes
    else:
        total_episodes = train_episodes + exploration_episodes
        custom_data = None

    # Initialize the arrays where to save data and the model parameters
    losses = jnp.empty([train_episodes], dtype=jnp.float32)
    returns = jnp.empty([train_episodes], dtype=jnp.float32)
    optimizer_state = optimizer.init(initial_vnet_params)
    training_state = _strip_weak_types((
        initial_vnet_params, initial_vnet_params, optimizer_state, buffer_state,
        jnp.asarray(current_buffer_size, dtype=jnp.int32), losses, returns,
    ))
    n_parallel_envs = min(n_parallel_envs, total_episodes)
    n_waves = -(-total_episodes // n_parallel_envs)
    stored_steps = int(env.reward_function.time_limit/env.robot_dt) + 1
    max_steps = stored_steps + 2 # Upper bound of the episode length
    print(f"Performing RL training for {train_episodes} episodes after {exploration_episodes} exploration episodes, {n_parallel_envs} episodes collected in parallel")
    for wave in tqdm(range(n_waves)):
        episode_ids = jnp.arange(wave * n_parallel_envs, (wave + 1) * n_parallel_envs, dtype=jnp.int32)
        training_state = vnet_rl_wave(
            episode_ids, total_episodes, random_seed, training_state, custom_data,
            policy, env, model, optimizer, replay_buffer, epsilon_decay_fn, epsilon_start, epsilon_end, decay_rate,
            max_steps, stored_steps, buffer_size, num_batches, target_update_interval, exploration_episodes,
        )
        training_state[4].block_until_ready() # SYNC (keeps the progress bar honest)
        # DEBUG: print the mean loss and the return every 100 training episodes
        if debugging:
            for i in range(wave * n_parallel_envs, min((wave + 1) * n_parallel_envs, total_episodes)):
                if (i >= exploration_episodes) and ((i - exploration_episodes) % 100 == 0):
                    tqdm.write(f"Episode {i-exploration_episodes} - Mean loss over {num_batches} batches: {float(training_state[5][i-exploration_episodes])} - Return: {float(training_state[6][i-exploration_episodes])}")

    output_dict = {}
    keys = ["model_params", "target_model_params", "optimizer_state", "buffer_state", "current_buffer_size", "losses", "returns"]
    for idx, value in enumerate(training_state): output_dict[keys[idx]] = value
    return output_dict

### IL

@partial(jit, static_argnames=("policy", "env", "replay_buffer", "max_steps", "stored_steps"))
def collect_vnet_il_chunk(
    episode_ids,
    n_episodes,
    random_seed,
    buffer_state,
    buffer_size,
    custom_data,
    policy:BasePolicy,
    env:BaseEnv,
    replay_buffer:BaseVNetReplayBuffer,
    max_steps:int,
    stored_steps:int,
):
    """
    Simulates the IL episodes with the given ids in parallel and adds the successful or failed ones to the buffer.
    Ids >= n_episodes are padding (last chunk) and are discarded. Returns the updated buffer and the episodes returns.
    """
    keys = vmap(random.PRNGKey)(random_seed + episode_ids)
    states, obses, infos, outcomes = _reset_episodes(env, keys, episode_ids, n_episodes, custom_data)
    # For imitation learning, set the humans' safety space to 0.1
    infos["humans_parameters"] = infos["humans_parameters"].at[:,:,-1].set(0.1)
    history, lengths, outcomes = simulate_il_episodes(states, obses, infos, outcomes, policy, env, max_steps)
    # vnet_rollouts.py stores at most stored_steps steps per episode: data of later steps (an episode can last one
    # more step, the one where the timeout is detected) is dropped, and reading it returns the last stored step.
    history = tree_map(lambda x: x.at[stored_steps:].set(x[stored_steps-1]), history)
    targets = compute_il_targets(history["rewards"], lengths, policy)
    added = (outcomes["success"] | outcomes["failure"]) & (episode_ids < n_episodes)
    keep = added[:,None] & (jnp.arange(max_steps)[None,:] < lengths[:,None])
    buffer_state, buffer_size = _add_experiences(
        buffer_state, buffer_size, _env_major(history["inputs"]), _env_major(targets), keep.reshape(-1), replay_buffer
    )
    return buffer_state, buffer_size, targets[0]

@partial(jit, static_argnames=("policy", "optimizer", "replay_buffer", "optimization_steps", "shuffle"))
def train_vnet_il_epoch(
    buffer_key,
    buffer_state,
    buffer_size,
    model_params,
    optimizer_state,
    policy:BasePolicy,
    optimizer:optax.GradientTransformation,
    replay_buffer:BaseVNetReplayBuffer,
    optimization_steps:int,
    shuffle:bool,
):
    # Shuffle the buffer if this is full (otherwise it is not possible without making a mess)
    if shuffle:
        buffer_state, buffer_key = replay_buffer.shuffle(buffer_state, buffer_key)
    def _batch_step(carry, k):
        model_params, optimizer_state = carry
        experiences_batch = replay_buffer.iterate(buffer_state, buffer_size, k)
        model_params, optimizer_state, loss = policy.update(model_params, optimizer, optimizer_state, experiences_batch)
        return (model_params, optimizer_state), loss
    (model_params, optimizer_state), losses = lax.scan(_batch_step, (model_params, optimizer_state), jnp.arange(optimization_steps))
    return buffer_key, buffer_state, model_params, optimizer_state, losses

def vnet_il_rollout(
        initial_vnet_params: dict,
        train_episodes: int,
        random_seed: int,
        optimizer: optax.GradientTransformation,
        buffer_state: dict,
        current_buffer_size: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: BaseVNetReplayBuffer,
        buffer_size: int,
        num_epochs: int,
        batch_size: int,
        custom_episodes: str = None,
        n_parallel_envs: int = 1000,
) -> dict:
    """
    Parallel GPU version of vnet_rollouts.vnet_il_rollout (same inputs and outputs, plus n_parallel_envs).
    Episodes are simulated n_parallel_envs at a time instead of one by one. Episode i is reset with the same key as
    in vnet_rollouts.py, so the collected experiences, returns, and buffer are the same.
    """
    # If policy is CADRL, the number of humans in the training environment must be 1
    if policy.name == "CADRL": assert env.n_humans == 1, "CADRL policy only supports training with one human."
    if default_backend() != "gpu":
        print(f"WARNING: JAX default backend is '{default_backend()}', IL will not run on GPU.")

    # Load custom episodes if provided
    if custom_episodes is not None:
        train_episodes, custom_data = _load_custom_episodes(custom_episodes, env)
    else:
        custom_data = None

    # Simulate the episodes in parallel chunks
    n_parallel_envs = min(n_parallel_envs, train_episodes)
    n_chunks = -(-train_episodes // n_parallel_envs)
    stored_steps = int(env.reward_function.time_limit/env.robot_dt)
    max_steps = stored_steps + 3 # Upper bound of the episode length (the timeout is detected after stored_steps steps)
    current_size = jnp.asarray(current_buffer_size, dtype=jnp.int32)
    chunk_returns = []
    print(f"Simulating {train_episodes} IL episodes, {n_parallel_envs} in parallel...")
    for chunk in tqdm(range(n_chunks)):
        episode_ids = jnp.arange(chunk * n_parallel_envs, (chunk + 1) * n_parallel_envs, dtype=jnp.int32)
        buffer_state, current_size, chunk_return = collect_vnet_il_chunk(
            episode_ids, train_episodes, random_seed, buffer_state, current_size, custom_data,
            policy, env, replay_buffer, max_steps, stored_steps,
        )
        current_size.block_until_ready() # SYNC (keeps the progress bar honest)
        chunk_returns.append(chunk_return)
    returns = jnp.concatenate(chunk_returns)[:train_episodes]
    current_buffer_size = current_size

    # Initialize the optimizer and the buffer key
    optimizer_state = optimizer.init(initial_vnet_params)
    buffer_key = random.PRNGKey(0) # We do not care to control the buffer shuffle

    # Update model parameters
    actual_buffer_size = jnp.min(jnp.array([current_buffer_size, buffer_size]))
    print(f"Buffer size after IL: {int(actual_buffer_size)}")
    optimization_steps = int(actual_buffer_size) // batch_size
    buffer_full = bool(actual_buffer_size == buffer_size)
    if buffer_full:
        print("Pre-shuffling the buffer...")
        buffer_state, buffer_key = replay_buffer.shuffle(buffer_state, buffer_key, 100)
    model_params, optimizer_state = _strip_weak_types((initial_vnet_params, optimizer_state))
    all_losses = [jnp.zeros((0, optimization_steps))]
    print(f"Optimizing model on generated experiences for {num_epochs} epochs...")
    for _ in tqdm(range(num_epochs)):
        buffer_key, buffer_state, model_params, optimizer_state, epoch_losses = train_vnet_il_epoch(
            buffer_key, buffer_state, actual_buffer_size, model_params, optimizer_state,
            policy, optimizer, replay_buffer, optimization_steps, buffer_full,
        )
        epoch_losses.block_until_ready() # SYNC (keeps the progress bar honest)
        all_losses.append(epoch_losses[None,:])
    all_losses = jnp.concatenate(all_losses)

    output_dict = {
        "model_params": model_params,
        "optimizer_state": optimizer_state,
        "buffer_state": buffer_state,
        "current_buffer_size": current_buffer_size,
        "buffer_key": buffer_key,
        "losses": jnp.mean(all_losses, axis=1),
        "returns": returns}
    return output_dict
