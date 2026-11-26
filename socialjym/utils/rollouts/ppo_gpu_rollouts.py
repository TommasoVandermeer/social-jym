import optax
from functools import partial
from jax import jit, lax, random, vmap, device_put, devices, default_backend
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer


def _select_envs(mask, on_true, on_false):
    # Per-env selection between two pytrees whose leaves have the env axis first
    return tree_map(lambda t, f: jnp.where(jnp.reshape(mask, (-1,) + (1,) * (t.ndim - 1)), t, f), on_true, on_false)

def _generate_reset_pool(env, reset_keys):
    # One fresh initial (state, obs, info) per env, with weak types stripped so that it can be carried by lax.scan
    states, reset_keys, obses, infos, _ = env.batch_reset(reset_keys)
    pool = tree_map(lambda x: lax.convert_element_type(x, x.dtype), (states, obses, infos))
    return pool, reset_keys

@partial(jit, static_argnames=("policy", "env", "n_steps"))
def collect_rollout_step(
    actor_params,
    critic_params,
    env_state,
    policy_keys,
    reset_keys,
    policy:BasePolicy,
    env:BaseEnv,
    n_steps:int,
):
    # Finished episodes are reset with initial states generated once per rollout (reset pool). The env auto-reset
    # (reset_if_done=True) is avoided: under vmap its lax.cond and scenario lax.switch are turned into selects, so
    # a full reset (all scenario generators, with their rejection-sampling while loops) runs for every env at every step.
    def _refill_pool(pool, pool_used, r_keys):
        fresh_pool, fresh_keys = _generate_reset_pool(env, r_keys)
        return _select_envs(pool_used, fresh_pool, pool), jnp.zeros_like(pool_used), _select_envs(pool_used, fresh_keys, r_keys)

    def _scan_step(carry, _):
        (actor_params, states, obses, infos, outcomes, p_keys, r_keys, pool, pool_used, episode_count, successes, failures, timeouts, returns) = carry
        n_envs = states.shape[0]
        # Update the running normalization state with the pre-step inputs
        if policy.normalize_and_clip_obs:
            raw_inputs = vmap(policy.batch_compute_vnet_input, in_axes=(0,0,0))(obses[:,-1], obses[:,:-1], infos)
            norm_state = policy.update_norm_state(
                jnp.reshape(raw_inputs, (n_envs * env.n_humans, policy.vnet_input_size)),
                actor_params["norm_state"],
            )
            actor_params = {**actor_params, "norm_state": norm_state}
        actions, new_p_keys, inputs, sampled_actions, distrs = policy.batch_act(p_keys, obses, infos, actor_params, sample=True)
        values = vmap(policy.critic.apply, in_axes=(None,None,0))(critic_params, None, inputs)
        new_states, new_obses, new_infos, (rewards, _), new_outcomes, _ = env.batch_step(
            states, infos, actions, r_keys, test=False, reset_if_done=False
        )
        # Reset finished episodes from the pool (refilled only if an env finishes twice in the same rollout)
        new_dones = ~(new_outcomes["nothing"])
        pool, pool_used, r_keys = lax.cond(
            jnp.any(new_dones & pool_used),
            _refill_pool,
            lambda pool, pool_used, r_keys: (pool, pool_used, r_keys),
            pool, pool_used, r_keys,
        )
        new_states, new_obses, new_infos = _select_envs(new_dones, pool, (new_states, new_obses, new_infos))
        pool_used = pool_used | new_dones
        step_data = {
            "inputs": inputs,
            "values": jnp.reshape(values, (n_envs,)),
            "actions": sampled_actions,
            "rewards": rewards,
            "dones": ~(outcomes["nothing"]), # Dones are computed with the pre-step outcomes (episode starts)
            "neglogpdfs": policy.distr.batch_neglogp(distrs, sampled_actions),
            "stds": policy.distr.batch_std(distrs),
        }
        episode_count = episode_count + jnp.sum(new_dones)
        successes = successes + jnp.sum(new_outcomes["success"])
        failures = failures + jnp.sum(new_outcomes["failure"])
        timeouts = timeouts + jnp.sum(new_outcomes["timeout"])
        # Same formula as ppo_rollouts.py (the discounted reward term is not masked by episode termination)
        returns = returns + (~new_outcomes["nothing"]) * infos['return'] + jnp.power(policy.reward_function.gamma, (infos['step']+1) * policy.dt * policy.v_max) * rewards
        return (actor_params, new_states, new_obses, new_infos, new_outcomes, new_p_keys, r_keys, pool, pool_used, episode_count, successes, failures, timeouts, returns), step_data

    states, obses, infos, outcomes = env_state
    pool, reset_keys = _generate_reset_pool(env, reset_keys)
    zero_count = jnp.zeros((), dtype=jnp.int32)
    init_carry = (
        actor_params,
        states,
        obses,
        infos,
        outcomes,
        policy_keys,
        reset_keys,
        pool,
        jnp.zeros((states.shape[0],), dtype=bool),
        zero_count,
        zero_count,
        zero_count,
        zero_count,
        jnp.zeros((states.shape[0],)),
    )
    final_carry, history = lax.scan(_scan_step, init_carry, None, length=n_steps)
    (actor_params, states, obses, infos, outcomes, policy_keys, reset_keys, _, _, episode_count, successes, failures, timeouts, returns) = final_carry
    stats = {
        "episodes": episode_count,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "returns": returns,
    }
    return actor_params, (states, obses, infos, outcomes), policy_keys, reset_keys, history, stats

@partial(jit, static_argnames=("policy", "lambda_gae"))
def process_buffer_and_gae(
    actor_params,
    critic_params,
    last_obses,
    last_infos,
    last_outcomes,
    history,
    policy:BasePolicy,
    lambda_gae:float,
):
    n_steps, n_envs = history["values"].shape
    # Compute the value of the last states
    inputs = vmap(policy.batch_compute_vnet_input, in_axes=(0,0,0))(last_obses[:,-1], last_obses[:,:-1], last_infos)
    if policy.normalize_and_clip_obs:
        inputs = jnp.reshape(
            policy.normalize_and_clip_inputs(jnp.reshape(inputs, (-1, policy.vnet_input_size)), actor_params["norm_state"]),
            inputs.shape,
        )
    last_values = jnp.reshape(vmap(policy.critic.apply, in_axes=(None,None,0))(critic_params, None, inputs), (n_envs,))
    values_ext = jnp.concatenate([history["values"], last_values[None,:]], axis=0)
    dones_ext = jnp.concatenate([history["dones"], ~(last_outcomes["nothing"])[None,:]], axis=0).astype(jnp.float32)
    rewards = history["rewards"]
    # Compute advantages with GAE (reverse scan over time)
    discount = pow(policy.reward_function.gamma, policy.dt * policy.v_max)
    gae_discount = pow(policy.reward_function.gamma * lambda_gae, policy.dt * policy.v_max)
    def _gae_step(adv_next, t):
        mask = 1 - dones_ext[t+1]
        delta = rewards[t] + discount * values_ext[t+1] * mask - values_ext[t]
        advantage = delta + gae_discount * adv_next * mask
        return advantage, advantage
    _, advantages = lax.scan(_gae_step, jnp.zeros((n_envs,)), jnp.arange(n_steps)[::-1])
    advantages = advantages[::-1]
    critic_targets = advantages + history["values"]
    # Flatten env-major, same ordering as the PPOBuffer filled by ppo_rollouts.py
    def flatten(x):
        return jnp.reshape(jnp.swapaxes(x, 0, 1), (n_envs * n_steps, *x.shape[2:]))
    return {
        "inputs": flatten(history["inputs"]),
        "critic_targets": flatten(critic_targets),
        "sample_actions": flatten(history["actions"]),
        "old_values": flatten(history["values"]),
        "old_neglogpdfs": flatten(history["neglogpdfs"]),
    }

@partial(jit, static_argnames=("policy", "actor_optimizer", "critic_optimizer", "n_epochs", "n_batches", "batch_size", "clip_range", "beta_entropy"))
def train_epochs(
    shuffle_key,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    buffer,
    policy:BasePolicy,
    actor_optimizer:optax.GradientTransformation,
    critic_optimizer:optax.GradientTransformation,
    n_epochs:int,
    n_batches:int,
    batch_size:int,
    clip_range:float,
    beta_entropy:float,
):
    buffer_size = buffer["critic_targets"].shape[0]

    def _batch_step(carry, experiences):
        actor_params, critic_params, actor_opt_state, critic_opt_state = carry
        new_critic_params, new_actor_params, critic_opt_state, actor_opt_state, critic_loss, actor_loss, entropy_loss = policy.update(
            critic_params,
            actor_params["actor_params"] if policy.normalize_and_clip_obs else actor_params,
            actor_optimizer,
            actor_opt_state,
            critic_optimizer,
            critic_opt_state,
            experiences,
            beta_entropy,
            clip_range,
            debugging=False,
        )
        if policy.normalize_and_clip_obs:
            new_actor_params = {
                "actor_params": new_actor_params,
                "norm_state": actor_params["norm_state"],
            }
        return (new_actor_params, new_critic_params, actor_opt_state, critic_opt_state), (actor_loss, critic_loss, entropy_loss)

    def _epoch_step(carry, _):
        shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, buffer = carry
        # Same shuffling as BaseVNetReplayBuffer.shuffle (the shuffled buffer is carried to the next epoch)
        shuffle_key, subkey = random.split(shuffle_key)
        perm = random.permutation(subkey, jnp.arange(buffer_size))
        buffer = tree_map(lambda x: x[perm], buffer)
        minibatches = tree_map(lambda x: jnp.reshape(x[:n_batches*batch_size], (n_batches, batch_size, *x.shape[1:])), buffer)
        (actor_params, critic_params, actor_opt_state, critic_opt_state), losses = lax.scan(
            _batch_step,
            (actor_params, critic_params, actor_opt_state, critic_opt_state),
            minibatches,
        )
        return (shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, buffer), losses

    (shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, _), (actor_losses, critic_losses, entropy_losses) = lax.scan(
        _epoch_step,
        (shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, buffer),
        None,
        length=n_epochs,
    )
    losses = {
        "actor_losses": actor_losses, # Shape (n_epochs, n_batches)
        "critic_losses": critic_losses,
        "entropy_losses": entropy_losses,
    }
    return shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, losses

@partial(jit, static_argnames=("policy", "env", "actor_optimizer", "critic_optimizer", "n_steps", "n_epochs", "n_batches", "batch_size", "clip_range", "beta_entropy", "lambda_gae"))
def ppo_update(
    upd_idx,
    runner_state,
    aux_data,
    policy:BasePolicy,
    env:BaseEnv,
    actor_optimizer:optax.GradientTransformation,
    critic_optimizer:optax.GradientTransformation,
    n_steps:int,
    n_epochs:int,
    n_batches:int,
    batch_size:int,
    clip_range:float,
    beta_entropy:float,
    lambda_gae:float,
):
    actor_params, critic_params, actor_opt_state, critic_opt_state, env_state, policy_keys, reset_keys, shuffle_key = runner_state
    # A. COLLECT ROLLOUT
    actor_params, env_state, policy_keys, reset_keys, history, stats = collect_rollout_step(
        actor_params, critic_params, env_state, policy_keys, reset_keys, policy, env, n_steps
    )
    # B. GAE AND BUFFER
    buffer = process_buffer_and_gae(
        actor_params, critic_params, env_state[1], env_state[2], env_state[3], history, policy, lambda_gae
    )
    # C. TRAIN
    shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, losses = train_epochs(
        shuffle_key, actor_params, critic_params, actor_opt_state, critic_opt_state, buffer,
        policy, actor_optimizer, critic_optimizer, n_epochs, n_batches, batch_size, clip_range, beta_entropy,
    )
    # D. SAVE AUXILIARY DATA
    new_aux_data = {
        "successes": stats["successes"],
        "failures": stats["failures"],
        "timeouts": stats["timeouts"],
        "returns": jnp.sum(stats["returns"]) / stats["episodes"],
        "episodes": stats["episodes"],
        "stds": jnp.mean(history["stds"], axis=(0,1)),
        "actor_losses": jnp.mean(losses["actor_losses"], axis=(0,1)),
        "critic_losses": jnp.mean(losses["critic_losses"], axis=(0,1)),
        "entropy_losses": jnp.mean(losses["entropy_losses"], axis=(0,1)),
    }
    aux_data = {k: aux_data[k].at[upd_idx].set(new_aux_data[k]) for k in aux_data}
    runner_state = (actor_params, critic_params, actor_opt_state, critic_opt_state, env_state, policy_keys, reset_keys, shuffle_key)
    return runner_state, aux_data

def _print_debug_stats(aux_data, upd_idx, debugging_interval, actor_params, policy):
    # Same statistics as ppo_rollouts.py, computed on the last debugging_interval updates
    window = slice(max(upd_idx - debugging_interval + 1, 0), upd_idx + 1)
    w = {k: np.asarray(v[window]) for k, v in aux_data.items()}
    total_episodes = int(jnp.sum(aux_data["episodes"]))
    window_episodes = np.sum(w["episodes"])
    tqdm.write(f"\n### Update {upd_idx}")
    tqdm.write(
        f"Episodes {total_episodes}\n"
        f"Actor loss: {np.nanmean(np.abs(w['actor_losses']))}\n"
        f"Critic loss: {np.nanmean(w['critic_losses'])}\n"
        f"Entropy: {np.nanmean(w['entropy_losses'])}\n"
        f"Std: {np.nanmean(w['stds'], axis=0)}\n"
        f"Return: {np.nanmean(w['returns'])}\n"
        f"Succ.Rate: {np.sum(w['successes']) / window_episodes}\n"
        f"Fail.Rate: {np.sum(w['failures']) / window_episodes}\n"
        f"Tim.Rate: {np.sum(w['timeouts']) / window_episodes}\n"
        f"Norm state mean: {np.asarray(actor_params['norm_state']['mean']) if policy.normalize_and_clip_obs else np.nan}\n"
        f"Norm state var: {np.asarray(actor_params['norm_state']['var']) if policy.normalize_and_clip_obs else np.nan}"
    )

def ppo_rl_rollout(
        initial_actor_params: dict,
        initial_critic_params: dict,
        n_parallel_envs:int,
        train_updates: int, # Number of episodes to train on, exploration episodes excluded
        random_seed: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        buffer_state: dict,
        buffer_capacity: int,
        policy: BasePolicy,
        env: BaseEnv,
        replay_buffer: PPOBuffer,
        clip_range: float,
        n_epochs: int,
        beta_entropy: float,
        lambda_gae: float,
        debugging: bool = False,
        debugging_interval: int = 100,
) -> dict:
    """
    Single-GPU drop-in replacement of ppo_rollouts.ppo_rl_rollout (same inputs, same output dict).
    Rollout collection, GAE, and PPO updates of each training update run in a single jitted call,
    so environment states, experiences, and network parameters never leave the device.
    The replay buffer is only used for its batch size: experiences are minibatched directly on device.
    Finished episodes are reset from initial states generated once per update (see collect_rollout_step), so the
    sequence of reset episodes differs from ppo_rollouts.py while following the same scenario distribution.
    """
    policies = ["SARL-PPO", "DIRSAFE"]
    assert policy.name in policies, "This function is only compatible with PPO policies."
    # Compute number of steps to simulate per update for each parallel env
    assert buffer_capacity % n_parallel_envs == 0, "The buffer capacity must be a multiple of the number of parallel environments. Otherwise you will trow away experiences."
    assert buffer_capacity == replay_buffer.buffer_size, "The buffer capacity must be equal to the buffer size of the replay buffer."
    n_steps = int(buffer_capacity / n_parallel_envs)
    n_batches = int(buffer_capacity / replay_buffer.batch_size)
    device = devices()[0]
    if default_backend() != "gpu":
        print(f"WARNING: JAX default backend is '{default_backend()}', PPO training will not run on GPU.")
    # Initialize the auxiliary data array
    aux_data = {
        "actor_losses": jnp.zeros([train_updates], dtype=jnp.float32),
        "critic_losses": jnp.zeros([train_updates], dtype=jnp.float32),
        "entropy_losses": jnp.zeros([train_updates], dtype=jnp.float32),
        "returns": jnp.zeros([train_updates], dtype=jnp.float32),
        "successes": jnp.zeros([train_updates], dtype=int),
        "failures": jnp.zeros([train_updates], dtype=int),
        "timeouts": jnp.zeros([train_updates], dtype=int),
        "episodes": jnp.zeros([train_updates], dtype=int),
        "stds": jnp.zeros([train_updates, 2], dtype=jnp.float32),
    }
    # Initialize the optimizer state
    if policy.normalize_and_clip_obs:
        actor_optimizer_state = actor_optimizer.init(initial_actor_params["actor_params"])
    else:
        actor_optimizer_state = actor_optimizer.init(initial_actor_params)
    critic_optimizer_state = critic_optimizer.init(initial_critic_params)
    # Initialize the random keys
    policy_keys = vmap(random.PRNGKey)(jnp.arange(n_parallel_envs, dtype=int) + random_seed)
    reset_keys = vmap(random.PRNGKey)(jnp.arange(n_parallel_envs, dtype=int) + random_seed)
    shuffle_key = random.PRNGKey(random_seed)
    # Initialize environments
    states, reset_keys, obses, infos, init_outcomes = env.batch_reset(reset_keys)
    # Move everything to the device once, it will stay there for the whole training
    runner_state = device_put(
        (
            initial_actor_params,
            initial_critic_params,
            actor_optimizer_state,
            critic_optimizer_state,
            (states, obses, infos, init_outcomes),
            policy_keys,
            reset_keys,
            shuffle_key,
        ),
        device,
    )
    # Strip weak types (e.g. infos from env.batch_reset) so that ppo_update input and output types match (compiled only once)
    runner_state = tree_map(lambda x: lax.convert_element_type(x, x.dtype), runner_state)
    aux_data = device_put(aux_data, device)
    update_step = partial(
        ppo_update,
        policy=policy,
        env=env,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        n_steps=n_steps,
        n_epochs=n_epochs,
        n_batches=n_batches,
        batch_size=replay_buffer.batch_size,
        clip_range=clip_range,
        beta_entropy=beta_entropy,
        lambda_gae=lambda_gae,
    )
    # Execute the training loop
    print(f"Performing RL training for {train_updates} updates on {device}...")
    for upd_idx in tqdm(range(train_updates)):
        runner_state, aux_data = update_step(upd_idx, runner_state, aux_data)
        aux_data["episodes"].block_until_ready() # SYNC (keeps the progress bar honest)
        if debugging and (upd_idx % debugging_interval == 0) and (upd_idx != 0):
            _print_debug_stats(aux_data, upd_idx, debugging_interval, runner_state[0], policy)
    actor_params, critic_params, actor_optimizer_state, critic_optimizer_state, (states, obses, infos, outcomes), policy_keys, reset_keys, shuffle_key = runner_state
    output_dict = {
        "policy_keys": policy_keys,
        "reset_keys": reset_keys,
        "shuffle_key": shuffle_key,
        "states": states,
        "obses": obses,
        "infos": infos,
        "outcomes": outcomes,
        "actor_params": actor_params,
        "critic_params": critic_params,
        "actor_optimizer_state": actor_optimizer_state,
        "critic_optimizer_state": critic_optimizer_state,
        "buffer_state": replay_buffer.empty(buffer_state),
        "current_buffer_size": jnp.asarray(0),
        "aux_data": aux_data,
    }
    return output_dict
