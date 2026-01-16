import optax
from jax import jit, lax, random, vmap, debug
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from jax import value_and_grad

from jhsfm.hsfm import get_linear_velocity


@partial(jit, static_argnames=("policy", "env", "n_steps"))
def collect_rollout_step(
    key, 
    network_params, 
    env_state, 
    policy_keys, 
    reset_keys, 
    template_outcomes, # <--- NUOVO ARGOMENTO
    policy, 
    env, 
    n_steps
):
    
    def _scan_step(carry, _):
        (states, obses, infos, p_keys, r_keys, outcomes_acc) = carry
        
        # ... (Il resto del codice di _scan_step rimane UGUALE) ...
        # Action Selection
        actions, new_p_keys, _, sampled_actions, _, actor_distrs, values = policy.batch_act(
            p_keys, obses, infos, network_params, sample=True
        )
        
        # Env Step
        new_states, new_obses, new_infos, rewards, outcomes, new_r_keys = env.batch_step(
            states, infos, actions, r_keys, test=False, reset_if_done=True
        )
        
        # ... (Logica Supervised e Pack Data rimane UGUALE) ...
        rc_humans_positions, _, rc_humans_velocities, rc_obstacles, _ = env.batch_robot_centric_transform(
            states[:,:-1,:2], 
            states[:,:-1,4], 
            vmap(vmap(get_linear_velocity))(states[:,:-1,4], states[:,:-1,2:4]),
            infos["static_obstacles"][:,-1], 
            states[:,-1,:2], 
            states[:,-1,4], 
            infos["robot_goal"],
        )
        
        humans_visibility, _ = env.batch_object_visibility(
            rc_humans_positions, infos["humans_parameters"][:,:,0], rc_obstacles
        )
        humans_in_range = env.batch_humans_inside_lidar_range(
            rc_humans_positions, infos["humans_parameters"][:,:,0]
        )

        step_data = {
            "obs": obses,
            "robot_goal": infos["robot_goal"],
            "gt_poses": rc_humans_positions,
            "gt_vels": rc_humans_velocities,
            "gt_mask": humans_visibility & humans_in_range,
            "values": values,
            "actions": sampled_actions,
            "rewards": rewards,
            # NOTA: Qui outcomes è disponibile direttamente dal return di batch_step
            "dones": ~(outcomes["nothing"]), 
            "neglogpdfs": policy.dirichlet.batch_neglogp(actor_distrs, sampled_actions),
            "stds": policy.dirichlet.batch_std(actor_distrs)
        }
        
        # Accumula outcomes
        new_outcomes_acc = {k: outcomes_acc[k] + outcomes[k] for k in outcomes}
        
        return (new_states, new_obses, new_infos, new_p_keys, new_r_keys, new_outcomes_acc), step_data

    # --- CORREZIONE QUI ---
    # Usiamo template_outcomes (passato come argomento) per creare gli zeri
    init_outcomes_acc = {
        k: jnp.zeros_like(template_outcomes[k], dtype=jnp.int32) 
        for k in template_outcomes
    }
    # ----------------------
    
    init_carry = (env_state[0], env_state[1], env_state[2], policy_keys, reset_keys, init_outcomes_acc)
    
    final_carry, history = lax.scan(_scan_step, init_carry, None, length=n_steps)
    
    (final_states, final_obses, final_infos, final_p_keys, final_r_keys, sum_outcomes) = final_carry
    
    next_env_state = (final_states, final_obses, final_infos)
    
    return next_env_state, final_p_keys, final_r_keys, history, sum_outcomes

@partial(jit, static_argnames=("policy",))
def process_buffer_and_gae(
    network_params, 
    last_obs, 
    last_info, 
    history, 
    policy, 
    gamma, 
    lambda_gae
):
    """
    Calcola l'ultimo value, GAE, Returns e appiattisce il buffer.
    """
    # 1. Calcola Value per l'ultimo stato (Bootstrapping)
    # Serve per il GAE dell'ultimo step
    perception_inputs, robot_state_inputs = vmap(policy.compute_e2e_input, in_axes=(0,0))(
        last_obs, last_info['robot_goal']
    )
    _, _, _, _, _ , _, last_values, _ = policy.e2e.apply(
        network_params, None, perception_inputs, robot_state_inputs
    )
    
    # Unpack history
    rewards = history["rewards"]
    values = history["values"]
    dones = history["dones"]
    
    # Aggiungi l'ultimo valore alla sequenza per semplificare il loop GAE
    # values shape: [Steps, Envs] -> [Steps + 1, Envs]
    values_ext = jnp.concatenate([values, last_values[None, :]], axis=0)
    
    # 2. GAE Calculation (Scan all'indietro)
    def _gae_step(gae_carry, i):
        # i va da n_steps-1 a 0
        adv_next = gae_carry
        
        # delta = r + gamma * V(s') * (1-done) - V(s)
        # Nota: Qui assumo gamma costante, se usi dynamic gamma adatta qui
        mask = 1.0 - dones[i].astype(jnp.float32)
        delta = rewards[i] + gamma * values_ext[i+1] * mask - values_ext[i]
        
        # A_t = delta + gamma * lambda * A_{t+1} * (1-done)
        advantage = delta + gamma * lambda_gae * adv_next * mask
        
        return advantage, advantage # Carry, Output

    # Scan inverso
    n_steps = rewards.shape[0]
    _, advantages = lax.scan(_gae_step, jnp.zeros_like(values[0]), jnp.arange(n_steps)[::-1])
    
    # Riordina gli advantages (perché scan li ha prodotti inversi)
    advantages = advantages[::-1]
    
    # Calcola Critic Targets (Returns)
    critic_targets = advantages + values
    
    # 3. Flattening del Buffer
    # Da [Steps, Envs, ...] a [Steps * Envs, ...]
    def flatten(x):
        return jnp.reshape(x, (-1, *x.shape[2:]))
        
    flattened_buffer = {
        "observations": flatten(history["obs"]),
        "robot_goals": flatten(history["robot_goal"]),
        "gt_poses": flatten(history["gt_poses"]),
        "gt_vels": flatten(history["gt_vels"]),
        "gt_mask": flatten(history["gt_mask"]),
        "actions": flatten(history["actions"]),
        "values": flatten(history["values"]),
        "neglogpdfs": flatten(history["neglogpdfs"]),
        "critic_targets": flatten(critic_targets),
        "advantages": flatten(advantages),
    }
    
    # Normalizza vantaggi (Opzionale ma consigliato per PPO)
    flat_adv = flattened_buffer["advantages"]
    flattened_buffer["advantages"] = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)
    
    return flattened_buffer

@partial(jit, static_argnames=("policy", "optimizer", "mini_batch_size", "micro_batch_size", "n_epochs", "clip_range", "beta_entropy"))
def run_ppo_update(
    key,
    network_params,
    opt_state,
    buffer,
    policy,
    optimizer,
    mini_batch_size,
    micro_batch_size, # <--- NUOVO PARAMETRO
    n_epochs,
    clip_range,
    beta_entropy
):
    total_samples = buffer["actions"].shape[0]
    n_minibatches = total_samples // mini_batch_size
    
    # Calcoliamo quante split fare
    # Es: mini=100, micro=20 -> n_micro_splits=5
    n_micro_splits = mini_batch_size // micro_batch_size
    
    def _epoch_step(carry, _):
        key, params, opt_st = carry
        key, subkey = random.split(key)
        
        perm = random.permutation(subkey, total_samples)
        shuffled_buffer = tree_map(lambda x: x[perm], buffer)
        
        def _batch_step(carry_inner, i):
            params_inner, opt_st_inner = carry_inner
            
            # 1. Estrai il Minibatch intero (es. 100 samples)
            start = i * mini_batch_size
            idxs = start + jnp.arange(mini_batch_size)
            mb = tree_map(lambda x: x[idxs], shuffled_buffer)
            
            # 2. Reshape per Micro-Batching
            # Da [100, ...] a [5, 20, ...]
            def reshape_for_micro(x):
                # Assumiamo che la dim 0 sia il batch
                return x.reshape((n_micro_splits, micro_batch_size, *x.shape[1:]))
            
            micro_batches = tree_map(reshape_for_micro, mb)
            
            # --- Funzione Loss su un SINGOLO Micro-Batch ---
            def micro_batch_loss_fn(p, u_mb):
                # Prepara input e2e
                inputs0, inputs1 = vmap(policy.compute_e2e_input, in_axes=(0,0))(
                    u_mb["observations"], u_mb["robot_goals"]
                )
                
                # Forward pass
                (perc_dist, _, _, actor_dist, _, _, pred_val, log_vars) = policy.e2e.apply(
                    p, None, inputs0, inputs1
                )
                
                # Actor Loss
                new_neglogp = vmap(policy.dirichlet.neglogp)(actor_dist, u_mb["actions"])
                ratio = jnp.exp(u_mb["neglogpdfs"] - new_neglogp)
                surr1 = ratio * u_mb["advantages"]
                surr2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * u_mb["advantages"]
                actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                
                # Critic Loss
                v_loss = jnp.square(pred_val - u_mb["critic_targets"])
                v_clipped = u_mb["values"] + jnp.clip(pred_val - u_mb["values"], -clip_range, clip_range)
                v_loss_clipped = jnp.square(v_clipped - u_mb["critic_targets"])
                critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss, v_loss_clipped))
                
                # Entropy
                entropy = jnp.mean(policy.dirichlet.entropy(actor_dist))
                entropy_loss = -beta_entropy * entropy
                
                # Perception Loss
                gt_dict = {
                    "gt_mask": u_mb["gt_mask"], 
                    "gt_poses": u_mb["gt_poses"], 
                    "gt_vels": u_mb["gt_vels"]
                }
                batch_perc_loss = policy._encoder_loss(perc_dist, gt_dict)
                perception_loss = jnp.mean(batch_perc_loss)
                
                # Total Loss
                w_actor = jnp.exp(-log_vars[0]) * actor_loss + log_vars[0]
                w_critic = jnp.exp(-log_vars[1]) * critic_loss + log_vars[1]
                w_perc = jnp.exp(-log_vars[2]) * perception_loss + log_vars[2]
                
                total_loss = 0.5 * (w_actor + w_critic + w_perc) + entropy_loss
                
                return total_loss, (actor_loss, critic_loss, perception_loss, entropy_loss, log_vars)

            # 3. Loop sui Micro-Batches per accumulare i gradienti
            def _micro_step_scan(unused_carry, u_mb):
                # Esegui forward + backward
                (loss, aux), grads = value_and_grad(micro_batch_loss_fn, has_aux=True)(params_inner, u_mb)
                
                # Scompatta le loss ausiliarie per chiarezza
                l_actor, l_critic, l_perc, l_entropy, l_logvars = aux
                
                # --- DETECT NAN SOURCES ---
                
                # 1. Controlla se i GRADIENTI sono rotti
                grads_any_nan = tree_reduce(
                    lambda a, b: a | b, 
                    tree_map(lambda x: jnp.any(jnp.isnan(x)), grads)
                )
                
                # 2. Controlla se le LOSS stesse sono rotte (NaN o Inf)
                vals_any_nan = jnp.isnan(loss) | jnp.isinf(loss)
                
                # Triggera se c'è un problema nei gradienti O nei valori
                is_bad = grads_any_nan | vals_any_nan

                def print_breakdown():
                    debug.print("\n🚨 CRITICAL FAILURE DETECTED 🚨")
                    debug.print("------------------------------------------------")
                    debug.print("Status: Gradients NaN? {} | Total Loss Infinite? {}", grads_any_nan, vals_any_nan)
                    debug.print("------------------------------------------------")
                    debug.print("1. Actor Loss:      {}", l_actor)
                    debug.print("2. Critic Loss:     {}", l_critic)
                    debug.print("3. Perception Loss: {}", l_perc)
                    debug.print("4. Entropy Loss:    {}", l_entropy)
                    debug.print("5. Log Vars:        {}", l_logvars)
                    debug.print("------------------------------------------------")
                    
                    # Logica deduttiva per aiutarti
                    # Se la loss è finita ma il gradiente no, controlla le operazioni matematiche (Norm, Sqrt)
                    lax.cond(
                        grads_any_nan & (~vals_any_nan),
                        lambda: debug.print("👉 DIAGNOSI: Le Loss sono valide (numeri finiti), ma i Gradienti sono NaN.\n   Cerca 'Division by Zero' o 'Norma di Zero' dentro la rete che ha la loss più alta."),
                        lambda: debug.print("👉 DIAGNOSI: Una delle Loss è diventata NaN/Inf. Guarda quale è 'nan' qui sopra.")
                    )

                # Stampa solo se c'è un errore
                lax.cond(is_bad, print_breakdown, lambda: None)
                
                return None, (loss, aux, grads)

            # Esegui scan
            _, (m_losses, m_aux, m_grads) = lax.scan(
                _micro_step_scan, None, micro_batches
            )
            
            # 4. Aggregazione Gradienti e Loss (Media)
            # Poiché la loss è una media, la media dei gradienti dei micro-batches è il gradiente del batch intero.
            grads_avg = tree_map(lambda x: jnp.mean(x, axis=0), m_grads)
            loss_avg = jnp.mean(m_losses)
            
            # Aggrega metriche ausiliarie
            aux_avg = tree_map(lambda x: jnp.mean(x, axis=0), m_aux)
            
            # Update Optimizer con i gradienti mediati
            updates, new_opt_st_inner = optimizer.update(grads_avg, opt_st_inner)
            new_params_inner = optax.apply_updates(params_inner, updates)
            
            return (new_params_inner, new_opt_st_inner), (loss_avg, aux_avg)

        # Loop sui minibatch
        (new_params, new_opt_st), (batch_losses, batch_aux) = lax.scan(
            _batch_step, (params, opt_st), jnp.arange(n_minibatches)
        )
        
        epoch_metrics = {
            "loss": jnp.mean(batch_losses),
            "actor": jnp.mean(batch_aux[0]),
            "critic": jnp.mean(batch_aux[1]),
            "perc": jnp.mean(batch_aux[2]),
            "entropy": jnp.mean(batch_aux[3]),
            "log_vars": jnp.mean(batch_aux[4], axis=0)
        }
        
        return (key, new_params, new_opt_st), epoch_metrics

    (final_key, final_params, final_opt_st), history_metrics = lax.scan(
        _epoch_step, (key, network_params, opt_state), None, length=n_epochs
    )
    
    return final_params, final_opt_st, history_metrics

def jessi_multitask_rl_rollout(
    initial_network_params,
    n_parallel_envs,
    train_updates,
    random_seed,
    network_optimizer,
    total_batch_size,
    mini_batch_size,
    micro_batch_size,
    policy,
    env,
    clip_range,
    n_epochs,
    beta_entropy,
    lambda_gae,
    # ... other args ...
):
    # --- 1. Init ---
    n_steps = total_batch_size // n_parallel_envs
    
    # Init Keys
    key = random.PRNGKey(random_seed)
    key, subkey = random.split(key)
    policy_keys = random.split(subkey, n_parallel_envs)
    key, subkey = random.split(key)
    reset_keys = random.split(subkey, n_parallel_envs)
    
    # Init Env
    states, reset_keys, obses, infos, init_outcomes = env.batch_reset(reset_keys)
    env_state = (states, obses, infos)
    
    # Init Optimizer
    params = initial_network_params
    opt_state = network_optimizer.init(params)
    
    # Init Logging Lists
    logs = {
        "losses": [], 
        "returns": [],  # <--- ADDED
        "successes": [], 
        "failures": [], 
        "timeouts": [],
        "episodes": [],
        # PPO Losses
        "perception_losses": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropy_losses": [],
        # Variances
        "loss_stds": [],
        "stds": []
    }

    # --- 2. Main Training Loop (PYTHON) ---
    print(f"Starting optimized training loop for {train_updates} updates.")
    
    for update in tqdm(range(train_updates)):
        
        # A. COLLECT ROLLOUT (JIT)
        env_state, policy_keys, reset_keys, history_raw, outcomes_sum = collect_rollout_step(
            key, params, env_state, policy_keys, reset_keys, init_outcomes, policy, env, n_steps
        )
        
        # B. PROCESS BUFFER (JIT)
        current_obs, current_infos = env_state[1], env_state[2]
        buffer = process_buffer_and_gae(
            params, current_obs, current_infos, history_raw, policy, policy.gamma, lambda_gae
        )
        
        # C. PPO UPDATE (JIT)
        key, subkey = random.split(key)
        params, opt_state, metrics = run_ppo_update(
            subkey, params, opt_state, buffer, policy, network_optimizer, 
            mini_batch_size, 
            micro_batch_size, 
            n_epochs, clip_range, beta_entropy
        )
        
        metrics["loss"].block_until_ready()

        # ... (dentro jessi_multitask_rl_rollout loop) ...

        # D. LOGGING (CPU/Python)
        
        # --- 1. Returns & Outcomes (Invariato) ---
        batch_mean_return = jnp.mean(jnp.sum(history_raw["rewards"], axis=0))
        logs["returns"].append(float(batch_mean_return))

        n_succ = jnp.sum(outcomes_sum["success"])
        n_coll_hum = jnp.sum(outcomes_sum["collision_with_human"])
        n_coll_obs = jnp.sum(outcomes_sum["collision_with_obstacle"])
        n_timeout = jnp.sum(outcomes_sum["timeout"])
        
        ep_count = n_succ + n_coll_hum + n_coll_obs + n_timeout
        n_fail = n_coll_hum + n_coll_obs

        logs["successes"].append(int(n_succ))
        logs["failures"].append(int(n_fail))
        logs["timeouts"].append(int(n_timeout))
        logs["episodes"].append(int(ep_count))
        
        # --- 2. PPO Metrics (FIXED) ---
        # metrics["loss"] ha shape (n_epochs,). Facciamo la media.
        logs["losses"].append(float(jnp.mean(metrics["loss"])))
        logs["perception_losses"].append(float(jnp.mean(metrics["perc"])))
        logs["actor_losses"].append(float(jnp.mean(metrics["actor"])))
        logs["critic_losses"].append(float(jnp.mean(metrics["critic"])))
        logs["entropy_losses"].append(float(jnp.mean(metrics["entropy"])))

        # --- 3. Variances (FIXED) ---
        # metrics["log_vars"] ha shape (n_epochs, 3). Facciamo la media sulle epoche (axis=0).
        avg_log_vars = jnp.mean(metrics["log_vars"], axis=0) # Shape (3,)
        current_loss_stds = jnp.exp(0.5 * avg_log_vars)
        logs["loss_stds"].append(current_loss_stds) 
        
        # Action Std (Invariato - questo viene dal rollout, non dalle epoche PPO)
        avg_action_std = jnp.mean(history_raw["stds"], axis=(0,1))  # Media su Steps e Envs
        logs["stds"].append(avg_action_std)
        # ...
        
        if update % 1 == 0:
            print(f"Upd {update} | Ret: {logs['returns'][-1]:.2f} | Loss: {logs['losses'][-1]:.4f} | Succ: {logs['successes'][-1]} | Fail: {logs['failures'][-1]} | Timeouts: {logs['timeouts'][-1]}")

    return params, logs