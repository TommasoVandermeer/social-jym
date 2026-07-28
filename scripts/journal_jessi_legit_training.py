from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import optax
from matplotlib import rc, rcParams
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.jessi import JESSI, ABLATIONS
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_rollouts import jessi_multitask_rl_rollout

random_seed = 0
perception_nn_name = 'pre_perception_network.pkl'
### Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 1.0
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 100
scenario = "hybrid_scenario"
n_stack = 5
hybrid_scenario_subset = jnp.array([0,1,2,3,4,6])  # Exclude circular_crossing_with_static_obstacles and corner_traffic
n_humans = 5
n_obstacles = 3
humans_policy = 'hsfm'
### IL Hyperparameters
policy_learning_rate = 0.005
policy_batch_size = 200
policy_n_epochs = 30 # Just a few to not overfit on DIR-SAFE data (if action space becomes too deterministic there will be no exploration in RL fine-tuning)
### MULTI-TASK RL Hyperparameters
rl_n_parallel_envs = 300 
rl_training_updates = 500
training_hyperparams = {
    'random_seed': 0,
    'n_humans': n_humans, 
    'n_obstacles': n_obstacles,
    'rl_training_updates': rl_training_updates,
    'rl_parallel_envs': rl_n_parallel_envs,
    'rl_learning_rate': 1e-4, # 3e-4
    'rl_learning_rate_final': 1e-5, # 2e-4
    'rl_total_batch_size': 30_000, # 50_000 Nsteps for env = rl_total_batch_size / rl_parallel_envs
    'rl_mini_batch_size': 1_000, # 2_000 Mini-batch size for each model update
    'rl_micro_batch_size': 500, # 1_000 # Micro-batch size for gradient accumulation 
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 6, # 6
    'rl_beta_entropy': 5e-4, # 1e-4
    'lambda_gae': 0.95, # 0.95
    # 'humans_policy': 'hsfm', It is set by default in the LaserNav env
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': hybrid_scenario_subset,
    'reward_function': 'lasernav_reward1',
    'gradient_norm_scale': 1, # Scale the gradient norm by this value
    'safety_loss': False,  # Whether to include safety loss in the RL training
    'target_kl': None,  # Target KL divergence for early stopping in each update
}
training_hyperparams['rl_num_batches'] = training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_mini_batch_size']

if not os.path.exists(os.path.join(os.path.dirname(__file__), f"jessi_il_out_legit.pkl")):
    ### PRE-TRAIN POLICY NETWORK (IMITATION LEARNING)
    # Load trained perception parameters
    with open(os.path.join(os.path.dirname(__file__), perception_nn_name), 'rb') as f:
        encoder_params = pickle.load(f)
    # CREATE ACTOR INPUTS DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl')):
        # LOAD DATASETs
        with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'rb') as f:
            raw_data = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
            robot_centric_data = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'final_hcg_training_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        # Compute actor-critic inputs for the entire dataset
        controller_dataset = {
            "observations": dataset['observations'],
            "rc_robot_goals": robot_centric_data["rc_robot_goals"],
            "actor_actions": raw_data["robot_actions"],
            "returns": raw_data["returns"],
        }
        # Save actor inputs
        with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'wb') as f:
            pickle.dump(controller_dataset, f)
        # FREE UNUSED MEMORY
        del dataset
        del robot_centric_data
        del raw_data
    else:
        # Load actor inputs
        with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'rb') as f:
            controller_dataset = pickle.load(f)
    # INITIALIZE ACTOR NETWORK
    jessi = JESSI(
        v_max=robot_vmax, 
        dt=robot_dt, 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack, 
        legit=True,
    )
    print(f"\nSTARTING JESSI-MULTITASK IL TRAINING - LEGIT\n")
    # Initialize actor network
    _, actor_critic_params, _ = jessi.init_nns(random.PRNGKey(random_seed))
    # Count network parameters
    def count_params(actor_critic_params):
        return sum(jnp.prod(jnp.array(p.shape)) for layer in actor_critic_params.values() for p in layer.values())
    n_params = count_params(actor_critic_params)
    print(f"# Controller network parameters: {n_params}")
    # TRAINING LOOP
    # Initialize optimizer and its state
    optimizer = optax.sgd(learning_rate=policy_learning_rate, momentum=0.9)
    optimizer_state = optimizer.init(actor_critic_params)
    n_data = controller_dataset["observations"].shape[0]
    n_train_batches = n_data // policy_batch_size
    print(f"# Training dataset size: {controller_dataset['observations'].shape[0]} experiences")
    @jit
    def batch_augment_data(
        batch:dict,
        keys:random.PRNGKey,
        base_lidar_noise_std:float = 0.01, # 1cm base noise
        proportional_lidar_noise_std:float = 0.01, # 1% proportional noise
        beam_dropout_prob:float = 0.03, # 3% beams dropout
    ) -> dict:
        return vmap(augment_data, in_axes=(0, 0, None, None, None))(batch, keys, base_lidar_noise_std, proportional_lidar_noise_std, beam_dropout_prob)
    @jit 
    def augment_data(
        data:dict,
        key:random.PRNGKey,
        base_lidar_noise_std:float,
        proportional_lidar_noise_std:float,
        beam_dropout_prob:float,
    ) -> dict:
        # data = {
        #     "inputs": shape (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
        #               each token: [norm_dist, hit, x, y, sin_fixed_theta, cos_fixed_theta, delta_t]
        #     "targets": {
        #         "gt_mask": shape (n_humans,),
        #         "gt_poses": shape (n_humans, 2),
        #         "gt_vels": shape (n_humans, 2),
        #     }
        # }
        input_key, beam_dropout_key = random.split(key, 2)
        ## Gaussian noise to LiDAR scans + Beam dropout
        raw_distances = data['inputs'][:,:,0] * jessi.max_beam_range  # (n_stack, lidar_num_rays)
        sigma = base_lidar_noise_std + proportional_lidar_noise_std * raw_distances  # (n_stack, lidar_num_rays)
        noise = random.normal(input_key, shape=raw_distances.shape) * sigma * data['inputs'][:,:,1]  # (n_stack, lidar_num_rays)
        noisy_distances = jnp.clip(raw_distances + noise, 0., jessi.max_beam_range) # (n_stack, lidar_num_rays)
        is_dropout = random.bernoulli(beam_dropout_key, p=beam_dropout_prob, shape=raw_distances.shape)
        noisy_distances = jnp.where(is_dropout, jessi.max_beam_range, noisy_distances)  # (n_stack, lidar_num_rays)
        new_hit = jnp.where(noisy_distances < jessi.max_beam_range, 1.0, 0.0) * (1.0 - is_dropout)  # (n_stack, lidar_num_rays)
        x = noisy_distances * data['inputs'][:,:,4]  # (n_stack, lidar_num_rays)
        y = noisy_distances * data['inputs'][:,:,5]  # (n_stack, lidar_num_rays)
        data['inputs'] = data['inputs'].at[:,:,0].set(noisy_distances / jessi.max_beam_range)
        data['inputs'] = data['inputs'].at[:,:,1].set(new_hit)
        data['inputs'] = data['inputs'].at[:,:,2].set(x)
        data['inputs'] = data['inputs'].at[:,:,3].set(y)
        return data
    @loop_tqdm(policy_n_epochs, desc="Training Controller network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        dataset, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = epoch_for_val
        # Shuffle dataset at the beginning of the epoch
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_data)
        shuffled_indexes = random.permutation(shuffle_key, indexes)
        epoch_data = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(shuffled_indexes, dataset)
        # Batch loop
        @jit
        def _batch_loop(
            j:int,
            batch_for_val:tuple
        ) -> tuple:
            epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(policy_batch_size) + j * policy_batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, epoch_data)
            # Compute training batch
            perception_input, last_lidar_point_clouds = vmap(jessi.compute_perception_input)(batch["observations"])
            ## DATA AUGMENTATION
            noisy_inputs = batch_augment_data(
                {
                    "inputs": perception_input,
                },
                random.split(random.PRNGKey(i * n_train_batches + j), policy_batch_size),
            )
            hcgs, scan_embeddings, _, _ = jessi.perception.apply(
                encoder_params, 
                None, 
                noisy_inputs["inputs"],
            )
            bounding_parameters = vmap(jessi.bound_action_space)(last_lidar_point_clouds)
            actor_input = vmap(jessi.compute_actor_input)(
                batch["observations"][:,:,:11],
                hcgs,
                bounding_parameters,
                batch["rc_robot_goals"],
            )
            train_batch = {
                "actor_inputs": actor_input,
                "scan_embeddings": scan_embeddings,
                "actor_actions": batch["actor_actions"],
                "returns": batch["returns"],
            }
            # Update parameters
            actor_critic_params, optimizer_state, loss, actor_loss, critic_loss = jessi.update_il(
                actor_critic_params, 
                optimizer, 
                optimizer_state,
                train_batch,
            )
            # Save loss
            losses = losses.at[i,j].set(loss)
            actor_losses = actor_losses.at[i,j].set(actor_loss)
            critic_losses = critic_losses.at[i,j].set(critic_loss)
            return epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses
        n_batches = n_data // policy_batch_size
        _, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
            0,
            n_batches,
            _batch_loop,
            (epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses)
        )
        return dataset, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses
    # Epoch loop
    _, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
        0,
        policy_n_epochs,
        _epoch_loop,
        (controller_dataset, actor_critic_params, optimizer_state, jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))), jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))), jnp.zeros((policy_n_epochs, int(n_data // policy_batch_size))))
    )
    # Save trained parameters
    with open(os.path.join(os.path.dirname(__file__), f"jessi_il_out_legit.pkl"), 'wb') as f:
        pickle.dump(actor_critic_params, f)
    # FREE MEMORY
    del controller_dataset
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    avg_actor_losses = jnp.mean(actor_losses, axis=1)
    avg_critic_losses = jnp.mean(critic_losses, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(jnp.arange(policy_n_epochs), avg_losses, c='red')
    ax[1].plot(jnp.arange(policy_n_epochs), avg_actor_losses, c='green')
    ax[2].plot(jnp.arange(policy_n_epochs), avg_critic_losses, c='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss (with regularization)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Actor Loss (weighted)")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Critic Loss (weighted)")
    fig.savefig(os.path.join(os.path.dirname(__file__), f"jessi_il_out_legit.eps"), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), f"jessi_il_out_legit.pkl"), 'rb') as f:
        actor_critic_params = pickle.load(f)

### JESSI-MULTITASK: MULTI-TASK REINFORCEMENT LEARNING
if not os.path.exists(os.path.join(os.path.dirname(__file__), f"jessi_multitask_rl_out_legit.pkl")):
    print(f"\nSTARTING JESSI-MULTITASK RL TRAINING\nParallel envs {training_hyperparams['rl_parallel_envs']}\nSteps per env {training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_parallel_envs']}\nTotal batch size {training_hyperparams['rl_total_batch_size']}\nMini-batch size {training_hyperparams['rl_mini_batch_size']}\nBatches per update {training_hyperparams['rl_num_batches']}\nMicro-batch size {training_hyperparams['rl_micro_batch_size']}\nTraining updates {training_hyperparams['rl_training_updates']}\nEpochs per update {training_hyperparams['rl_num_epochs']}\n")
    # Initialize reward function
    if training_hyperparams['reward_function'] == 'lasernav_reward1': 
        reward_function = Reward1(
            robot_radius=0.3,
            collision_with_humans_penalty=-.5,
        )
    else:
        raise ValueError(f"{training_hyperparams['reward_function']} is not a valid reward function")
    # Environment parameters
    env_params = {
        'robot_radius': 0.3,
        'n_humans': training_hyperparams['n_humans'],
        'n_obstacles': training_hyperparams['n_obstacles'],
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': False,
        'scenario': training_hyperparams['scenario'],
        'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
        'circle_radius': 7,
        'reward_function': reward_function,
        'kinematics': 'unicycle',
        'lidar_noise': True,
    }
    # Initialize environment
    env = LaserNav(**env_params)
    _, _, obs, info, _ = env.reset(random.PRNGKey(training_hyperparams['random_seed']))
    # Initialize robot policy and vnet params
    policy = JESSI(
        robot_radius=env_params['robot_radius'],
        v_max=robot_vmax, 
        dt=env_params['robot_dt'], 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        beam_dropout_rate=0.2,
        legit=True,
    )
    # Load pre-trained weights
    with open(os.path.join(os.path.dirname(__file__), perception_nn_name), 'rb') as f:
        il_encoder_params = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), f"jessi_il_out_legit.pkl"), 'rb') as f:
        il_actor_params = pickle.load(f)
    il_network_params = policy.merge_nns_params(il_encoder_params, il_actor_params)
    def label_params(params):
        labels = {}
        for module_name, module_params in params.items():
            if policy.perception_name in module_name.lower(): 
                label = 'perception'
            elif policy.actor_critic_name in module_name.lower():
                label = 'actor_critic'
            labels[module_name] = {k: label for k in module_params.keys()}
        return labels
    network_optimizer = optax.multi_transform(
        {
        'perception': optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(
                learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                    init_value=0.,
                    peak_value=training_hyperparams['rl_learning_rate'],
                    end_value=training_hyperparams['rl_learning_rate_final'], 
                    warmup_steps=(training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches']) // 10,
                    decay_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
                ), 
                eps=1e-7, 
            ),
        ),
        'actor_critic': optax.chain(
            optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    init_value=training_hyperparams['rl_learning_rate'], 
                    end_value=training_hyperparams['rl_learning_rate_final'], 
                    transition_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
                    transition_begin=0
                ), 
                eps=1e-7, 
            ),
        )
        },
        label_params(il_network_params)
    )
    # Initialize RL rollout params
    rl_rollout_params = {
        'initial_network_params': il_network_params,
        'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
        'train_updates': training_hyperparams['rl_training_updates'],
        'random_seed': training_hyperparams['random_seed'],
        'network_optimizer': network_optimizer,
        'total_batch_size': training_hyperparams['rl_total_batch_size'],
        'mini_batch_size': training_hyperparams['rl_mini_batch_size'],
        'micro_batch_size': training_hyperparams['rl_micro_batch_size'],
        'policy': policy,
        'env': env,
        'clip_range': training_hyperparams['rl_clip_frac'],
        'n_epochs': training_hyperparams['rl_num_epochs'],
        'beta_entropy': training_hyperparams['rl_beta_entropy'],
        'lambda_gae': training_hyperparams['lambda_gae'],
        'safety_loss': training_hyperparams['safety_loss'],
        'training_type': "multitask",
        'target_kl': training_hyperparams['target_kl'],
        'debugging': False,
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = jessi_multitask_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    with open(os.path.join(os.path.dirname(__file__), f"jessi_multitask_rl_out_legit.pkl"), 'wb') as f:
        pickle.dump(rl_out, f)
    final_params, _, metrics = rl_out
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            processed_metrics[key] = jnp.array(value)
        if isinstance(value, dict):
            processed_metrics[key] = tree_map(lambda x: jnp.array(x), value)
    # Other metrics
    losses = processed_metrics['losses']
    perception_losses = processed_metrics['perception_losses']
    actor_losses = processed_metrics['actor_losses']
    critic_losses = processed_metrics['critic_losses']
    entropy_losses = processed_metrics['entropy_losses']
    returns_during_rl = processed_metrics['returns']
    success_during_rl = processed_metrics['successes']
    failure_during_rl = processed_metrics['failures']
    timeout_during_rl = processed_metrics['timeouts']
    episodes_during_rl = processed_metrics['episodes']
    stds_during_rl = processed_metrics['stds']
    grad_norms_during_rl = processed_metrics['grad_norm']
    collisions_humans_during_rl = processed_metrics['collisions_humans']
    collisions_obstacles_during_rl = processed_metrics['collisions_obstacles']
    times_to_goal_during_rl = processed_metrics['times_to_goal']
    approx_kl_during_rl = processed_metrics['approx_kl']
    episode_count = jnp.sum(episodes_during_rl)
    window = 10 if rl_training_updates > 1000 else 1
    ## Plot RL training stats
    from matplotlib import rc
    font = {'weight' : 'regular',
            'size'   : 18}
    rc('font', **font)
    figure, ax = plt.subplots(4,3,figsize=(15,15))
    figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95, right=0.95, left=0.1, wspace=0.35)
    # Plot returns during RL
    ax[0,0].grid()
    ax[0,0].set(
        xlabel='Training Update', 
        ylabel=f'Return ({window} upd. window)', 
        title='Return'
    )
    ax[0,0].plot(
        jnp.arange(len(returns_during_rl)-(window-1))+window, 
        jnp.convolve(returns_during_rl, jnp.ones(window,), 'valid') / window,
    )
    # Plot success, failure, and timeout rates during RL
    success_rate_during_rl = success_during_rl / episodes_during_rl
    failure_rate_during_rl = failure_during_rl / episodes_during_rl
    timeout_rate_during_rl = timeout_during_rl / episodes_during_rl
    ax[0,1].grid()
    ax[0,1].set(
        xlabel='Training Update', 
        ylabel=f'Rate ({window} upd. window)', 
        title='Success, Failure, and Timeout rates',
        ylim=(-0.1,1.1)
    )
    ax[0,1].plot(
        jnp.arange(len(success_rate_during_rl)-(window-1))+window, 
        jnp.convolve(success_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Success rate',
        color='g',
    )
    ax[0,1].plot(
        jnp.arange(len(failure_rate_during_rl)-(window-1))+window, 
        jnp.convolve(failure_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Failure rate',
        color='r',
    )
    ax[0,1].plot(
        jnp.arange(len(timeout_rate_during_rl)-(window-1))+window, 
        jnp.convolve(timeout_rate_during_rl, jnp.ones(window,), 'valid') / window,
        label='Timeout rate',
        color='yellow',
    )
    # ax[0,1].legend()
    # Plot time to goal during RL
    ax[0,2].grid()
    ax[0,2].set(
        xlabel='Training Update',
        ylabel=f'Time  ({window} upd. window)',
        title='Time to Goal',
    )
    ax[0,2].plot(
        jnp.arange(len(times_to_goal_during_rl)-(window-1))+window, 
        jnp.convolve(times_to_goal_during_rl, jnp.ones(window,), 'valid') / window,
    )
    # Plot actor loss during RL
    ax[1,0].grid()
    ax[1,0].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Actor Loss'
    )
    ax[1,0].plot(
        jnp.arange(len(actor_losses)-(window-1))+window, 
        jnp.convolve(actor_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot critic loss during RL
    ax[1,1].grid()
    ax[1,1].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Critic Loss'
    )
    ax[1,1].plot(
        jnp.arange(len(critic_losses)-(window-1))+window, 
        jnp.convolve(critic_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot perception loss during RL
    ax[1,2].grid()
    ax[1,2].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Perception Loss'
    )
    ax[1,2].plot(
        jnp.arange(len(perception_losses)-(window-1))+window, 
        jnp.convolve(perception_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot entropy loss during RL
    ax[2,0].grid()
    ax[2,0].set(
        xlabel='Training Update', 
        ylabel=f'Loss ({window} upd. window)', 
        title='Entropy Loss'
    )
    ax[2,0].plot(
        jnp.arange(len(entropy_losses)-(window-1))+window, 
        jnp.convolve(entropy_losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot stds[0] during RL
    ax[2,1].grid()
    ax[2,1].set(
        xlabel='Training Update',
        ylabel='$\sigma(v)$',
        title='Std velocity',
        ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
    )
    ax[2,1].plot(
        jnp.arange(len(stds_during_rl)),
        stds_during_rl[:,0],
    )
    # Plot stds[1] during RL
    ax[2,2].grid()
    ax[2,2].set(
        xlabel='Training Update',
        ylabel='$\sigma(\omega)$',
        title='Std ang. vel.',
        ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
    )
    ax[2,2].plot(
        jnp.arange(len(stds_during_rl)),
        stds_during_rl[:,1],
    )
    # Plot actor loss std during RL
    ax[3,0].grid()
    ax[3,0].set(
        xlabel='Training Update',
        ylabel='Average Norm',
        title='Gradients L2 norm',
    )
    ax[3,0].plot(
        jnp.arange(len(grad_norms_during_rl[:])),
        grad_norms_during_rl,
    )
    # Plot Total Loss during RL
    ax[3,1].grid()
    ax[3,1].set(
        xlabel='Training Update',
        ylabel=f'Loss  ({window} upd. window)',
        title='Total Loss',
    )
    ax[3,1].plot(
        jnp.arange(len(losses)-(window-1))+window, 
        jnp.convolve(losses, jnp.ones(window,), 'valid') / window,
    )
    # Plot Collisions with humans and obstacles during RL
    ax[3,2].grid()
    ax[3,2].set(
        xlabel='Training Update',
        ylabel='Collisions',
        title='Colls. hum/obs',
    )
    ax[3,2].plot(
        jnp.arange(len(collisions_humans_during_rl)),
        collisions_humans_during_rl/episodes_during_rl,
        label='Collisions humans',
        color='blue',
    )
    ax[3,2].plot(
        jnp.arange(len(collisions_obstacles_during_rl)),
        collisions_obstacles_during_rl/episodes_during_rl,
        label='Collisions obstacles',
        color='black',
    )
    figure.savefig(os.path.join(os.path.dirname(__file__), f"jessi_multitask_rl_out_legit.eps"), format='eps')
else:
    print(f"JESSI-MULTITASK RL training already done and saved... Remove 'jessi_multitask_rl_out_legit.pkl' to retrain.")