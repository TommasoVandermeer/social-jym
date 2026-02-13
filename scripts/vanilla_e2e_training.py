from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import optax
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.vanilla_e2e import VanillaE2E
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.vanilla_e2e_rollouts import vanilla_e2e_rl_rollout

unbounded_policy_nn_name = 'vanilla_e2e_il_out.pkl'
unbounded_network_name = 'vanilla_e2e_rl_out.pkl'
bounded_policy_nn_name = 'bounded_vanilla_e2e_il_out'
bounded_network_name = 'bounded_vanilla_e2e_rl_out.pkl'
### Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 1.0
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 100
scenario = "hybrid_scenario"
hybrid_scenario_subset = jnp.array([0,1,2,3,4,6])  # Exclude circular_crossing_with_static_obstacles and corner_traffic
n_humans = 5
n_obstacles = 3
humans_policy = 'hsfm'
### IL Hyperparameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 500_000  # Number of labeled examples to train Perception network
n_parallel_envs = 1000  # Number of parallel environments to simulate to generate the dataset
policy_learning_rate = 0.005
policy_batch_size = 200
policy_n_epochs = 10 # Just a few to not overfit on DIR-SAFE data (if action space becomes too deterministic there will be no exploration in RL fine-tuning)
### RL Hyperparameters
rl_n_parallel_envs = 500 
rl_training_updates = 500
training_hyperparams = {
    'random_seed': 0,
    'n_humans': n_humans, 
    'n_obstacles': n_obstacles,
    'rl_training_updates': rl_training_updates,
    'rl_parallel_envs': rl_n_parallel_envs,
    'rl_learning_rate': 1e-4, # 3e-4
    'rl_learning_rate_final': 1e-5, # 2e-4
    'rl_total_batch_size': 50_000, # 50_000 Nsteps for env = rl_total_batch_size / rl_parallel_envs
    'rl_mini_batch_size': 2_000, # 2_000 Mini-batch size for each model update
    'rl_micro_batch_size': 1_000, # 1_000 # Micro-batch size for gradient accumulation 
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 6, # 6
    'rl_beta_entropy': 5e-4, # 1e-4
    'lambda_gae': 0.95, # 0.95
    # 'humans_policy': 'hsfm', It is set by default in the LaserNav env
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': hybrid_scenario_subset,
    'reward_function': 'lasernav_reward1',
    'gradient_norm_scale': 1, # Scale the gradient norm by this value
    'target_kl': 0.01,  # Target KL divergence for early stopping in each update
}
training_hyperparams['rl_num_batches'] = training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_mini_batch_size']


### PRE-TRAIN UNBOUNDED POLICY NETWORK (IMITATION LEARNING)
if not os.path.exists(os.path.join(os.path.dirname(__file__), unbounded_policy_nn_name)):
    # CREATE ACTOR INPUTS DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl')):
        print("Execute jessi_training.py to generate the initial dataset for Imitation Learning")
    else:
        # Load actor inputs
        with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'rb') as f:
            # controller_dataset = {
            #     "observations": 
            #     "rc_robot_goals":
            #     "actor_actions":
            #     "returns":
            # }
            controller_dataset = pickle.load(f)
    # INITIALIZE VanillaE2E
    policy = VanillaE2E(
        v_max=robot_vmax, 
        dt=robot_dt, 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        action_space_bounding=False,
    )
    # Initialize actor network
    actor_critic_params = policy.init_nn(random.PRNGKey(random_seed))
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
    @loop_tqdm(policy_n_epochs, desc="Training UNBOUNDED Vanilla-E2E policy network")
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
            inputs0, inputs1 = vmap(policy.compute_actor_inputs)(batch["observations"], batch["rc_robot_goals"])
            train_batch = {
                "inputs0": inputs0,
                "inputs1": inputs1,
                "actor_actions": batch["actor_actions"],
                "returns": batch["returns"],
            }
            # Update parameters
            actor_critic_params, optimizer_state, loss, actor_loss, critic_loss = policy.update_il(
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
    with open(os.path.join(os.path.dirname(__file__), unbounded_policy_nn_name), 'wb') as f:
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
    fig.savefig(os.path.join(os.path.dirname(__file__), unbounded_policy_nn_name.replace('.pkl', '.eps')), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), unbounded_policy_nn_name), 'rb') as f:
        actor_critic_params = pickle.load(f)

### PRE-TRAIN BOUNDED POLICY NETWORK (IMITATION LEARNING)
if not os.path.exists(os.path.join(os.path.dirname(__file__), bounded_policy_nn_name)):
    # CREATE ACTOR INPUTS DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl')):
        print("Execute jessi_training.py to generate the initial dataset for Imitation Learning")
    else:
        # Load actor inputs
        with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'rb') as f:
            # controller_dataset = {
            #     "observations": 
            #     "rc_robot_goals":
            #     "actor_actions":
            #     "returns":
            # }
            controller_dataset = pickle.load(f)
    # INITIALIZE VanillaE2E
    policy = VanillaE2E(
        v_max=robot_vmax, 
        dt=robot_dt, 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        action_space_bounding=True,
    )
    # Initialize actor network
    actor_critic_params = policy.init_nn(random.PRNGKey(random_seed))
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
    @loop_tqdm(policy_n_epochs, desc="Training UNBOUNDED Vanilla-E2E policy network")
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
            inputs0, inputs1 = vmap(policy.compute_actor_inputs)(batch["observations"], batch["rc_robot_goals"])
            train_batch = {
                "inputs0": inputs0,
                "inputs1": inputs1,
                "actor_actions": batch["actor_actions"],
                "returns": batch["returns"],
            }
            # Update parameters
            actor_critic_params, optimizer_state, loss, actor_loss, critic_loss = policy.update_il(
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
    with open(os.path.join(os.path.dirname(__file__), bounded_policy_nn_name), 'wb') as f:
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
    fig.savefig(os.path.join(os.path.dirname(__file__), bounded_policy_nn_name.replace('.pkl', '.eps')), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), bounded_policy_nn_name), 'rb') as f:
        actor_critic_params = pickle.load(f)

### RL UNBOUNDED POLICY NETWORK
if not os.path.exists(os.path.join(os.path.dirname(__file__), unbounded_network_name)):
    print(f"\nSTARTING UNBOUNDED VANILLA-E2E RL TRAINING\nParallel envs {training_hyperparams['rl_parallel_envs']}\nSteps per env {training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_parallel_envs']}\nTotal batch size {training_hyperparams['rl_total_batch_size']}\nMini-batch size {training_hyperparams['rl_mini_batch_size']}\nBatches per update {training_hyperparams['rl_num_batches']}\nMicro-batch size {training_hyperparams['rl_micro_batch_size']}\nTraining updates {training_hyperparams['rl_training_updates']}\nEpochs per update {training_hyperparams['rl_num_epochs']}\n")
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
    policy = VanillaE2E(
        robot_radius=env_params['robot_radius'],
        v_max=robot_vmax, 
        dt=robot_dt, 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        action_space_bounding=False, 
    )
    # Load pre-trained weights
    with open(os.path.join(os.path.dirname(__file__), unbounded_policy_nn_name), 'rb') as f:
        il_network_params = pickle.load(f)
    # Initialize optimizer
    network_optimizer = optax.chain(
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
        'target_kl': training_hyperparams['target_kl'],
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = vanilla_e2e_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    with open(os.path.join(os.path.dirname(__file__),unbounded_network_name), 'wb') as f:
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
    # Plot approximate KL divergence during RL
    ax[1,2].grid()
    ax[1,2].set(
        xlabel='Training Update', 
        ylabel=f'KL div. ({window} upd. window)', 
        title='Approx. KL div.'
    )
    ax[1,2].plot(
        jnp.arange(len(approx_kl_during_rl)-(window-1))+window, 
        jnp.convolve(approx_kl_during_rl, jnp.ones(window,), 'valid') / window,
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
    figure.savefig(os.path.join(os.path.dirname(__file__),unbounded_network_name.replace('.pkl', '.eps')), format='eps')
else:
    print(f"UNBOUNDED VANILLA-E2E training already done and saved... Remove '{unbounded_network_name}' to retrain.")


### RL BOUNDED POLICY NETWORK
if not os.path.exists(os.path.join(os.path.dirname(__file__), bounded_network_name)):
    print(f"\nSTARTING BOUNDED VANILLA-E2E RL TRAINING\nParallel envs {training_hyperparams['rl_parallel_envs']}\nSteps per env {training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_parallel_envs']}\nTotal batch size {training_hyperparams['rl_total_batch_size']}\nMini-batch size {training_hyperparams['rl_mini_batch_size']}\nBatches per update {training_hyperparams['rl_num_batches']}\nMicro-batch size {training_hyperparams['rl_micro_batch_size']}\nTraining updates {training_hyperparams['rl_training_updates']}\nEpochs per update {training_hyperparams['rl_num_epochs']}\n")
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
    policy = VanillaE2E(
        robot_radius=env_params['robot_radius'],
        v_max=robot_vmax, 
        dt=robot_dt, 
        lidar_num_rays=lidar_num_rays, 
        lidar_max_dist=lidar_max_dist,
        lidar_angular_range=lidar_angular_range,
        n_stack=n_stack,
        action_space_bounding=True, 
    )
    # Load pre-trained weights
    with open(os.path.join(os.path.dirname(__file__), bounded_policy_nn_name), 'rb') as f:
        il_network_params = pickle.load(f)
    # Initialize optimizer
    network_optimizer = optax.chain(
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
        'target_kl': training_hyperparams['target_kl'],
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = vanilla_e2e_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    with open(os.path.join(os.path.dirname(__file__),bounded_network_name), 'wb') as f:
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
    # Plot approximate KL divergence during RL
    ax[1,2].grid()
    ax[1,2].set(
        xlabel='Training Update', 
        ylabel=f'KL div. ({window} upd. window)', 
        title='Approx. KL div.'
    )
    ax[1,2].plot(
        jnp.arange(len(approx_kl_during_rl)-(window-1))+window, 
        jnp.convolve(approx_kl_during_rl, jnp.ones(window,), 'valid') / window,
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
    figure.savefig(os.path.join(os.path.dirname(__file__),bounded_network_name.replace('.pkl', '.eps')), format='eps')
else:
    print(f"BOUNDED VANILLA-E2E training already done and saved... Remove '{bounded_network_name}' to retrain.")