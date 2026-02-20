from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import os
import pickle
import optax
from matplotlib import rc, rcParams
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.jessi import JESSI
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_rollouts import jessi_multitask_rl_rollout

network_name = 'jessi_multitask_rl_out.pkl'
finetune_network_name = 'jessi_finetuned_rl_out.pkl'
### Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 1.0
kinematics = "unicycle"
lidar_angular_range = jnp.pi * 70 / 180 
lidar_max_dist = 10.
lidar_num_rays = 100
scenario = "hybrid_scenario"
hybrid_scenario_subset = jnp.array([0,1,2,3,4,6,7,8,9])  # Exclude circular_crossing_with_static_obstacles and corner_traffic
n_humans = 3
n_obstacles = 3
humans_policy = 'hsfm'
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

### JESSI FINE TUNE: MULTI-TASK REINFORCEMENT LEARNING
if not os.path.exists(os.path.join(os.path.dirname(__file__), finetune_network_name)):
    print(f"\nSTARTING JESSI FINE TUNE RL TRAINING\nParallel envs {training_hyperparams['rl_parallel_envs']}\nSteps per env {training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_parallel_envs']}\nTotal batch size {training_hyperparams['rl_total_batch_size']}\nMini-batch size {training_hyperparams['rl_mini_batch_size']}\nBatches per update {training_hyperparams['rl_num_batches']}\nMicro-batch size {training_hyperparams['rl_micro_batch_size']}\nTraining updates {training_hyperparams['rl_training_updates']}\nEpochs per update {training_hyperparams['rl_num_epochs']}\n")
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
        'lidar_noise_fixed_std': 0.02, 
        'lidar_noise_proportional_std': 0.02,
        'lidar_salt_and_pepper_prob': 0.05,
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
        n_stack_for_action_space_bounding=3,
        beam_dropout_rate=0.2
    )
    # Load pre-trained weights
    with open(os.path.join(os.path.dirname(__file__), network_name), 'rb') as f:
        il_network_params, _, _ = pickle.load(f)
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
                    peak_value=training_hyperparams['rl_learning_rate'] * 0.1,
                    end_value=training_hyperparams['rl_learning_rate_final'] * 0.05, 
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
    with open(os.path.join(os.path.dirname(__file__),finetune_network_name), 'wb') as f:
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
    figure.savefig(os.path.join(os.path.dirname(__file__),finetune_network_name.replace('.pkl', '.eps')), format='eps')
else:
    print(f"JESSI FINE TUNE RL training already done and saved... Remove '{finetune_network_name}' to retrain.")
