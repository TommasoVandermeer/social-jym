import jax.numpy as jnp
from jax import random
import os
import optax
import matplotlib.pyplot as plt
import pickle
# from matplotlib import rc, rcParams
# rc('font', weight='regular', size=20)
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42

from socialjym.envs.lasernav import LaserNav
# from socialjym.utils.aux_functions import test_k_trials, initialize_metrics_dict
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_rollouts import jessi_multitask_rl_rollout
from socialjym.policies.jessi import JESSI

### Hyperparameters
n_humans_for_tests = [5, 10, 15, 20, 25]
test_robot_visibility = [False, True]
n_trials = 100
n_parallel_envs = 2048 
training_updates = 30_000 # 30_000
rl_debugging_interval = 10
robot_vmax = 1
training_hyperparams = {
    'random_seed': 0,
    'n_humans': 5, 
    'n_obstacles': 3,
    'rl_training_updates': training_updates,
    'rl_parallel_envs': n_parallel_envs,
    'rl_learning_rate': 3e-4,
    'rl_total_batch_size': 262_144, # Nsteps for env = rl_total_batch_size / rl_parallel_envs
    'rl_mini_batch_size': 32_768, # Mini-batch size for each model update
    'rl_clip_frac': 0.2, # 0.2
    'rl_num_epochs': 4,
    'rl_beta_entropy': 1e-4, # 5e-4
    'lambda_gae': 0.95, # 0.95
    'humans_policy': 'hsfm',
    'scenario': 'hybrid_scenario',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6], jnp.int32), # Subset of the hybrid scenarios to use for training
    'reward_function': 'lasernav_reward1',
    'gradient_norm_scale': 0.5, # Scale the gradient norm by this value
}
training_hyperparams['rl_num_batches'] = training_hyperparams['rl_total_batch_size'] // training_hyperparams['rl_mini_batch_size']
print(f"STARTING RL TRAINING\n{training_hyperparams['rl_parallel_envs']} parallel envs,\ntotal batch size {training_hyperparams['rl_total_batch_size']},\nmini-batch size {training_hyperparams['rl_mini_batch_size']}\n{training_hyperparams['rl_num_batches']} batches per update\nfor a total of {training_hyperparams['rl_training_updates']} updates.")

# Initialize reward function
if training_hyperparams['reward_function'] == 'lasernav_reward1': 
    reward_function = Reward1(
        robot_radius=0.3,
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
    'humans_policy': training_hyperparams['humans_policy'],
    'circle_radius': 7,
    'reward_function': reward_function,
    'kinematics': 'unicycle',
}
# Initialize environment
env = LaserNav(**env_params)
_, _, obs, info, _ = env.reset(random.PRNGKey(training_hyperparams['random_seed']))
# Initialize robot policy and vnet params
policy = JESSI(
    env.reward_function, 
    v_max=robot_vmax, 
    dt=env_params['robot_dt'], 
)
# Load pre-trained weights
with open(os.path.join(os.path.dirname(__file__), 'perception_network.pkl'), 'rb') as f:
    il_encoder_params = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'rb') as f:
    il_actor_params = pickle.load(f)
il_network_params = policy.merge_nns_params(il_encoder_params, il_actor_params)

# Initialize RL optimizer
network_optimizer = optax.chain(
    optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
    optax.adam(
        learning_rate=optax.schedules.linear_schedule(
            init_value=training_hyperparams['rl_learning_rate'], 
            end_value=0., 
            transition_steps=training_hyperparams['rl_training_updates']*training_hyperparams['rl_num_epochs']*training_hyperparams['rl_num_batches'],
            transition_begin=0
        ), 
        eps=1e-7, 
        b1=0.9,
    ),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
    optax.sgd(
        learning_rate=training_hyperparams['rl_critic_learning_rate'], 
        momentum=0.9
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
    'policy': policy,
    'env': env,
    'clip_range': training_hyperparams['rl_clip_frac'],
    'n_epochs': training_hyperparams['rl_num_epochs'],
    'beta_entropy': training_hyperparams['rl_beta_entropy'],
    'lambda_gae': training_hyperparams['lambda_gae'],
    'debugging': True,
    'debugging_interval': rl_debugging_interval,
}

# REINFORCEMENT LEARNING ROLLOUT
if not os.path.exists(os.path.join(os.path.dirname(__file__),"rl_out.pkl")):
    rl_out = jessi_multitask_rl_rollout(**rl_rollout_params)
    # Save RL rollout output
    with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'wb') as f:
        pickle.dump(rl_out, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"rl_out.pkl"), 'rb') as f:
        rl_out = pickle.load(f)
print(f"Total episodes simulated: {jnp.sum(rl_out['aux_data']['episodes'])}")

# Save the training returns
rl_network_params = rl_out['network_params']
returns_during_rl = rl_out['aux_data']['returns']  
losses = rl_out['aux_data']['losses']
perception_losses = rl_out['aux_data']['perception_losses']
actor_losses = rl_out['aux_data']['actor_losses']
critic_losses = rl_out['aux_data']['critic_losses']
entropy_losses = rl_out['aux_data']['entropy_losses']
loss_stds = rl_out['aux_data']['loss_stds']
success_during_rl = rl_out['aux_data']['successes']
failure_during_rl = rl_out['aux_data']['failures']
timeout_during_rl = rl_out['aux_data']['timeouts']
episodes_during_rl = rl_out['aux_data']['episodes']
stds_during_rl = rl_out['aux_data']['stds']
episode_count = jnp.sum(episodes_during_rl)
window = 500 if training_updates > 1000 else 50

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
success_rate_during_rl = success_during_rl / rl_out['aux_data']['episodes']
failure_rate_during_rl = failure_during_rl / rl_out['aux_data']['episodes']
timeout_rate_during_rl = timeout_during_rl / rl_out['aux_data']['episodes']
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
ax[0,1].legend()
# Plot total loss during RL
ax[0,2].grid()
ax[0,2].set(
    xlabel='Training Update',
    ylabel=f'Loss  ({window} upd. window)',
    title='Total Loss',
)
ax[0,2].plot(
    jnp.arange(len(losses)-(window-1))+window, 
    jnp.convolve(losses, jnp.ones(window,), 'valid') / window,
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
# Plot actor loss std during RL
ax[1,2].grid()
ax[1,2].set(
    xlabel='Training Update',
    ylabel=r'$\sigma(\mathcal{L}_{act})$',
    title='Std actor loss',
)
ax[1,2].plot(
    jnp.arange(len(loss_stds[:,0])),
    loss_stds[:,0],
)
# Plot entropy loss during RL
entropy_window = window // 10
ax[2,0].grid()
ax[2,0].set(
    xlabel='Training Update', 
    ylabel=f'Loss ({entropy_window} upd. window)', 
    title='Entropy Loss'
)
ax[2,0].plot(
    jnp.arange(len(entropy_losses)-(entropy_window-1))+entropy_window, 
    jnp.convolve(entropy_losses, jnp.ones(entropy_window,), 'valid') / entropy_window,
)
# Plot perception loss during RL
ax[2,1].grid()
ax[2,1].set(
    xlabel='Training Update', 
    ylabel=f'Loss ({window} upd. window)', 
    title='Perception Loss'
)
ax[2,1].plot(
    jnp.arange(len(perception_losses)-(window-1))+window, 
    jnp.convolve(perception_losses, jnp.ones(window,), 'valid') / window,
)
# Plot critic loss std during RL
ax[2,2].grid()
ax[2,2].set(
    xlabel='Training Update',
    ylabel=r'$\sigma(\mathcal{L}_{crit})$',
    title='Std critic loss',
)
ax[2,2].plot(
    jnp.arange(len(loss_stds[:,1])),
    loss_stds[:,1],
)
# Plot stds[0] during RL
ax[3,0].grid()
ax[3,0].set(
    xlabel='Training Update',
    ylabel='$\sigma(v)$',
    title='Std velocity',
    ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
)
ax[3,0].plot(
    jnp.arange(len(stds_during_rl)),
    stds_during_rl[:,0],
)
# Plot stds[1] during RL
ax[3,1].grid()
ax[3,1].set(
    xlabel='Training Update',
    ylabel='$\sigma(\omega)$',
    title='Std ang. vel.',
    ylim=(jnp.min(stds_during_rl)-0.01, jnp.max(stds_during_rl)+0.01),
)
ax[3,1].plot(
    jnp.arange(len(stds_during_rl)),
    stds_during_rl[:,1],
)
# Plot perception loss std during RL
ax[3,2].grid()
ax[3,2].set(
    xlabel='Training Update',
    ylabel=r'$\sigma(\mathcal{L}_{perc})$',
    title='Std perception loss',
)
ax[3,2].plot(
    jnp.arange(len(loss_stds[:,2])),
    loss_stds[:,2],
)
figure.savefig(os.path.join(os.path.dirname(__file__),"jessi_rl_training_plots.eps"), format='eps')