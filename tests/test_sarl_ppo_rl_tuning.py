import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import os
import optax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from decimal import Decimal

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.rollouts.ppo_rollouts import ppo_rl_rollout
from socialjym.utils.replay_buffers.ppo_replay_buffer import PPOBuffer
from socialjym.policies.sarl_ppo import SARLPPO

### Hyperparameters sets
distribution = 'gaussian'
hyperparamters_seed = 0
n_trials = 20
training_updates = 1_000 # For each set of hyperparameters
num_batches = 30
hp_lims = {
    'actor_learning_rates': [3e-4, 5e-6],
    'critic_learning_rates': [3e-3, 3e-4],
    'n_parallel_envs_multiplier': [1, 10],
    'buffer_capacities_multiplier': [20, 60],
    'betas_entropy': [1e-3, 5e-7],
}

# Initialize output dict
hpt_out = {
    "actor_losses": jnp.zeros((n_trials, training_updates), dtype=jnp.float32),
    "critic_losses": jnp.zeros((n_trials, training_updates), dtype=jnp.float32),
    "entropy_losses": jnp.zeros((n_trials, training_updates), dtype=jnp.float32),
    "returns": jnp.zeros((n_trials, training_updates), dtype=jnp.float32),
    "successes": jnp.zeros((n_trials, training_updates), dtype=int),
    "failures": jnp.zeros((n_trials, training_updates), dtype=int),
    "timeouts": jnp.zeros((n_trials, training_updates), dtype=int),
    "episodes": jnp.zeros((n_trials, training_updates), dtype=int),
    "actor_lr": jnp.zeros((n_trials,), dtype=jnp.float32),
    "critic_lr": jnp.zeros((n_trials,), dtype=jnp.float32),
    "n_parallel_envs": jnp.zeros((n_trials,), dtype=int),
    "buffer_capacity": jnp.zeros((n_trials,), dtype=int),
    "beta_entropy": jnp.zeros((n_trials,), dtype=jnp.float32),
}

# Load IL rollout output
with open(os.path.join(os.path.dirname(__file__),"il_out.pkl"), 'rb') as f:
    il_out = pickle.load(f)
# Save the IL model parameters, buffer state, and keys
il_actor_params = il_out['actor_params']
il_critic_params = il_out['critic_params']

### Trial loop
np.random.seed(hyperparamters_seed)
for trial in range(n_trials):
    ### Sample from hyperparameters space
    actor_lr = np.random.uniform(*hp_lims['actor_learning_rates'])
    critic_lr = np.random.uniform(*hp_lims['critic_learning_rates'])
    n_parallel_envs = num_batches * np.random.randint(*hp_lims['n_parallel_envs_multiplier'])
    buffer_capacity = n_parallel_envs * np.random.randint(*hp_lims['buffer_capacities_multiplier'])
    beta_entropy = np.random.uniform(*hp_lims['betas_entropy'])
    print(f"\n#################################### \nTRIAL {trial+1}/{n_trials}\nActor LR: {actor_lr}\nCritic LR: {critic_lr}\nParallel envs: {n_parallel_envs}\nBuffer capacity: {buffer_capacity}\nBeta entropy: {beta_entropy}")
    ### Hyperparameters initialization
    training_hyperparams = {
        'random_seed': 0,
        'kinematics': 'unicycle',
        'policy_name': 'sarl-ppo',
        'distribution': distribution,
        'n_humans': 5, 
        'rl_training_updates': training_updates,
        'rl_parallel_envs': n_parallel_envs,
        'rl_actor_learning_rate': actor_lr, 
        'rl_critic_learning_rate': critic_lr,
        'rl_buffer_capacity': buffer_capacity,
        'rl_clip_frac': 0.2, 
        'rl_num_epochs': 10, 
        'rl_num_batches': num_batches, 
        'rl_beta_entropy': beta_entropy, 
        'lambda_gae': 0.95, 
        'humans_policy': 'hsfm',
        'scenario': 'hybrid_scenario',
        'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5], np.int32),
        'reward_function': 'socialnav_reward2',
        'custom_episodes': False,
        'gradient_norm_scale': 0.5,
    }
    # Initialize reward function
    if training_hyperparams['reward_function'] == 'socialnav_reward1': 
        reward_function = Reward1(
            kinematics=training_hyperparams['kinematics']
        )
    elif training_hyperparams['reward_function'] == 'socialnav_reward2':
        reward_function = Reward2(
            progress_to_goal_reward = True,
            progress_to_goal_weight = 0.03,
        )
    else:
        raise ValueError(f"{training_hyperparams['reward_function']} is not a valid reward function")
    # Environment parameters
    env_params = {
        'robot_radius': 0.3,
        'n_humans': training_hyperparams['n_humans'],
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': False,
        'scenario': training_hyperparams['scenario'],
        'hybrid_scenario_subset': training_hyperparams['hybrid_scenario_subset'],
        'humans_policy': training_hyperparams['humans_policy'],
        'circle_radius': 7,
        'reward_function': reward_function,
        'kinematics': training_hyperparams['kinematics'],
    }
    # Initialize environment
    env = SocialNav(**env_params)
    # Initialize robot policy and vnet params
    policy = SARLPPO(env.reward_function, dt=env_params['robot_dt'], kinematics=env_params['kinematics'], distribution=training_hyperparams['distribution'])
    # Initialize RL optimizer
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(training_hyperparams['gradient_norm_scale']),
        optax.adam(
            learning_rate=optax.schedules.linear_schedule(
                init_value=training_hyperparams['rl_actor_learning_rate'], 
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
    # Initialize custom episodes path
    if training_hyperparams['custom_episodes']:
        rl_custom_episodes_path = os.path.join(os.path.expanduser("~"),f"Repos/social-jym/custom_episodes/rl_{training_hyperparams['scenario']}_{training_hyperparams['n_humans']}_humans.pkl")
    else:
        rl_custom_episodes_path = None
    # Initialize RL replay buffer
    replay_buffer = PPOBuffer(training_hyperparams['rl_buffer_capacity'], int(training_hyperparams['rl_buffer_capacity']/training_hyperparams['rl_num_batches']))
    # Initialize RL buffer state
    buffer_state = {
        'inputs': jnp.empty((training_hyperparams['rl_buffer_capacity'], env.n_humans, policy.vnet_input_size)),
        'critic_targets': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
        'sample_actions': jnp.empty((training_hyperparams['rl_buffer_capacity'], 2)),
        'old_values': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
        'old_neglogpdfs': jnp.empty((training_hyperparams['rl_buffer_capacity'],)),
    }
    # Initialize RL rollout params
    rl_rollout_params = {
        'initial_actor_params': il_actor_params,
        'initial_critic_params': il_critic_params,
        'n_parallel_envs': training_hyperparams['rl_parallel_envs'],
        'train_updates': training_hyperparams['rl_training_updates'],
        'random_seed': training_hyperparams['random_seed'],
        'actor_optimizer': actor_optimizer,
        'critic_optimizer': critic_optimizer,
        'buffer_state': buffer_state,
        'buffer_capacity': training_hyperparams['rl_buffer_capacity'],
        'policy': policy,
        'env': env,
        'replay_buffer': replay_buffer,
        'clip_range': training_hyperparams['rl_clip_frac'],
        'n_epochs': training_hyperparams['rl_num_epochs'],
        'beta_entropy': training_hyperparams['rl_beta_entropy'],
        'lambda_gae': training_hyperparams['lambda_gae'],
        'debugging': False,
        'debugging_interval': 1,
    }
    # REINFORCEMENT LEARNING ROLLOUT
    rl_out = ppo_rl_rollout(**rl_rollout_params)
    # Save the training data
    rl_out['aux_data']['actor_lr'] = actor_lr
    rl_out['aux_data']['critic_lr'] = critic_lr
    rl_out['aux_data']['n_parallel_envs'] = n_parallel_envs
    rl_out['aux_data']['buffer_capacity'] = buffer_capacity
    rl_out['aux_data']['beta_entropy'] = beta_entropy
    hpt_out = tree_map(lambda x, y: x.at[trial].set(y), hpt_out, rl_out['aux_data'])

### SAVE Hyperparameters tuning output
with open(os.path.join(os.path.dirname(__file__),"hpt_out.pkl"), 'wb') as f:
    pickle.dump(hpt_out, f)

### LOAD Hyperparameters tuning output
with open(os.path.join(os.path.dirname(__file__),"hpt_out.pkl"), 'rb') as f:
    hpt_out = pickle.load(f)

### PLOTTING
window = 50
returns_during_rl = hpt_out['returns']  
actor_losses = hpt_out['actor_losses']
critic_losses = hpt_out['critic_losses']
entropy_losses = hpt_out['entropy_losses']
success_during_rl = hpt_out['successes']
failure_during_rl = hpt_out['failures']
timeout_during_rl = hpt_out['timeouts']
episodes_during_rl = hpt_out['episodes']
colors = list(mcolors.TABLEAU_COLORS.values())
styles = ['solid','dashed','dotted']
## Plot RL training stats
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 18}
rc('font', **font)
figure, ax = plt.subplots(4,2,figsize=(15,15))
figure.subplots_adjust(hspace=0.5, bottom=0.05, top=0.90, right=0.9, left=0.1, wspace=0.35)
figure.suptitle(f"Hyperparameters tuning - {n_trials} Trials - Moving average of results ({window} updates window)")
# Plot returns during RL
ax[0,0].grid()
ax[0,0].set(
    xlabel='Training Update', 
    ylabel=f'Return', 
    title='Return (not episodic)'
)
for i in range(n_trials):
    ax[0,0].plot(
        jnp.arange(len(returns_during_rl[i])-(window-1))+window, 
        jnp.convolve(returns_during_rl[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot success rate during RL
success_rate_during_rl = success_during_rl / episodes_during_rl
ax[0,1].grid()
ax[0,1].set(
    xlabel='Training Update', 
    ylabel=f'Rate', 
    title='Success rate',
    ylim=(-0.1,1.1)
)
for i in range(n_trials):
    ax[0,1].plot(
        jnp.arange(len(success_rate_during_rl[i])-(window-1))+window, 
        jnp.convolve(success_rate_during_rl[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot failure rate during RL
failure_rate_during_rl = failure_during_rl / episodes_during_rl
ax[1,0].grid()
ax[1,0].set(
    xlabel='Training Update', 
    ylabel=f'Rate', 
    title='Failure rate',
    ylim=(-0.1,1.1)
)
for i in range(n_trials):
    ax[1,0].plot(
        jnp.arange(len(failure_rate_during_rl[i])-(window-1))+window, 
        jnp.convolve(failure_rate_during_rl[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot timeout rate during RL
timeout_rate_during_rl = timeout_during_rl / episodes_during_rl
ax[1,1].grid()
ax[1,1].set(
    xlabel='Training Update', 
    ylabel=f'Rate', 
    title='Timeout rate',
    ylim=(-0.1,1.1)
)
for i in range(n_trials):
    ax[1,1].plot(
        np.arange(len(timeout_rate_during_rl[i])-(window-1))+window, 
        jnp.convolve(timeout_rate_during_rl[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot actor loss during RL
ax[2,0].grid()
ax[2,0].set(
    xlabel='Training Update', 
    ylabel=f'Loss', 
    title='Actor Loss'
)
for i in range(n_trials):
    ax[2,0].plot(
        np.arange(len(actor_losses[i])-(window-1))+window, 
        jnp.convolve(actor_losses[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot critic loss during RL
ax[2,1].grid()
ax[2,1].set(
    xlabel='Training Update', 
    ylabel=f'Loss', 
    title='Critic Loss'
)
for i in range(n_trials):
    ax[2,1].plot(
        np.arange(len(critic_losses[i])-(window-1))+window, 
        jnp.convolve(critic_losses[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot entropy loss during RL
ax[3,0].grid()
ax[3,0].set(
    xlabel='Training Update', 
    ylabel=f'Loss', 
    title='Entropy Loss'
)
for i in range(n_trials):
    ax[3,0].plot(
        np.arange(len(entropy_losses[i])-(window-1))+window, 
        jnp.convolve(entropy_losses[i], jnp.ones(window,), 'valid') / window,
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Plot episodes during RL
ax[3,1].grid()
ax[3,1].set(
    xlabel='Training Update', 
    ylabel=f'Episodes', 
    title='Cumulative sim. episodes (no mv. avg.)'
)
for i in range(n_trials):
    ax[3,1].plot(
        jnp.arange(len(episodes_during_rl[i])),
        jnp.cumsum(episodes_during_rl[i]),
        color=colors[i%len(colors)],
        linestyle=styles[i//len(colors)],
        label=f'T{i}',
    )
# Save figure
handles, labels = ax[0,0].get_legend_handles_labels()
figure.legend(handles, labels, loc='center right', title='Trial')
figure.savefig(os.path.join(os.path.dirname(__file__),f"rl_hyperparam_tuning.eps"), format='eps')

### Print trial data
trial = 19
print(f"Trial {trial}")
print("Return: ", hpt_out["returns"][trial,-1])
print("Success rate:", success_rate_during_rl[trial,-1])
print("Actor learning rate: ", hpt_out["actor_lr"][trial])
print("Critic learning rate: ", hpt_out["critic_lr"][trial])
print("Parallel environments: ", hpt_out["n_parallel_envs"][trial])
print("Buffer capacity :", hpt_out["buffer_capacity"][trial])
print("Beta entropy : ", hpt_out["beta_entropy"][trial])