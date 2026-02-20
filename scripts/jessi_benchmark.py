import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward as DummySocialReward
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2 as SocialReward2
from socialjym.utils.aux_functions import initialize_metrics_dict
from socialjym.policies.jessi import JESSI
from socialjym.policies.dwa import DWA
from socialjym.policies.sarl import SARL
from socialjym.policies.cadrl import CADRL
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.sarl_star import SARLStar
from socialjym.policies.vanilla_e2e import VanillaE2E
from socialjym.policies.mppi import MPPI
from socialjym.policies.dra_mppi import DRAMPPI

# Hyperparameters
random_seed = 1_000_000 # Make sure test episodes are not the same as the training ones
n_trials = 100
# Tests
tests_n_humans = [1, 3, 5, 10]
tests_n_obstacles = [1, 3, 5]

# Plots utils
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "collisions_with_human": {"label": "Coll. w/ Hum. Rate", "episodic": False},
    "collisions_with_obstacle": {"label": "Coll. w/ Obs. Rate", "episodic": False},
    "timeouts": {"label": "Timeout Rate", "episodic": False}, 
    "returns": {"label": "Return ($\gamma = 0.9$)", "episodic": True},
    "times_to_goal": {"label": "Time to goal ($s$)", "episodic": True},
    "average_speed": {"label": "Lin. speed ($m/s$)", "episodic": True},
    "average_acceleration": {"label": "Lin. accel. ($m/s^2$)", "episodic": True},
    "average_jerk": {"label": "Lin. jerk ($m/s^3$)", "episodic": True},
    "average_angular_speed": {"label": "Ang. speed ($rad/s$)", "episodic": True},
    "average_angular_acceleration": {"label": "Ang. accel. ($rad/s^2$)", "episodic": True},
    "average_angular_jerk": {"label": "Ang. jerk ($rad/s^3$)", "episodic": True},
    "min_distance": {"label": "Minimum distance to humans ($m$)", "episodic": True},
    "space_compliance": {"label": "Space compliance", "episodic": True},
    "episodic_spl": {"label": "Episodic SPL", "episodic": True},
    "path_length": {"label": "Path length ($m$)", "episodic": True},
    "precision": {"label": "Precision (%)"},
    "recall": {"label": "Recall (%) "},
    "ADE": {"label": "Displacement Error ($m$)"},
    "AVE": {"label": "Velocity Error ($m/s$)"},
    "mahalanobis_pos": {"label": "Mahalanobis Dist. Pos."},
    "mahalanobis_vel": {"label": "Mahalanobis Dist. Vel."},
}
scenarios = {
    "parallel_traffic": {"label": "PaT"},
    "perpendicular_traffic": {"label": "PeT"},
    "corner_traffic": {"label": "CoT"},
    "circular_crossing": {"label": "CiC"},
    "circular_crossing_with_static_obstacles": {"label": "CCSO"},
    "delayed_circular_crossing": {"label": "DCC"},
    "robot_crowding": {"label": "RoC"},
    "crowd_navigation": {"label": "CrN"},
    "door_crossing": {"label": "DoC"},
    "crowd_chasing": {"label": "CrC"},
}
policies = {
    "jessi_multitask": {"label": "JESSI-MULTITASK", "short": "JESSI-MT", "only_ccso": False, "color": "tab:blue"},
    "jessi_modular": {"label": "JESSI-MODULAR", "short": "JESSI-MD", "only_ccso": False, "color": "tab:orange"},
    "jessi_policy": {"label": "JESSI-POLICY", "short": "JESSI-P", "only_ccso": False, "color": "tab:green"},
    "vanilla_e2e": {"label": "Vanilla E2E", "short": "V-E2E", "only_ccso": False, "color": "tab:red"},
    "bounded_vanilla_e2e": {"label": "Bounded Vanilla E2E", "short": "BV-E2E", "only_ccso": False, "color": "#800080"}, # purple
    "dir_safe": {"label": "DIR-SAFE", "short": "DIR-SAFE", "only_ccso": False, "color":"tab:purple"},
    "sarl_star": {"label": "SARL*", "short": "SARL*", "only_ccso": False, "color": "tab:brown"},
    "sarl": {"label": "SARL", "short": "SARL", "only_ccso": True, "color": "tab:pink"},
    "cadrl": {"label": "CADRL", "short": "CADRL", "only_ccso": True, "color": "tab:gray"},
    "dwa": {"label": "DWA", "short": "DWA", "only_ccso": False, "color": "tab:olive"},
    "mppi": {"label": "MPPI", "short": "MPPI", "only_ccso": False, "color": "tab:cyan"},
    "dra_mppi": {"label": "DRA-MPPI", "short": "DRA-MPPI", "only_ccso": False, "color": "black"},
}

def jessi_tests(jessi_params):
    metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
    all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
    policy = JESSI(
        lidar_num_rays=100,
        lidar_angular_range=jnp.pi * 2,
        lidar_max_dist=10.0,
        n_stack=5,
        n_stack_for_action_space_bounding=1,
    )
    for i, n_obstacle in enumerate(tests_n_obstacles):
        for j, n_human in enumerate(tests_n_humans):
            seen_env_params = {
                'n_stack': 5,
                'lidar_num_rays': 100,
                'lidar_angular_range': jnp.pi * 2,
                'lidar_max_dist': 10.0,
                'n_humans': n_human,
                'n_obstacles': n_obstacle,
                'robot_radius': 0.3,
                'robot_dt': 0.25,
                'humans_dt': 0.01,      
                'robot_visible': True,
                'scenario': 'training_scenario', 
                # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                'ccso_n_static_humans': 0,
                'ccso_static_humans_radius_mean': 0.3,
                'ccso_static_humans_radius_std': 0.025,
                'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                'kinematics': 'unicycle',
                'lidar_noise': True,
            }
            ct_env_params = seen_env_params.copy()
            ct_env_params['scenario'] = 'testing_scenario'
            ccso_env_params = seen_env_params.copy()
            ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
            ccso_env_params['ccso_n_static_humans'] = n_obstacle
            ccso_env_params['n_humans'] = n_human + n_obstacle
            # Initialize the environments
            seen_env = LaserNav(**seen_env_params)
            ct_env = LaserNav(**ct_env_params) # Unseen scenario
            ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
            # Test the trained JESSI-MULTITASK policy
            metrics_seen_scenarios = policy.evaluate(
                n_trials,
                random_seed,
                seen_env,
                jessi_params,
            )
            metrics_ct = policy.evaluate(
                n_trials,
                random_seed,
                ct_env,
                jessi_params,
            )
            metrics_ccso = policy.evaluate(
                n_trials,
                random_seed,
                ccso_env,
                jessi_params,
            )
            all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
            all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
            all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
    return all_metrics

if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests.pkl")):

    ### JESSI-MULTITASK tests ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests.pkl")):
        # Load JESSI-MULTITASK policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_multitask_rl_out.pkl'), 'rb') as f:
            jessi_params, _, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)
            

    ### JESSI-MODULAR tests ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_modular_tests.pkl")):
        # Load JESSI-MODULAR policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_modular_rl_out.pkl'), 'rb') as f:
            jessi_params, _, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_modular_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)
        

    ### JESSI-POLICY TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"jessi_policy_tests.pkl")):
        # Load JESSI-MODULAR policy parameters
        with open(os.path.join(os.path.dirname(__file__), 'jessi_policy_rl_out.pkl'), 'rb') as f:
            jessi_params, _, _ = pickle.load(f)
        # Execute tests
        all_metrics = jessi_tests(jessi_params)
        with open(os.path.join(os.path.dirname(__file__),"jessi_policy_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### VANILLA E2E TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"vanilla_e2e_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = VanillaE2E(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            action_space_bounding=False,
        )
        with open(os.path.join(os.path.dirname(__file__), 'vanilla_e2e_rl_out.pkl'), 'rb') as f:
            network_params, _, _ = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate(
                    n_trials,
                    random_seed,
                    seen_env,
                    network_params,
                )
                metrics_ct = policy.evaluate(
                    n_trials,
                    random_seed,
                    ct_env,
                    network_params,
                )
                metrics_ccso = policy.evaluate(
                    n_trials,
                    random_seed,
                    ccso_env,
                    network_params,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"vanilla_e2e_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### BOUNDED VANILLA E2E TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"bounded_vanilla_e2e_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = VanillaE2E(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            action_space_bounding=True,
        )
        with open(os.path.join(os.path.dirname(__file__), 'bounded_vanilla_e2e_rl_out.pkl'), 'rb') as f:
            network_params, _, _ = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate(
                    n_trials,
                    random_seed,
                    seen_env,
                    network_params,
                )
                metrics_ct = policy.evaluate(
                    n_trials,
                    random_seed,
                    ct_env,
                    network_params,
                )
                metrics_ccso = policy.evaluate(
                    n_trials,
                    random_seed,
                    ccso_env,
                    network_params,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"bounded_vanilla_e2e_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### DIR-SAFE TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"dir_safe_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = DIRSAFE(
            reward_function=DummySocialReward(kinematics='unicycle'),
        )
        jessi = JESSI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            n_stack_for_action_space_bounding=1,
        )
        with open(os.path.join(os.path.dirname(__file__), 'dir_safe.pkl'), 'rb') as f:
            network_params = pickle.load(f)['actor_params']
        with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
            perception_params = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    seen_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                metrics_ct = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ct_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                metrics_ccso = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ccso_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"dir_safe_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### SARL-STAR TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"sarl_star_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        dummy_env = LaserNav(kinematics='unicycle', reward_function=LaserReward(robot_radius=0.3), robot_radius=0.3, humans_dt=0.01, robot_dt=0.25, n_humans=1, n_obstacles=1, scenario='hybrid_scenario')
        policy = SARLStar(
            reward_function = SocialReward2(
                target_reached_reward = True,
                collision_penalty_reward = True,
                discomfort_penalty_reward = True,
                v_max = 1.,
                progress_to_goal_reward = True,
                progress_to_goal_weight = 0.03,
                high_rotation_penalty_reward=True,
                angular_speed_bound=1.,
                angular_speed_penalty_weight=0.0075,
            ),
            grid_size = dummy_env.get_grid_size(),
            use_planner = False,
            kinematics='unicycle'
        )
        jessi = JESSI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            n_stack_for_action_space_bounding=1,
        )
        with open(os.path.join(os.path.dirname(__file__), 'sarl.pkl'), 'rb') as f:
            network_params = pickle.load(f)['policy_params']
        with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
            perception_params = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                    'grid_map_computation': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    seen_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses
                )
                metrics_ct = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ct_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                metrics_ccso = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ccso_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"sarl_star_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### SARL TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"sarl_tests.pkl")):
        metrics_dims = (1,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = SARL(
            reward_function = SocialReward2(
                target_reached_reward = True,
                collision_penalty_reward = True,
                discomfort_penalty_reward = True,
                v_max = 1.,
                progress_to_goal_reward = True,
                progress_to_goal_weight = 0.03,
                high_rotation_penalty_reward=True,
                angular_speed_bound=1.,
                angular_speed_penalty_weight=0.0075,
            ),
            kinematics='unicycle'
        )
        jessi = JESSI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            n_stack_for_action_space_bounding=1,
        )
        with open(os.path.join(os.path.dirname(__file__), 'sarl.pkl'), 'rb') as f:
            network_params = pickle.load(f)['policy_params']
        with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
            perception_params = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
                ccso_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human + n_obstacle,
                    'n_obstacles': 0,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'circular_crossing_with_static_obstacles', 
                    'ccso_n_static_humans': n_obstacle,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                # Initialize the environments
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_ccso = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ccso_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"sarl_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### CADRL TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"cadrl_tests.pkl")):
        metrics_dims = (1,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = CADRL(
            reward_function = SocialReward2(
                target_reached_reward = True,
                collision_penalty_reward = True,
                discomfort_penalty_reward = True,
                v_max = 1.,
                progress_to_goal_reward = True,
                progress_to_goal_weight = 0.03,
                high_rotation_penalty_reward=True,
                angular_speed_bound=1.,
                angular_speed_penalty_weight=0.0075,
            ),
            kinematics='unicycle'
        )
        jessi = JESSI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            n_stack_for_action_space_bounding=1,
        )
        with open(os.path.join(os.path.dirname(__file__), 'cadrl.pkl'), 'rb') as f:
            network_params = pickle.load(f)['policy_params']
        with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
            perception_params = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                humans_radius_hypotheses = jnp.full((jessi.n_detectable_humans,), 0.3)
                ccso_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human + n_obstacle,
                    'n_obstacles': 0,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'circular_crossing_with_static_obstacles', 
                    'ccso_n_static_humans': n_obstacle,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                # Initialize the environments
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_ccso = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ccso_env,
                    jessi,
                    perception_params,
                    network_params,
                    humans_radius_hypotheses,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"cadrl_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### DWA TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"dwa_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = DWA(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
        )
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate(
                    n_trials,
                    random_seed,
                    seen_env,
                )
                metrics_ct = policy.evaluate(
                    n_trials,
                    random_seed,
                    ct_env,
                )
                metrics_ccso = policy.evaluate(
                    n_trials,
                    random_seed,
                    ccso_env,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"dwa_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### MPPI TESTS ###
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"mppi_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = MPPI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
        )
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate(
                    n_trials,
                    random_seed,
                    seen_env,
                )
                metrics_ct = policy.evaluate(
                    n_trials,
                    random_seed,
                    ct_env,
                )
                metrics_ccso = policy.evaluate(
                    n_trials,
                    random_seed,
                    ccso_env,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"mppi_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ## DRA-MPPI TESTS ##
    if not os.path.exists(os.path.join(os.path.dirname(__file__),"dra_mppi_tests.pkl")):
        metrics_dims = (3,len(tests_n_obstacles),len(tests_n_humans))
        all_metrics = initialize_metrics_dict(n_trials, metrics_dims)
        policy = DRAMPPI()
        jessi = JESSI(
            lidar_num_rays=100,
            lidar_angular_range=jnp.pi * 2,
            lidar_max_dist=10.0,
            n_stack=5,
            n_stack_for_action_space_bounding=1,
        )
        with open(os.path.join(os.path.dirname(__file__), 'pre_perception_network.pkl'), 'rb') as f:
            perception_params = pickle.load(f)
        for i, n_obstacle in enumerate(tests_n_obstacles):
            for j, n_human in enumerate(tests_n_humans):
                seen_env_params = {
                    'n_stack': 5,
                    'lidar_num_rays': 100,
                    'lidar_angular_range': jnp.pi * 2,
                    'lidar_max_dist': 10.0,
                    'n_humans': n_human,
                    'n_obstacles': n_obstacle,
                    'robot_radius': 0.3,
                    'robot_dt': 0.25,
                    'humans_dt': 0.01,      
                    'robot_visible': True,
                    'scenario': 'training_scenario', 
                    # 'hybrid_scenario_subset': jnp.array([0,1,2,3,4,6]), # Exclude circular_crossing_with_static_obstacles and corner_traffic - SEEN SCENARIO
                    'ccso_n_static_humans': 0,
                    'ccso_static_humans_radius_mean': 0.3,
                    'ccso_static_humans_radius_std': 0.025,
                    'reward_function': LaserReward(robot_radius=0.3,collision_with_humans_penalty=-.5),
                    'kinematics': 'unicycle',
                    'lidar_noise': True,
                }
                ct_env_params = seen_env_params.copy()
                ct_env_params['scenario'] = 'testing_scenario'
                ccso_env_params = seen_env_params.copy()
                ccso_env_params['scenario'] = 'circular_crossing_with_static_obstacles'
                ccso_env_params['ccso_n_static_humans'] = n_obstacle
                ccso_env_params['n_humans'] = n_human + n_obstacle
                # Initialize the environments
                seen_env = LaserNav(**seen_env_params)
                ct_env = LaserNav(**ct_env_params) # Unseen scenario
                ccso_env = LaserNav(**ccso_env_params) # Unseen scenario
                # Test the trained JESSI-MULTITASK policy
                metrics_seen_scenarios = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    seen_env,
                    jessi,
                    perception_params,
                )
                metrics_ct = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ct_env,
                    jessi,
                    perception_params,
                )
                metrics_ccso = policy.evaluate_on_jessi_perception(
                    n_trials,
                    random_seed,
                    ccso_env,
                    jessi,
                    perception_params,
                )
                all_metrics = tree_map(lambda x, y: x.at[0,i,j].set(y), all_metrics, metrics_seen_scenarios)
                all_metrics = tree_map(lambda x, y: x.at[1,i,j].set(y), all_metrics, metrics_ct)
                all_metrics = tree_map(lambda x, y: x.at[2,i,j].set(y), all_metrics, metrics_ccso)
        with open(os.path.join(os.path.dirname(__file__),"dra_mppi_tests.pkl"), 'wb') as f:
            pickle.dump(all_metrics, f)


    ### AGGREGATE ALL RESULTS ###
    # Load all test results and aggregate them in a single dictionary
    with open(os.path.join(os.path.dirname(__file__),"jessi_multitask_tests.pkl"), 'rb') as f:
            jessi_multitask_results = pickle.load(f)  
    with open(os.path.join(os.path.dirname(__file__),"jessi_modular_tests.pkl"), 'rb') as f:
            jessi_modular_results = pickle.load(f) 
    with open(os.path.join(os.path.dirname(__file__),"jessi_policy_tests.pkl"), 'rb') as f:
            jessi_policy_results = pickle.load(f)  
    with open(os.path.join(os.path.dirname(__file__),"vanilla_e2e_tests.pkl"), 'rb') as f:
            vanilla_e2e_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"bounded_vanilla_e2e_tests.pkl"), 'rb') as f:
            bounded_vanilla_e2e_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"dir_safe_tests.pkl"), 'rb') as f:
            dir_safe_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"sarl_star_tests.pkl"), 'rb') as f:
            sarl_star_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"sarl_tests.pkl"), 'rb') as f:
            sarl_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"cadrl_tests.pkl"), 'rb') as f:
            cadrl_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"dwa_tests.pkl"), 'rb') as f:
            dwa_results = pickle.load(f)  
    with open(os.path.join(os.path.dirname(__file__),"mppi_tests.pkl"), 'rb') as f:
            mppi_results = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__),"dra_mppi_tests.pkl"), 'rb') as f:
            dra_mppi_results = pickle.load(f)
    all_results = {
        'jessi_multitask': jessi_multitask_results,
        'jessi_modular': jessi_modular_results,
        'jessi_policy': jessi_policy_results,
        'vanilla_e2e': vanilla_e2e_results,
        'bounded_vanilla_e2e': bounded_vanilla_e2e_results,
        'dir_safe': dir_safe_results,
        'sarl_star': sarl_star_results,
        'sarl': sarl_results,
        'cadrl': cadrl_results,
        'dwa': dwa_results,
        'mppi': mppi_results,
        'dra_mppi': dra_mppi_results,
    }
    with open(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
else:
    with open(os.path.join(os.path.dirname(__file__),"jessi_benchmark_tests.pkl"), 'rb') as f:
        all_results = pickle.load(f)

### PRINT RESULTS SUMMARIES ###
metrics_to_plot = ["successes","collisions_with_human","collisions_with_obstacle","timeouts","times_to_goal","average_jerk","average_angular_jerk","space_compliance"]
higher_is_better = ["successes", "space_compliance"]
train_scenarios_summary = {p: {} for p in policies.keys()}
test_scenarios_summary = {p: {} for p in policies.keys()}
ccso_scenarios_summary = {p: {} for p in policies.keys()}
complete_summary = {p: {} for p in policies.keys()}
for metric in metrics_to_plot:
    # TRAIN Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :, :])
        train_scenarios_summary[p][metric] = float(y_data)
    # TEST Scenarios
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :, :])
        test_scenarios_summary[p][metric] = float(y_data)
    # CCSO Scenario
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts', 'collisions_with_obstacle', 'collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :]) / n_trials
        else:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :, :])
        ccso_scenarios_summary[p][metric] = float(y_data)
    ### Complete summary
    for p in all_results.keys():
        v_train = train_scenarios_summary[p].get(metric, jnp.nan)
        v_test  = test_scenarios_summary[p].get(metric, jnp.nan)
        v_ccso  = ccso_scenarios_summary[p].get(metric, jnp.nan)
        avg_val = float(jnp.nanmean(jnp.array([v_train, v_test, v_ccso])))
        complete_summary[p][metric] = avg_val
def print_pretty_table(summary_dict, title, latex_mode=False):
    print(f"\n{'-'*30} {title.upper()} {'-'*30}")
    if latex_mode:
        headers = ["Policy", "SR (\\%)", "Coll. Hum (\\%)", "Coll. Obs (\\%)", "Timeout (\\%)", "TTG (s)", "Lin Jerk", "Ang Jerk", "Space Comp."]
    else:
        headers = ["Policy", "SR (%)", "Coll. Hum (%)", "Coll. Obs (%)", "Timeout (%)", "TTG (s)", "Lin Jerk", "Ang Jerk", "Space Comp."]
    top_3_values = {}
    for metric in metrics_to_plot:
        valid_vals = [m[metric] for p, m in summary_dict.items() if m and not jnp.isnan(m.get(metric, float('nan')))]
        if not valid_vals:
            continue
        unique_vals = list(set([round(v, 5) for v in valid_vals]))
        unique_vals.sort(reverse=(metric in higher_is_better))
        top_3_values[metric] = unique_vals[:3]
    table_data = []
    for p, metrics in summary_dict.items():
        if not metrics: continue
        row = [policies[p]["short"]]
        for metric in metrics_to_plot:
            val = metrics.get(metric, float('nan'))
            if jnp.isnan(val):
                row.append("N/A")
                continue
            if metric in ['successes', 'collisions_with_human', 'collisions_with_obstacle', 'timeouts']:
                val_str = f"{val*100:.1f}\\%" if latex_mode else f"{val*100:.1f}%"
            else:
                val_str = f"{val:.2f}"
            val_rounded = round(val, 5)
            if metric in top_3_values and val_rounded in top_3_values[metric]:
                if latex_mode:
                    val_str = f"\\textbf{{{val_str}}}"
                else:
                    val_str = f"\033[1m{val_str}\033[0m" # Codice ANSI per grassetto terminale  
            row.append(val_str)
        table_data.append(row)
    if latex_mode:
        num_cols = len(headers)
        col_format = "c" * num_cols
        latex_lines = []
        latex_lines.append(f"\\begin{{table}}[thpb]")
        latex_lines.append(f"\\centering")
        latex_lines.append(f"\\caption{{{title}}}")
        latex_lines.append(f"\\resizebox{{\\columnwidth}}{{!}}{{")
        latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex_lines.append(f"\\toprule")
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append(f"\\midrule")
        for row in table_data:
            latex_lines.append(" & ".join(str(item) for item in row) + " \\\\")
        latex_lines.append(f"\\bottomrule")
        latex_lines.append(f"\\end{{tabular}}")
        latex_lines.append(f"}}")
        latex_lines.append(f"\\label{{tab:{title.lower().replace(' ', '_')}}}")
        latex_lines.append(f"\\end{{table}}\n")
        print("\n".join(latex_lines))
    else:
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))
print_pretty_table(complete_summary, "Overall Results", latex_mode=False)
print_pretty_table(train_scenarios_summary, "Train Scenarios Results", latex_mode=False)
print_pretty_table(test_scenarios_summary, "Test Scenarios Results", latex_mode=False)
print_pretty_table(ccso_scenarios_summary, "CCSO Scenarios Results", latex_mode=False)

### PLOT RESULTS ###
# Plot metrics against number of humans on TRAIN scenarios
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :], axis=0) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(tests_n_humans)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(TRAIN scenarios)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_1.eps"), format='eps')
# Plot metrics against number of obstacles on TRAIN scenarios
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° obstacles',
        title=metrics[metric]['label'],)
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_obstacles)))
    ax[i,j].set_xticklabels(tests_n_obstacles)
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :], axis=1) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][0, :, :, :], axis=(1,2))
        ax[i, j].plot(jnp.arange(len(tests_n_obstacles)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(TRAIN scenarios)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_2.eps"), format='eps')
# Plot metrics against number of humans on TEST scenario
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :], axis=0) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(tests_n_humans)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(TEST scenario)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_3.eps"), format='eps')
# Plot metrics against number of obstacles on TEST scenario
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° obstacles',
        title=metrics[metric]['label'],)
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_obstacles)))
    ax[i,j].set_xticklabels(tests_n_obstacles)
    for p in all_results.keys():
        if policies[p]["only_ccso"]: continue
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :], axis=1) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][1, :, :, :], axis=(1,2))
        ax[i, j].plot(jnp.arange(len(tests_n_obstacles)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(TEST scenario)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_4.eps"), format='eps')
# Plot metrics against number of humans on CCSO scenario
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° humans',
        title=metrics[metric]['label'],
    )
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_humans)))
    ax[i,j].set_xticklabels(tests_n_humans)
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :], axis=0) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :, :], axis=(0,2))
        ax[i, j].plot(jnp.arange(len(tests_n_humans)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(CCSO scenario)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_5.eps"), format='eps')
# Plot metrics against number of obstacles on CCSO scenario
metrics_to_plot = ["successes", "collisions_with_human", "collisions_with_obstacle", "timeouts", "times_to_goal", "average_speed", "average_jerk", "average_angular_speed", "average_angular_jerk","episodic_spl", "space_compliance","returns"]
figure, ax = plt.subplots(4, 3, figsize=(15, 20))
figure.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05, top=0.95, left=0.08, right=0.82)
for m, metric in enumerate(metrics_to_plot):
    i = m // 3
    j = m % 3
    ax[i,j].set(
        xlabel='N° obstacles',
        title=metrics[metric]['label'],)
    ax[i,j].grid(zorder=0)
    ax[i,j].set_xticks(jnp.arange(len(tests_n_obstacles)))
    ax[i,j].set_xticklabels(tests_n_obstacles)
    for p in all_results.keys():
        idx = 0 if policies[p]["only_ccso"] else 2
        if metric in ['successes', 'collisions', 'timeouts','collisions_with_obstacle','collisions_with_human']:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :], axis=1) / n_trials
            ax[i, j].set_ylim(-0.05, 1.05)
        else:
            y_data = jnp.nanmean(all_results[p][metric][idx, :, :, :], axis=(1,2))
        ax[i, j].plot(jnp.arange(len(tests_n_obstacles)), y_data, label=policies[p]['short'], color=policies[p]['color'], linewidth=2.5)
h, l = ax[0,0].get_legend_handles_labels()
figure.legend(h, l, loc='center right', title='Policy\n(CCSO scenario)')
figure.savefig(os.path.join(os.path.dirname(__file__), "jessi_benchmark_tests_6.eps"), format='eps')