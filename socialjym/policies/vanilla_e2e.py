import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
from functools import partial
import haiku as hk
import optax
import os
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

from socialjym.envs.base_env import ROBOT_KINEMATICS, SCENARIOS, EPSILON, HUMAN_POLICIES
from socialjym.utils.distributions.dirichlet import Dirichlet
from socialjym.policies.base_policy import BasePolicy
from jhsfm.hsfm import get_linear_velocity
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics

NETWORK_TYPES = ["MLP", "CNN", "Transformer"]
    
class MLPActorCritic(hk.Module):
    def __init__(
            self,
            v_max: float,
            wheels_distance: float,
            mlp_params: dict = {
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
            action_space_bounding:bool = False,
    ) -> None:
        super().__init__()
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.unbounded_action_vertices = jnp.array([
            [self.vmax, 0.],  # Forward
            [0., self.wmax],   # Rotate in place (left)
            [0., self.wmin],   # Rotate in place (right)
        ])  # Shape: (3, 2)
        self.action_space_bounding = action_space_bounding
        # Dimensions
        self.n_outputs = 3  # Dirichlet distribution over 3 action vertices
        self.mlp_params = mlp_params
        # 2. Self Attention Mechanism
        self.shared_backbone = hk.nets.MLP(
            **mlp_params,
            output_sizes=[128, 128],
            name="shared_backbone"
        )
        # 3. Final Output MLP
        self.actor_head = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50, self.n_outputs], 
            name="actor_head"
        )
        self.critic_head = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50, 1],
            name="critic_head"
        )
        self.dirichlet = Dirichlet()

    def __call__(
            self, 
            x: jnp.ndarray,
            y: jnp.ndarray,
            **kwargs: dict,
    ) -> tuple:
        """
        Args:
            x: Aligned LiDAR stack input. Shape (n_stack, lidar_num_rays, 2)
            y: Robot state input. Shape (6,) or (6 + 3,) depending on whether action space bounding parameters are included.

        Returns:
            sampled_actions: Sampled actions from the policy. Shape (2,) or (batch_size, 2)
            distributions: Dict containing the Dirichlet distribution parameters.
            state_values: State value estimates from the critic. Shape (,) or (batch_size,)
        """
        has_batch = x.ndim == 4  # (batch_size, n_stack, lidar_num_rays, 2)
        if not has_batch:
            x = jnp.expand_dims(x, 0)
            y = jnp.expand_dims(y, 0)
        batch_size = x.shape[0]
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        keys = random.split(random_key, batch_size)
        # Shared backbone
        x_flat = jnp.reshape(x, (batch_size, -1))
        shared_features = self.shared_backbone(x_flat)
        inputs = jnp.concatenate([shared_features, y], axis=-1)  # Shape: (batch_size, feature_dim + 6) or (feature_dim + 6,)
        ### ACTOR
        ## Compute Dirichlet distribution parameters
        alphas = nn.softplus(self.actor_head(inputs)) + 1  # (Batch, 3)
        ## Compute dirchlet distribution vetices
        if self.action_space_bounding:
            action_space_params = y[:,-3:]  # Shape: (batch_size, 3) or (3,)
            zeros = jnp.zeros((batch_size,))
            v1 = jnp.stack([zeros, action_space_params[:, 1] * self.wmax], axis=-1)
            v2 = jnp.stack([zeros, action_space_params[:, 2] * self.wmin], axis=-1)
            v3 = jnp.stack([action_space_params[:, 0] * self.vmax, zeros], axis=-1)
            vertices = jnp.stack([v1, v2, v3], axis=1)  # Shape: (batch_size, 3, 2)
        else:
            vertices = jnp.tile(self.unbounded_action_vertices, (batch_size, 1, 1))
        distributions = {"alphas": alphas, "vertices": vertices}
        ## Sample action
        sampled_actions = vmap(self.dirichlet.sample)(distributions, keys)
        ### CRITIC
        state_values = self.critic_head(inputs) # (Batch, 1)
        state_values = jnp.squeeze(state_values, axis=-1) # (Batch,)
        if not has_batch:
            sampled_actions = sampled_actions[0]
            state_values = state_values[0]
            distributions = tree_map(lambda t: t[0], distributions)
        # Actor head
        return sampled_actions, distributions, state_values

class VanillaE2E(BasePolicy):
    def __init__(
        self, 
        robot_radius:float=0.3,
        v_max:float=1., 
        gamma:float=0.9, 
        dt:float=0.25, 
        wheels_distance:float=0.7, 
        n_stack:int=5,
        lidar_angular_range=2*jnp.pi,
        lidar_max_dist=10.,
        lidar_num_rays=100,
        lidar_angles_robot_frame=None, # If not specified, rays are evenly distributed in the angular range
        network_type:str = "MLP",
        action_space_bounding:bool = False,
        n_stack_for_action_space_bounding:int = 1, # Number of recent observations to consider for action space bounding. If > 1, the LiDAR stacks are concatenated in the bounding process.
    ) -> None:
        """
        VANILLA-E2E (Simple E2E RL-based social navigation from LiDAR inputs).
        """
        # Input validation
        assert robot_radius > 0, "Robot radius must be positive"
        assert v_max > 0, "Maximum robot velocity must be positive"
        assert gamma >= 0 and gamma <= 1, "Discount factor must be in [0, 1]"
        assert dt > 0, "Time step must be positive"
        assert wheels_distance > 0, "Wheels distance must be positive"
        assert n_stack >= 2, "Number of stacked observations must be at least 2, to observe motion"
        assert lidar_angular_range > 0 and lidar_angular_range <= 2*jnp.pi, "LiDAR angular range must be in (0, 2pi]"
        assert lidar_max_dist > 1, "LiDAR maximum distance must be greater than 1 meter"
        assert lidar_num_rays >= 10, "LiDAR number of rays must be at least 10"
        assert network_type in NETWORK_TYPES, f"Network type must be one of {NETWORK_TYPES}"
        # Configurable attributes
        super().__init__(discount=gamma)
        self.robot_radius = robot_radius
        self.v_max = v_max
        self.dt = dt
        self.wheels_distance = wheels_distance
        self.n_stack = n_stack
        self.lidar_angular_range = lidar_angular_range
        self.lidar_max_dist = lidar_max_dist
        self.lidar_num_rays = lidar_num_rays
        if lidar_angles_robot_frame is None:
            self.lidar_angles_robot_frame = jnp.linspace(-lidar_angular_range/2, lidar_angular_range/2, lidar_num_rays)
        else:
            assert len(lidar_angles_robot_frame) == lidar_num_rays, "Length of lidar_angles_robot_frame must be equal to lidar_num_rays"
            self.lidar_angles_robot_frame = lidar_angles_robot_frame
        self.network_type = NETWORK_TYPES.index(network_type)
        self.action_space_bounding = action_space_bounding
        self.n_stack_for_action_space_bounding = n_stack_for_action_space_bounding
        # Default attributes
        self.name = "VanillaE2E"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.dirichlet = Dirichlet()
        self.unbounded_action_vertices = jnp.array([
            [self.v_max, 0.],  # Forward
            [0., 2 * self.v_max / self.wheels_distance],   # Rotate in place (left)
            [0., -2 * self.v_max / self.wheels_distance],   # Rotate in place (right)
        ])  # Shape: (3, 2)
        # Initialize Actor Critic network
        if self.network_type == NETWORK_TYPES.index("MLP"):
             @hk.transform
             def actor_critic_network(x, y, **kwargs) -> jnp.ndarray:
                 actor_critic = MLPActorCritic(
                     self.v_max, 
                     self.wheels_distance, 
                     action_space_bounding=self.action_space_bounding,
                 ) 
                 return actor_critic(x, y, **kwargs)
             self.actor_critic = actor_critic_network
        elif self.network_type == NETWORK_TYPES.index("CNN"):
             raise NotImplementedError("CNN network type not implemented yet")
        elif self.network_type == NETWORK_TYPES.index("Transformer"):
             raise NotImplementedError("Transformer network type not implemented yet")
    
    # Private methods

    @partial(jit, static_argnames=("self"))
    def _align_lidar_stack(self, obs_stack, ref_position, ref_orientation):
        """
        args:
        - obs_stack (lidar_num_rays + 6):  [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].

        outputs:
        - pointcloud_and_action (lidar_num_rays, 2): LiDAR points in robot reference frame
        - pointcloud_world_frame (lidar_num_rays, 2): LiDAR points in world frame
        """
        ## Split obs stack
        robot_position = obs_stack[:2]  # Shape: (2,)
        robot_orientation = obs_stack[2]  # Shape: ()
        #robot_radius = obs_stack[3]  # Shape: ()
        #robot_action = obs_stack[4:6]  # Shape: (2,)
        lidar_measurements = obs_stack[6:]  # Shape: (lidar_num_rays)
        ## Align scan to reference frame
        # Compute LiDAR angles in world frame
        lidar_angles = self.lidar_angles_robot_frame + robot_orientation  # Shape: (lidar_num_rays)
        # Compute cartesian coordinates of LiDAR points in world frame
        xs = lidar_measurements * jnp.cos(lidar_angles) + robot_position[0]
        ys = lidar_measurements * jnp.sin(lidar_angles) + robot_position[1]
        points_world = jnp.stack((xs, ys), axis=-1)  # Shape: (lidar_num_rays, 2)
        # Roto-translate points to robot frame
        c, s = jnp.cos(ref_orientation), jnp.sin(ref_orientation)
        R = jnp.array([
            [c, -s],
            [s,  c]
        ])
        points_robot = jnp.dot(points_world - ref_position, R)
        return points_robot, points_world
    
    # Public methods

    @partial(jit, static_argnames=("self"))
    def init_nn(
        self, 
        key:random.PRNGKey, 
    ) -> tuple:
        # Inputs are shaped:
        # x: (self.n_stack, self.lidar_num_rays, 2)
        # y: (6,) or (6 + 3,) depending on whether action space bounding is enabled
        network_params = self.actor_critic.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 2)), jnp.zeros((6 + 3) if self.action_space_bounding else 6), random_key=key)
        return network_params

    @partial(jit, static_argnames=("self"))
    def bound_action_space(self, lidar_point_cloud, eps=1e-6):
        """
        Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
        WARNING: Assumes LiDAR orientation is align with robot frame.
        """
        # Lower ALPHA
        is_inside_frontal_rect = (
            (lidar_point_cloud[:,0] >=  0 + eps) & # xmin
            (lidar_point_cloud[:,0] <= self.v_max * self.dt + self.robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= -self.robot_radius + eps) &  # ymin
            (lidar_point_cloud[:,1] <= self.robot_radius - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_frontal_rect[:, None],
            lidar_point_cloud,
            jnp.full(shape=(self.n_stack_for_action_space_bounding*self.lidar_num_rays, 2), fill_value=jnp.nan)
        )
        min_x = jnp.nanmin(intersection_points[:,0])
        new_alpha = lax.cond(
            ~jnp.isnan(min_x),
            lambda _: jnp.max(jnp.array([0, min_x - self.robot_radius])) / (self.v_max * self.dt),
            lambda _: 1.,
            None,
        )
        @jit
        def _lower_beta_and_gamma(tup:tuple):
            lidar_point_cloud, new_alpha, vmax, wheels_distance, dt = tup
            # Lower BETA
            is_inside_left_rect = (
                (lidar_point_cloud[:,0] >= -self.robot_radius + eps) & # xmin
                (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + self.robot_radius - eps) & # xmax
                (lidar_point_cloud[:,1] >= self.robot_radius + eps) &  # ymin
                (lidar_point_cloud[:,1] <= self.robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance)) - eps) # ymax
            )
            intersection_points = jnp.where(
                is_inside_left_rect[:, None],
                lidar_point_cloud,
                jnp.full(shape=(self.n_stack_for_action_space_bounding*self.lidar_num_rays, 2), fill_value=jnp.nan)
            )
            min_y = jnp.nanmin(intersection_points[:,1])
            new_beta = lax.cond(
                ~jnp.isnan(min_y),
                lambda _: (min_y - self.robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            # Lower GAMMA
            is_inside_right_rect = (
                (lidar_point_cloud[:,0] >=  -self.robot_radius + eps) & # xmin
                (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + self.robot_radius - eps) & # xmax
                (lidar_point_cloud[:,1] >= -self.robot_radius - (new_alpha*dt**2*vmax**2/(4*wheels_distance)) + eps) & # ymin
                (lidar_point_cloud[:,1] <= -self.robot_radius - eps) # ymax
            )
            intersection_points = jnp.where(
                is_inside_right_rect[:, None],
                lidar_point_cloud,
                jnp.full(shape=(self.n_stack_for_action_space_bounding*self.lidar_num_rays, 2), fill_value=jnp.nan)
            )
            max_y = jnp.nanmax(intersection_points[:,1])
            new_gamma = lax.cond(
                ~jnp.isnan(max_y),
                lambda _: (-max_y - self.robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            return new_beta, new_gamma
        new_beta, new_gamma = lax.cond(
            new_alpha == 0.,
            lambda _: (1., 1.),
            _lower_beta_and_gamma,
            (lidar_point_cloud, new_alpha, self.v_max, self.wheels_distance, self.dt)
        )
        # Apply lower bound to new_alpha, new_beta, new_gamma
        new_alpha = jnp.max(jnp.array([EPSILON, new_alpha]))
        new_beta = jnp.max(jnp.array([EPSILON, new_beta]))
        new_gamma = jnp.max(jnp.array([EPSILON, new_gamma]))
        return jnp.array([new_alpha, new_beta, new_gamma])

    @partial(jit, static_argnames=("self"))
    def align_lidar(
        self,
        obs:jnp.ndarray,
    ):
        """
        Align lidar scans in the observation stacks to the robot frame of the most recent observation.
        Prepare the input for the encoder network.

        args:
        - obs (n_stack, lidar_num_rays + 6): Each stack [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        The first stack is the most recent one.

        output:
        - processed_obs (n_stack, lidar_num_rays, 2): aligned LiDAR stack. First information corresponds to the most recent observation.
        """
        ref_position = obs[0,:2]
        ref_orientation = obs[0,2]
        return vmap(VanillaE2E._align_lidar_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)

    @partial(jit, static_argnames=("self"))
    def compute_actor_inputs(
        self,
        obs:jnp.ndarray,
        robot_goal:jnp.ndarray,
    ):
        """
        Compute the inputs for the actor network from the raw observation.

        args:
        - obs (n_stack, lidar_num_rays + 6): Each stack [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        The first stack is the most recent one.
        - robot_goal (2,): Goal position in world frame.

        output:
        - aligned_lidar_stack (n_stack, lidar_num_rays, 2): Aligned LiDAR stack. First information corresponds to the most recent observation.
        - robot_state_input (6 + optional 3 bounding parameters): Robot state input for the actor network.
        """
        # Align LiDAR scans
        aligned_lidar_stack, _ = self.align_lidar(obs)
        # Robot state input
        robot_position = obs[0,:2]
        robot_orientation = obs[0,2]
        c, s = jnp.cos(-robot_orientation), jnp.sin(-robot_orientation)
        R = jnp.array([[c, -s],
                    [s,  c]])
        translated_position = robot_goal - robot_position
        rc_robot_goal = R @ translated_position
        robot_goal_dist = jnp.linalg.norm(rc_robot_goal)
        robot_goal_theta = jnp.arctan2(rc_robot_goal[1], rc_robot_goal[0])
        robot_goal_sin_theta = jnp.sin(robot_goal_theta)
        robot_goal_cos_theta = jnp.cos(robot_goal_theta)
        if self.action_space_bounding:
            point_cloud_for_bounding = aligned_lidar_stack[:self.n_stack_for_action_space_bounding,:, :]  # Shape: (n_stack_for_action_space_bounding, lidar_num_rays, 2)
            point_cloud_for_bounding = jnp.reshape(
                point_cloud_for_bounding,
                (self.n_stack_for_action_space_bounding * self.lidar_num_rays, 2)
            )
            bounding_parameters = self.bound_action_space(
                point_cloud_for_bounding,  
            )
            robot_state_input = jnp.array([robot_goal_dist, robot_goal_sin_theta, robot_goal_cos_theta, self.v_max, self.robot_radius, self.wheels_distance, *bounding_parameters]) 
        else:
            robot_state_input = jnp.array([robot_goal_dist, robot_goal_sin_theta, robot_goal_cos_theta, self.v_max, self.robot_radius, self.wheels_distance])
        return aligned_lidar_stack, robot_state_input

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict,
        network_params:dict,
        sample:bool = False,
    ) -> jnp.ndarray:
        # Compute encoder input and last lidar point cloud (for action bounding)
        aligned_lidar_readings, robot_state_input = self.compute_actor_inputs(
            obs,
            info["robot_goal"],
        )
        # Compute action
        key, subkey = random.split(key)
        sampled_action, actor_distr, state_value = self.actor_critic.apply(
            network_params, 
            None, 
            aligned_lidar_readings,
            robot_state_input,
            random_key=subkey
        )
        action = lax.cond(sample, lambda _: sampled_action, lambda _: self.dirichlet.mean(actor_distr), None)
        return action, key, aligned_lidar_readings, robot_state_input, sampled_action, actor_distr, state_value
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        network_params,
        sample,
    ):
        return vmap(VanillaE2E.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos,
            network_params,
            sample,
        )   

    def evaluate(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
        network_params:dict,
    ) -> dict:
        """
        Test the trained policy over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "VanillaE2E policy can only be evaluated on unicycle kinematics"
        assert env.robot_dt == self.dt, f"Environment time step (dt={env.dt}) must be equal to policy time step (dt={self.dt}) for evaluation"
        assert env.lidar_angular_range == self.lidar_angular_range, f"Environment LiDAR angular range (lidar_angular_range={env.lidar_angular_range}) must be equal to policy LiDAR angular range (lidar_angular_range={self.lidar_angular_range}) for evaluation"
        assert env.lidar_max_dist == self.lidar_max_dist, f"Environment LiDAR max distance (lidar_max_dist={env.lidar_max_dist}) must be equal to policy LiDAR max distance (lidar_max_dist={self.lidar_max_dist}) for evaluation"
        assert env.lidar_num_rays == self.lidar_num_rays, f"Environment LiDAR number of rays (lidar_num_rays={env.lidar_num_rays}) must be equal to policy LiDAR number of rays (lidar_num_rays={self.lidar_num_rays}) for evaluation"
        assert env.n_stack == self.n_stack, f"Environment observation stack size (n_stack={env.n_stack}) must be equal to policy observation stack size (n_stack={self.n_stack}) for evaluation"
        time_limit = env.reward_function.time_limit
        @loop_tqdm(n_trials)
        @jit
        def _fori_body(i:int, for_val:tuple):   
            @jit
            def _while_body(while_val:tuple):
                # Retrieve data from the tuple
                state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states = while_val
                action, policy_key, _, _, _, _, _ = self.act(policy_key, obs, info, network_params, sample=False)
                state, obs, info, _, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states

            ## Retrieve data from the tuple
            seed, metrics = for_val
            policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
            env_key = random.PRNGKey(seed + 1_000_000)
            ## Reset the environment
            state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            initial_robot_position = state[-1,:2]
            ## Episode loop
            all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
            all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
            while_val_init = (state, obs, info, init_outcome, policy_key, env_key, 0, all_actions, all_states)
            _, _, end_info, outcome, policy_key, env_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
            ## Update metrics
            metrics = compute_episode_metrics(
                environment=env.environment,
                metrics=metrics,
                episode_idx=i, 
                initial_robot_position=initial_robot_position, 
                all_states=all_states, 
                all_actions=all_actions, 
                outcome=outcome, 
                episode_steps=episode_steps, 
                end_info=end_info, 
                max_steps=int(time_limit/env.robot_dt)+1, 
                personal_space=0.5,
                robot_dt=env.robot_dt,
                robot_radius=env.robot_radius,
                ccso_n_static_humans=env.ccso_n_static_humans,
                robot_specs={'kinematics': env.kinematics, 'v_max': self.v_max, 'wheels_distance': self.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
            )
            seed += 1
            return seed, metrics
        # Initialize metrics
        metrics = initialize_metrics_dict(n_trials)
        # Execute n_trials tests
        if env.scenario == SCENARIOS.index('circular_crossing_with_static_obstacles'):
            print(f"\nExecuting {n_trials} tests with {env.n_humans - env.ccso_n_static_humans} humans and {env.ccso_n_static_humans} obstacles...")
        else:
            print(f"\nExecuting {n_trials} tests with {env.n_humans} humans and {env.n_obstacles} obstacles...")
        _, metrics = lax.fori_loop(0, n_trials, _fori_body, (random_seed, metrics))
        # Print results
        print_average_metrics(n_trials, metrics)
        return metrics