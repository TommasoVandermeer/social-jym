import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from functools import partial
import haiku as hk
import optax

from socialjym.envs.base_env import wrap_angle
from socialjym.utils.distributions.base_distribution import DISTRIBUTIONS
from socialjym.utils.distributions.dirichlet import Dirichlet
from socialjym.utils.distributions.gaussian_mixture_model import BivariateGMM
from socialjym.policies.base_policy import BasePolicy

EPSILON = 1e-5

class Encoder(hk.Module):
    def __init__(
            self,
            mean_limits:jnp.ndarray,
            n_gaussian_mixture_components:int,
            lidar_num_rays:int,
            n_stack:int,
            prediction_time:float,
            max_humans_velocity:float,
            mlp_params:dict={
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
        ) -> None:
        super().__init__(name="lidar_network") 
        self.mean_limits = mean_limits 
        self.n_components = n_gaussian_mixture_components
        self.lidar_rays = lidar_num_rays
        self.n_stack = n_stack
        self.prediction_time = prediction_time
        self.max_humans_velocity = max_humans_velocity
        self.max_displacement = self.max_humans_velocity * self.prediction_time
        self.n_inputs = n_stack * (2 * lidar_num_rays + 2)
        self.n_outputs = self.n_components * 6 * 3  # 6 outputs per GMM cell (mean_x, mean_y, sigma_x, sigma_y, correlation, weight) times  3 GMMs (current and next)
        self.backbone = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_inputs * 2, self.n_inputs, self.n_outputs * 2], 
            name="mlp"
        )
        self.head1 = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_outputs * 2, self.n_outputs // 3], 
            name="head1"
        )
        self.head2 = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_outputs * 2, self.n_outputs // 3], 
            name="head2"
        )
        self.head3 = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_outputs * 3, self.n_outputs * 2, (self.n_outputs // 3) - self.n_components], # We take GMM component weights predicted by head2
            name="head3"
        )

    def __call__(
            self, 
            x: jnp.ndarray
        ) -> jnp.ndarray:
        """
        Maps Lidar scan to GMM parameters
        """
        ### Process inputs
        backbone_out = self.backbone(x)
        obstacles_gmm = self.head1(backbone_out)
        humans_gmm = self.head2(backbone_out)
        next_humans_info = self.head3(jnp.concatenate((backbone_out, humans_gmm), axis=-1))
        ### Clip next humans displacement means
        x_displacement_means = next_humans_info[:, :self.n_components]
        y_displacement_means = next_humans_info[:, self.n_components:2*self.n_components]
        dists = jnp.sqrt(x_displacement_means**2 + y_displacement_means**2)
        soft_clipped_dists = nn.tanh(dists) * self.max_displacement
        x_displacement_means = x_displacement_means / (dists + 1e-6) * soft_clipped_dists
        y_displacement_means = y_displacement_means / (dists + 1e-6) * soft_clipped_dists
        next_humans_gmm = jnp.concatenate((
            humans_gmm[:, :self.n_components] + x_displacement_means,
            humans_gmm[:, self.n_components:2*self.n_components] + y_displacement_means,
            next_humans_info[:, 2*self.n_components:5*self.n_components],
            humans_gmm[:, 5*self.n_components:],
        ), axis=-1)
        ### Transform outputs to GMM parameters
        @jit
        def vector_to_gmm_params(vector:jnp.ndarray, x_mean_bounds:jnp.ndarray, y_mean_bounds:jnp.ndarray) -> dict:  
            ### Separate outputs
            x_means = nn.tanh(vector[:, :self.n_components])
            x_means = ((x_means + 1) / 2) * (x_mean_bounds[1] - x_mean_bounds[0]) + x_mean_bounds[0]  # Scale to box limits
            y_means = nn.tanh(vector[:, self.n_components:2*self.n_components])
            y_means = ((y_means + 1) / 2) * (y_mean_bounds[1] - y_mean_bounds[0]) + y_mean_bounds[0]  # Scale to box limits
            x_log_sigmas = vector[:, 2*self.n_components:3*self.n_components]  # Std in x
            y_log_sigmas = vector[:, 3*self.n_components:4*self.n_components]  # Std in y
            correlations = nn.tanh(vector[:, 4*self.n_components:5*self.n_components])  # Correlations
            weights = nn.softmax(vector[:, 5*self.n_components:], axis=-1)  # Weights
            distr = {
                "means": jnp.stack((x_means, y_means), axis=-1), # Shape (batch_size, n_components, n_dimensions)
                "logsigmas": jnp.stack((x_log_sigmas, y_log_sigmas), axis=-1), # Shape (batch_size, n_components, n_dimensions)
                "correlations": correlations,  # Shape (batch_size, n_components)
                "weights": weights,  # Shape (batch_size, n_components)
            }
            return distr
        obs_distr = vector_to_gmm_params(obstacles_gmm, self.mean_limits[0], self.mean_limits[1])
        hum_distr = vector_to_gmm_params(humans_gmm, self.mean_limits[0], self.mean_limits[1])
        next_hum_distr = vector_to_gmm_params(next_humans_gmm, self.mean_limits[0], self.mean_limits[1])
        return obs_distr, hum_distr, next_hum_distr

class Actor(hk.Module):
    def __init__(
            self,
            n_gaussian_mixture_components:int,
            v_max:float,
            wheels_distance:float,
            mlp_params:dict={
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
        ) -> None:
        super().__init__(name="actor_network") 
        self.n_components = n_gaussian_mixture_components
        self.vmax = v_max
        self.wheels_distance = wheels_distance
        self.n_inputs = 3 * 6 * self.n_components  # 6 outputs per GMM cell (mean_x, mean_y, sigma_x, sigma_y, correlation, weight) times  3 GMMs (obstacles, current humans, next humans)
        self.n_outputs = 3 # Dirichlet distribution over 3 action vertices
        self.mlp = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_inputs * 5, self.n_inputs * 5, self.n_inputs * 3, self.n_outputs], 
            name="mlp"
        )
        self.dirichlet = Dirichlet()

    def __call__(
            self, 
            x,
            **kwargs:dict,
        ) -> jnp.ndarray:
        ## Get kwargs
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        alphas = self.mlp(x)
        alphas = nn.softplus(alphas) + 1
        ## Compute dirchlet distribution parameters
        vertices = jnp.array([
            [0, (2 * self.vmax / self.wheels_distance)],
            [0, (-2 * self.vmax / self.wheels_distance)],
            [self.vmax, 0]
        ])
        distribution = {"alphas": alphas, "vertices": vertices}
        ## Sample action
        sampled_action = self.dirichlet.sample(distribution, random_key)
        return sampled_action, distribution
    
class Critic(hk.Module):
    def __init__(
            self,
            n_gaussian_mixture_components:int,
            mlp_params:dict={
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
        ) -> None:
        super().__init__(name="critic_network") 
        self.n_components = n_gaussian_mixture_components
        self.n_inputs = 3 * 6 * self.n_components  # 6 outputs per GMM cell (mean_x, mean_y, sigma_x, sigma_y, correlation, weight) times  3 GMMs (obstacles, current humans, next humans)
        self.n_outputs = 1 # State value
        self.mlp = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_inputs * 5, self.n_inputs * 5, self.n_inputs * 3, self.n_outputs], 
            name="mlp"
        )

    def __call__(
            self, 
            x,
        ) -> jnp.ndarray:
        return self.mlp(x)

class JESSI(BasePolicy):
    def __init__(
        self, 
        v_max:float=1., 
        gamma:float=0.9, 
        dt:float=0.25, 
        wheels_distance:float=0.7, 
        n_stack:int=5,
        lidar_angular_range=2*jnp.pi,
        lidar_max_dist=10.,
        lidar_num_rays=100,
        lidar_angles_robot_frame=None, # If not specified, rays are evenly distributed in the angular range
        n_gmm_components:int=10,
        prediction_horizon:int=4,
        max_humans_velocity:float=1.5,
        gmm_means_limits:jnp.ndarray=jnp.array([[-2,4], [-3,3]]),
    ) -> None:
        """
        JESSI (JAX-based E2E Safe Social Interpretable autonomous navigation).
        """
        # Configurable attributes
        super().__init__(discount=gamma)
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
        self.n_gmm_components = n_gmm_components
        self.gmm_means_limits = gmm_means_limits
        self.prediction_horizon = prediction_horizon
        self.max_humans_velocity = max_humans_velocity
        # Default attributes
        self.name = "JESSI"
        self.dirichlet = Dirichlet()
        self.gmm = BivariateGMM(self.n_gmm_components)
        # Initialize Encoder network
        @hk.transform
        def encoder_network(x):
            net = Encoder(self.gmm_means_limits, self.n_gmm_components, self.lidar_num_rays, self.n_stack, self.dt * self.prediction_horizon, self.max_humans_velocity)
            return net(x)
        self.encoder = encoder_network
        # Initialize Actor
        @hk.transform
        def actor_network(x, **kwargs) -> jnp.ndarray:
            actor = Actor(self.n_gmm_components, self.v_max, self.wheels_distance) 
            return actor(x, **kwargs)
        self.actor = actor_network
        # Initialize Critic
        @hk.transform
        def critic_network(x) -> jnp.ndarray:
            critic = Critic(self.n_gmm_components) 
            return critic(x)
        self.critic = critic_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _compute_rl_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
        current_beta_entropy:float,
        clip_range:float,
        debugging:bool=False,
    ) -> tuple:
        
        # Experiences: {
        #   "inputs":jnp.ndarray, 
        #   "critic_targets":jnp.ndarray, 
        #   "sample_actions":jnp.ndarray, 
        #   "old_values":jnp.ndarray, 
        #   "old_neglogpdfs":jnp.ndarray
        # },

        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray, 
            old_values:jnp.ndarray, 
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0))
            def _rl_loss_function(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:float, 
                old_value:float,
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the clipped prediction
                clipped_prediction = jnp.clip(prediction, old_value - clip_range, old_value + clip_range)
                # Compute the loss
                return jnp.maximum(jnp.square(target - prediction), jnp.square(target - clipped_prediction))
            
            critic_loss = _rl_loss_function(current_critic_params, inputs, critic_targets, old_values)
            return 0.5 * jnp.mean(critic_loss)
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
            advantages:jnp.ndarray,  
            old_neglogpdfs:jnp.ndarray,
            beta_entropy:float = 0.0001,
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0, 0))
            def _rl_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
                advantage:jnp.ndarray, 
                old_neglogpdf:jnp.ndarray,
            ) -> jnp.ndarray:
                # Compute the prediction
                _, distr = self.actor.apply(current_actor_params, None, input)
                # Compute the log probability of the action
                neglogpdf = self.dirichlet.neglogp(distr, sample_action)
                # Compute policy ratio
                ratio = jnp.exp(old_neglogpdf - neglogpdf)
                lax.cond(
                    debugging,
                    lambda _: debug.print(
                        "Ratio: {x} - Old neglogp: {y} - New neglogp: {z} - distr: {w} - action: {a} - advantage: {b}", 
                        x=ratio,
                        y=old_neglogpdf,
                        z=neglogpdf,
                        w=distr,
                        a=sample_action,
                        b=advantage,
                    ),
                    lambda _: None,
                    None,
                )
                # Compute actor loss
                actor_loss = jnp.maximum(- ratio * advantage, - jnp.clip(ratio, 1-clip_range, 1+clip_range) * advantage)
                # Compute the entropy loss
                entropy_loss = self.dirichlet.entropy(distr)
                # Compute the loss
                return actor_loss, entropy_loss
            
            actor_losses, entropy_losses = _rl_loss_function(current_actor_params, inputs, sample_actions, advantages, old_neglogpdfs)
            actor_loss = jnp.mean(actor_losses)
            entropy_loss = - beta_entropy * jnp.mean(entropy_losses)
            loss = actor_loss + entropy_loss
            return loss, {"actor_loss": actor_loss, "entropy_loss": entropy_loss}

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        old_values = experiences["old_values"]
        old_neglogpdfs = experiences["old_neglogpdfs"]
        # Compute and normalize advantages
        advantages = critic_targets - old_values
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + EPSILON)
        # Compute critic loss and gradients
        critic_loss, critic_grads = value_and_grad(_batch_critic_loss_function)(
            current_critic_params, 
            inputs,
            critic_targets,
            old_values
        )
        # Compute actor loss and gradients
        actor_and_entropy_loss, actor_grads = value_and_grad(_batch_actor_loss_function, has_aux=True)(
            current_actor_params, 
            inputs,
            sample_actions, 
            advantages,
            old_neglogpdfs,
            current_beta_entropy,
        )
        _, all_losses = actor_and_entropy_loss
        actor_loss = all_losses["actor_loss"]
        entropy_loss = all_losses["entropy_loss"]
        return critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss

    @partial(jit, static_argnames=("self"))
    def _compute_il_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
    ) -> tuple:
        
        # Experiences: {
        #   "inputs":jnp.ndarray, 
        #   "critic_targets":jnp.ndarray, 
        #   "sample_actions":jnp.ndarray, 
        # },

        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray, 
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _il_loss_function(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:float, 
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the loss
                return jnp.square(target - prediction)
            
            critic_loss = _il_loss_function(
                current_critic_params,
                inputs,
                critic_targets)
            return jnp.mean(critic_loss)
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _il_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
            ) -> jnp.ndarray:
                # Compute the prediction (here we should input a key but for now we work only with mean actions)
                _, distr = self.actor.apply(current_actor_params, None, input)
                # Get mean action
                action = self.dirichlet.mean(distr)
                # Compute the loss
                return 0.5 * jnp.sum(jnp.square(action - sample_action))
            
            actor_losses = _il_loss_function(current_actor_params, inputs, sample_actions)
            return jnp.mean(actor_losses)

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        # Compute critic loss and gradients
        critic_loss, critic_grads = value_and_grad(_batch_critic_loss_function)(
            current_critic_params, 
            inputs,
            critic_targets,
        )
        # Compute actor loss and gradients
        actor_loss, actor_grads = value_and_grad(_batch_actor_loss_function)(
            current_actor_params, 
            inputs,
            sample_actions,
        )
        return critic_loss, critic_grads, actor_loss, actor_grads, 0.

    @partial(jit, static_argnames=("self"))
    def _segment_rectangle_intersection(self, x1, y1, x2, y2, xmin, xmax, ymin, ymax):
        """
        This is the Liang-Barsky algorithm for line clipping.
        """
        @jit
        def _nan_segment(val):
            return False, jnp.array([jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan])
        @jit
        def _not_nan_segment(val):
            x1, y1, x2, y2, xmin, xmax, ymin, ymax = val
            dx = x2 - x1
            dy = y2 - y1
            p = jnp.array([-dx, dx, -dy, dy])
            q = jnp.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1])
            @jit
            def loop_body(i, tup):
                t, p, q = tup
                t0, t1 = t
                t0, t1 = lax.switch(
                    (jnp.sign(p[i])+1).astype(jnp.int32),
                    [
                        lambda t: lax.cond(q[i]/p[i] > t[1], lambda _: (2.,1.), lambda x: (jnp.max(jnp.array([x[0],q[i]/p[i]])), x[1]), t),  # p[i] < 0
                        lambda t: lax.cond(q[i] < 0, lambda _: (2.,1.), lambda x: x, t),  # p[i] == 0
                        lambda t: lax.cond(q[i]/p[i] < t[0], lambda _: (2.,1.), lambda x: (x[0], jnp.min(jnp.array([x[1],q[i]/p[i]]))), t),  # p[i] > 0
                    ],
                    (t0, t1),
                )
                return ((t0, t1), p ,q)
            t, p, q = lax.fori_loop(
                0, 
                4,
                loop_body,
                ((0., 1.), p, q),
            )
            t0, t1 = t
            inside_or_intersects = ~(t0 > t1)
            intersection_point_0 = lax.switch(
                jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t0 == 0), (inside_or_intersects) & (t0 > 0)])),
                [
                    lambda _: jnp.array([jnp.nan, jnp.nan]),
                    lambda _: jnp.array([x1, y1]),
                    lambda _: jnp.array([x1 + t0 * dx, y1 + t0 * dy]),
                ],
                None,
            )
            intersection_point_1 = lax.switch(
                jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t1 == 1), (inside_or_intersects) & (t1 < 1)])),
                [
                    lambda _: jnp.array([jnp.nan, jnp.nan]),
                    lambda _: jnp.array([x2, y2]),
                    lambda _: jnp.array([x1 + t1 * dx, y1 + t1 * dy]),
                ],
                None,
            )
            return inside_or_intersects, intersection_point_0, intersection_point_1
        return lax.cond(
            jnp.any(jnp.isnan(jnp.array([x1, y1, x2, y2]))),
            _nan_segment,
            _not_nan_segment,
            (x1, y1, x2, y2, xmin, xmax, ymin, ymax),
        )

    @partial(jit, static_argnames=("self"))
    def _batch_segment_rectangle_intersection(self, x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax):
        return vmap(JESSI._segment_rectangle_intersection, in_axes=(None,0,0,0,0,None,None,None,None))(self, x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax)
    
    @partial(jit, static_argnames=("self"))
    def _process_obs_stack(self, obs_stack, ref_position, ref_orientation):
        """
        args:
        - obs_stack (lidar_num_rays + 6):  [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        """
        ## Split obs stack
        robot_position = obs_stack[:2]  # Shape: (2,)
        robot_orientation = obs_stack[2]  # Shape: ()
        robot_radius = obs_stack[3]  # Shape: ()
        robot_action = obs_stack[4:6]  # Shape: (2,)
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
        return jnp.concatenate((points_robot.reshape(self.lidar_num_rays * 2,), robot_action), axis=-1)

    # Public methods

    @partial(jit, static_argnames=("self"))
    def init_nns(
        self, 
        key:random.PRNGKey, 
    ) -> tuple:
        encoder_params = self.encoder.init(key, jnp.zeros((1, self.n_stack * (2 * self.lidar_num_rays + 2))))
        actor_params = self.actor.init(key, jnp.zeros((1, 3 * 6 * self.n_gmm_components)))
        critic_params = self.critic.init(key, jnp.zeros((1, 3 * 6 * self.n_gmm_components)))
        return encoder_params, actor_params, critic_params

    @partial(jit, static_argnames=("self"))
    def bound_action_space(self, lidar_scan, robot_position, robot_orientation, robot_radius):
        """
        Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
        """
        # TODO: Implement action space bounding with LiDAR scan
        pass

    @partial(jit, static_argnames=("self"))
    def process_obs(
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
        - processed_obs (n_stack * (lidar_num_rays * 2 + 2)): flattened aligned observation stack. First information corresponds to the least recent observation.
        """
        ref_position = obs[0,:2]
        ref_orientation = obs[0,2]
        return vmap(JESSI._process_obs_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)[::-1,:].flatten()

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict,
        encoder_params:dict,
        actor_params:dict, 
        sample:bool = False,
    ) -> jnp.ndarray:
        # Compute encoder input
        encoder_input = self.process_obs(obs)
        # Compute GMMs (with encoder)
        obs_distr, hum_distr, next_hum_distr = self.encoder.apply(
            encoder_params, 
            None, 
            jnp.reshape(encoder_input, (1, self.n_stack * (2 * self.lidar_num_rays + 2))),
        )
        encoder_distrs = {
            "obs_distr": obs_distr,
            "hum_distr": hum_distr,
            "next_hum_distr": next_hum_distr,
        }
        # Prepare input for actor
        robot_goal = info["robot_goal"]  # Shape: (2,)
        robot_position = obs[0,:2]
        robot_orientation = obs[0,2]
        rc_robot_goal = jnp.array([
            jnp.cos(-robot_orientation) * (robot_goal[0] - robot_position[0]) - jnp.sin(-robot_orientation) * (robot_goal[1] - robot_position[1]),
            jnp.sin(-robot_orientation) * (robot_goal[0] - robot_position[0]) + jnp.cos(-robot_orientation) * (robot_goal[1] - robot_position[1]),
        ])
        actor_input = jnp.concatenate((
            rc_robot_goal,
            obs_distr["means"].flatten(),
            obs_distr["logsigmas"].flatten(),
            obs_distr["correlations"].flatten(),
            obs_distr["weights"].flatten(),
            hum_distr["means"].flatten(),
            hum_distr["logsigmas"].flatten(),
            hum_distr["correlations"].flatten(),
            hum_distr["weights"].flatten(),
            next_hum_distr["means"].flatten(),
            next_hum_distr["logsigmas"].flatten(),
            next_hum_distr["correlations"].flatten(),
            next_hum_distr["weights"].flatten(),
        ))
        # Compute bounded action space parameters and add it to the input
        # TODO: Implement action space bounding with LiDAR scan
        # Compute action
        key, subkey = random.split(key)
        sampled_action, actor_distr = self.actor.apply(
            actor_params, 
            None, 
            actor_input, 
            random_key=subkey
        )
        action = lax.cond(sample, lambda _: sampled_action, lambda _: self.dirichlet.mean(actor_distr), None)
        return action, key, actor_input, sampled_action, encoder_distrs, actor_distr
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        encoder_params,
        actor_params,
        sample,
    ):
        return vmap(JESSI.act, in_axes=(None, 0, 0, None, None, None))(
            self,
            keys, 
            obses, 
            encoder_params, 
            actor_params, 
            sample,
        )

    @partial(jit, static_argnames=("self","actor_optimizer","critic_optimizer"))
    def update(
        self, 
        critic_params:dict, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        critic_optimizer:optax.GradientTransformation,
        critic_opt_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray], 
        beta_entropy:float,
        clip_range:float,
        debugging:bool=False,
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss = self._compute_rl_loss_and_gradients(
                critic_params, 
                actor_params,
                experiences,
                beta_entropy,
                clip_range,
                debugging=debugging, #debugging,
        )
        ## CRITIC
        # Compute parameter updates
        critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
        # Apply updates
        updated_critic_params = optax.apply_updates(critic_params, critic_updates)
        ## ACTOR
        # Compute parameter updates
        actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
        # Apply updates
        updated_actor_params = optax.apply_updates(actor_params, actor_updates)
        return (
            updated_critic_params, 
            updated_actor_params, 
            critic_opt_state, 
            actor_opt_state, 
            critic_loss, 
            actor_loss, 
            entropy_loss
        )
    
    @partial(jit, static_argnames=("self","actor_optimizer","critic_optimizer"))
    def update_il(
        self, 
        critic_params:dict, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        critic_optimizer:optax.GradientTransformation,
        critic_opt_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray], 
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss = self._compute_il_loss_and_gradients(
                critic_params, 
                actor_params,
                experiences,
        )
        ## CRITIC
        # Compute parameter updates
        critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
        # Apply updates
        updated_critic_params = optax.apply_updates(critic_params, critic_updates)
        ## ACTOR
        # Compute parameter updates
        actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
        # Apply updates
        updated_actor_params = optax.apply_updates(actor_params, actor_updates)
        return (
            updated_critic_params, 
            updated_actor_params, 
            critic_opt_state, 
            actor_opt_state, 
            critic_loss, 
            actor_loss, 
            entropy_loss
        )
    
    @partial(jit, static_argnames=("self","actor_optimizer"))
    def update_il_only_actor(
        self, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        experiences:dict[str:jnp.ndarray], 
    ) -> tuple:
        @jit
        def _compute_loss_and_gradients(
            current_actor_params:dict,  
            experiences:dict,
            # Experiences: {"inputs":dict, "actor_actions":jnp.ndarray}
        ) -> tuple:
            @jit
            def _batch_loss_function(
                current_actor_params:dict,
                inputs:jnp.ndarray,
                sample_actions:jnp.ndarray,
                ) -> jnp.ndarray:
                
                @partial(vmap, in_axes=(None, 0, 0))
                def _loss_function(
                    current_actor_params:dict,
                    input:jnp.ndarray,
                    sample_action:jnp.ndarray,
                    ) -> jnp.ndarray:
                    # Concatenate GMM parameters into a single vector as actor input
                    actor_input = jnp.concatenate((
                        input["rc_robot_goals"],
                        jnp.reshape(input["obs_distrs"]["means"], (-1,)),
                        jnp.reshape(input["obs_distrs"]["logsigmas"], (-1,)),
                        jnp.reshape(input["obs_distrs"]["correlations"], (-1,)),
                        jnp.reshape(input["obs_distrs"]["weights"], (-1,)),
                        jnp.reshape(input["hum_distrs"]["means"], (-1,)),
                        jnp.reshape(input["hum_distrs"]["logsigmas"], (-1,)),
                        jnp.reshape(input["hum_distrs"]["correlations"], (-1,)),
                        jnp.reshape(input["hum_distrs"]["weights"], (-1,)),
                        jnp.reshape(input["next_hum_distrs"]["means"], (-1,)),
                        jnp.reshape(input["next_hum_distrs"]["logsigmas"], (-1,)),
                        jnp.reshape(input["next_hum_distrs"]["correlations"], (-1,)),
                        jnp.reshape(input["next_hum_distrs"]["weights"], (-1,)),
                    ), axis=0)
                    # Compute the prediction (here we should input a key but for now we work only with mean actions)
                    _, distr = self.actor.apply(current_actor_params, None, actor_input)
                    # Get mean action
                    action = self.dirichlet.mean(distr)
                    # Compute the loss
                    return 0.5 * jnp.sum(jnp.square(action - sample_action))
                
                return jnp.mean(_loss_function(
                    current_actor_params,
                    inputs,
                    sample_actions
                ))

            inputs = experiences["inputs"]
            sample_actions = experiences["actor_actions"]
            # Compute the loss and gradients
            loss, grads = value_and_grad(_batch_loss_function)(
                current_actor_params, 
                inputs,
                sample_actions,
            )
            return loss, grads
        # Compute loss and gradients for actor and critic
        actor_loss, actor_grads = _compute_loss_and_gradients(actor_params,experiences)
        # Compute parameter updates
        actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
        # Apply updates
        updated_actor_params = optax.apply_updates(actor_params, actor_updates)
        return (
            updated_actor_params, 
            actor_opt_state, 
            actor_loss, 
        )
    
    @partial(jit, static_argnames=("self","encoder_optimizer"))
    def update_encoder(
        self,
        current_params:dict, 
        encoder_optimizer:optax.GradientTransformation, 
        optimizer_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"inputs":jnp.ndarray, "obstacles_samples":jnp.ndarray, "humans_samples":jnp.ndarray, "next_humans_samples":jnp.ndarray}
    ) -> tuple:
        @jit
        def _compute_loss_and_gradients(
            current_params:dict,  
            experiences:dict[str:jnp.ndarray],
            # Experiences: {"inputs":jnp.ndarray, "obstacles_samples":jnp.ndarray, "humans_samples":jnp.ndarray, "next_humans_samples":jnp.ndarray}
        ) -> tuple:
            @jit
            def _batch_loss_function(
                current_params:dict,
                inputs:jnp.ndarray,
                obstacles_samples:jnp.ndarray,
                humans_samples:jnp.ndarray,
                next_humans_samples:jnp.ndarray,
                ) -> jnp.ndarray:
                
                @partial(vmap, in_axes=(None, 0, 0, 0, 0))
                def _loss_function(
                    current_params:dict,
                    input:jnp.ndarray,
                    obstacles_samples:jnp.ndarray,
                    humans_samples:jnp.ndarray,
                    next_humans_samples:jnp.ndarray,
                    ) -> jnp.ndarray:
                    # Compute the prediction
                    input = jnp.reshape(input, (1, self.n_stack * (2 * self.lidar_num_rays + 2)))
                    obs_prediction, humans_prediction, next_humans_prediction = self.encoder.apply(current_params, None, input)
                    obs_prediction = {k: jnp.squeeze(v) for k, v in obs_prediction.items()}
                    humans_prediction = {k: jnp.squeeze(v) for k, v in humans_prediction.items()}
                    next_humans_prediction = {k: jnp.squeeze(v) for k, v in next_humans_prediction.items()}
                    # Compute the loss
                    loss1 = jnp.mean(self.gmm.batch_contrastivelogp(obs_prediction, obstacles_samples["position"], obstacles_samples["is_positive"]))
                    loss2 = jnp.mean(self.gmm.batch_contrastivelogp(humans_prediction, humans_samples["position"], humans_samples["is_positive"]))
                    loss3 = jnp.mean(self.gmm.batch_contrastivelogp(next_humans_prediction, next_humans_samples["position"], next_humans_samples["is_positive"]))
                    contrastive_loss = 0.5 * loss1 + 0.5 * loss2 + 0.5 * loss3
                    # Weights entropy regularization
                    obs_weights = obs_prediction["weights"]
                    hum_weights = humans_prediction["weights"]
                    next_hum_weights = next_humans_prediction["weights"]
                    eloss1 = -jnp.sum(obs_weights * jnp.log(obs_weights + 1e-8))
                    eloss2 = -jnp.sum(hum_weights * jnp.log(hum_weights + 1e-8))
                    eloss3 = -jnp.sum(next_hum_weights * jnp.log(next_hum_weights + 1e-8))
                    entropy_loss = 1e-3 * (eloss1 + eloss2 + eloss3)
                    # debug.print("nll_loss: {x} - entropy_loss: {y}", x=nll_loss, y=entropy_loss)
                    return contrastive_loss + entropy_loss
                
                return jnp.mean(_loss_function(
                    current_params,
                    inputs,
                    obstacles_samples,
                    humans_samples,
                    next_humans_samples
                ))

            inputs = experiences["inputs"]
            obstacles_samples = experiences["obstacles_samples"]
            humans_samples = experiences["humans_samples"]
            next_humans_samples = experiences["next_humans_samples"]
            # Compute the loss and gradients
            loss, grads = value_and_grad(_batch_loss_function)(
                current_params, 
                inputs,
                obstacles_samples,
                humans_samples,
                next_humans_samples
            )
            return loss, grads
        # Compute loss and gradients
        loss, grads = _compute_loss_and_gradients(current_params, experiences)
        # Compute parameter updates
        updates, optimizer_state = encoder_optimizer.update(grads, optimizer_state)
        # Apply updates
        updated_params = optax.apply_updates(current_params, updates)
        return updated_params, optimizer_state, loss