import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from jax.tree_util import tree_map
from functools import partial
import haiku as hk
import optax
import os
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

from socialjym.envs.base_env import ROBOT_KINEMATICS, EPSILON
from socialjym.utils.distributions.dirichlet import Dirichlet
from socialjym.utils.distributions.gaussian import BivariateGaussian
from socialjym.policies.base_policy import BasePolicy
from jhsfm.hsfm import get_linear_velocity

class SinusoidalPositionalEncoding(hk.Module):
    def __init__(self, d_model, max_len=5000, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_len = max_len

    def __call__(self, x):
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.shape[1]
        position = jnp.arange(seq_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[None, :, :]

class SpatioTemporalEncoder(hk.Module):
    def __init__(self, embed_dim=128, name=None): 
        super().__init__(name=name)
        self.embed_dim = embed_dim
        # 1. Spatial Feature Extraction
        self.spatial_net = hk.Sequential([
            hk.Conv1D(output_channels=64, kernel_shape=5, stride=1, padding="SAME"),
            nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            
            hk.Conv1D(output_channels=128, kernel_shape=3, stride=2, padding="SAME"),
            nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            
            hk.Conv1D(output_channels=embed_dim, kernel_shape=3, stride=1, padding="SAME"),
            nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        ])
        # 2. Positional Encodings
        self.pos_encoder_space = SinusoidalPositionalEncoding(embed_dim, max_len=2048)
        self.pos_encoder_time = SinusoidalPositionalEncoding(embed_dim, max_len=100)
        # 3. Temporal Attention
        self.temporal_attn = hk.MultiHeadAttention(num_heads=4, key_size=32, w_init_scale=1.0)
        self.temporal_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x):
        # x shape: [Batch, Time, Beams, Features]
        B, T, L, F = x.shape
        x_flat = x.reshape(B * T, L, F) 
        h_spatial = self.spatial_net(x_flat) # [B*T, L/2, embed_dim] (nota L/2 per stride)
        h_spatial = self.pos_encoder_space(h_spatial)
        L_new = h_spatial.shape[1]
        h_spatial = h_spatial.reshape(B, T, L_new, self.embed_dim)
        h_time_in = h_spatial.transpose(0, 2, 1, 3).reshape(B * L_new, T, self.embed_dim)
        h_time_in = self.pos_encoder_time(h_time_in)
        t_out = self.temporal_attn(query=h_time_in, key=h_time_in, value=h_time_in)
        h_time_out = self.temporal_norm(h_time_in + t_out)
        h_final = h_time_out.reshape(B, L_new, T, self.embed_dim).transpose(0, 2, 1, 3)
        return h_final

class HCGQueryDecoder(hk.Module):
    def __init__(self, n_detectable_humans, embed_dim, name=None):
        super().__init__(name=name)
        self.n_detectable_humans = n_detectable_humans
        self.embed_dim = embed_dim
        # LEARNABLE QUERIES: DETR-like.
        self.query_embeddings = hk.get_parameter(
            "query_embeddings", 
            shape=[1, self.n_detectable_humans, embed_dim], 
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        self.cross_attn = hk.MultiHeadAttention(num_heads=4, key_size=embed_dim//4, value_size=embed_dim//4, w_init_scale=1.0, model_size=embed_dim)
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.nets.MLP([embed_dim * 2, embed_dim], activation=nn.gelu, activate_final=False)
        self.norm_ffn = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, encoder_output):
        # encoder_output: [B, T, L, D]
        B, T, L, D = encoder_output.shape
        kv = encoder_output.reshape(B, T * L, D)
        q = jnp.tile(self.query_embeddings, (B, 1, 1))
        # Cross Attention
        attn_out = self.cross_attn(query=q, key=kv, value=kv)
        q = self.norm(q + attn_out)
        ffn_out = self.ffn(q)
        q = self.norm_ffn(q + ffn_out)
        
        return q # [B, n_detectable_humans, D]

class Perception(hk.Module):
    def __init__(
            self,
            n_detectable_humans: int,
            max_humans_velocity: float,
            max_lidar_distance: float,
            embed_dim: int = 128
        ):
        super().__init__(name="lidar_perception")
        self.n_detectable_humans = n_detectable_humans
        self.embed_dim = embed_dim
        self.max_humans_velocity = max_humans_velocity
        self.max_lidar_distance = max_lidar_distance
        # Modules
        self.input_proj = hk.Linear(embed_dim, name="input_projection")
        self.perception = SpatioTemporalEncoder(embed_dim=embed_dim, name="spatio_temporal_encoder")
        self.decoder = HCGQueryDecoder(n_detectable_humans, embed_dim, name="decoder")
        # Head: 11 params (weight, mu_x, mu_y, log_sig_x, log_sig_y, corr, mu_vx, mu_vy, log_sig_vx, log_sig_vy, corr_v)
        self.head_hum = hk.Linear(11)

    def limit_vector_norm(self, raw_vector:jnp.ndarray, max_norm:float) -> jnp.ndarray:
        current_norm = jnp.linalg.norm(raw_vector, axis=-1, keepdims=True) + 1e-6
        target_norm = jnp.tanh(current_norm) * max_norm
        return (raw_vector / current_norm) * target_norm

    def _params_to_human_centric_gaussians(self, raw_params):
        """
        Takes the raw network outputs and transforms them into valid human-centered Gaussian parameters.
        """
        raw_pos = raw_params[..., :2]
        means_pos = self.limit_vector_norm(raw_pos, self.max_lidar_distance)
        log_sig_x = 5.0 * nn.tanh(raw_params[..., 2])
        log_sig_y = 5.0 * nn.tanh(raw_params[..., 3])
        rho_pos   = 0.99 * nn.tanh(raw_params[..., 4])
        raw_vel = raw_params[..., 5:7]
        mean_vel = self.limit_vector_norm(raw_vel, self.max_humans_velocity)
        v_log_sig_x = 5.0 * nn.tanh(raw_params[..., 7])
        v_log_sig_y = 5.0 * nn.tanh(raw_params[..., 8])
        v_rho       = 0.99 * nn.tanh(raw_params[..., 9])
        w_logits = raw_params[..., 10]
        weights = nn.sigmoid(w_logits)
        return {
            "pos_distrs": {
                "means": means_pos,
                "logsigmas": jnp.stack([log_sig_x, log_sig_y], axis=-1),
                "correlation": rho_pos
            },
            "vel_distrs": {
                "means": mean_vel,
                "logsigmas": jnp.stack([v_log_sig_x, v_log_sig_y], axis=-1),
                "correlation": v_rho
            },
            "weights": weights
        }

    def __call__(self, x):
        # x input: [n_stack, num_beams, 7]
        has_batch = x.ndim == 4
        if not has_batch:
            x = jnp.expand_dims(x, 0) # [1, T, L, F]
        # 1. Feature Projection
        x_emb = self.input_proj(x) # [B, T, L, embed_dim]
        # 2. Spatio-Temporal Encoding
        encoded_features = self.perception(x_emb) # [B, T, L, embed_dim]
        # 3. Decoding via Queries
        latents = self.decoder(encoded_features) # [B, n_detectable_humans, embed_dim]        
        # 4. Heads & Parameter Transformation
        raw_hum = self.head_hum(latents)
        hum_distr = self._params_to_human_centric_gaussians(raw_hum)
        if not has_batch:
            hum_distr = tree_map(lambda t: t[0], hum_distr)
        return hum_distr

class Actor(hk.Module):
    def __init__(
            self,
            n_detectable_humans:int,
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
        self.n_detectable_humans = n_detectable_humans
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.n_inputs = (3, self.n_detectable_humans, 7+9) # (3, Number of GMM compoents, 7 params per GMM component + 9 robot goal, robot vmax, robot radius, robot wheels distance, action space params)
        self.n_outputs = 3 # Dirichlet distribution over 3 action vertices
        # Obstacles MLPs
        self.features_mlp_obs = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50, 50], 
            name="features_mlp_obs"
        )
        # Humans MLPs
        self.embedding_mlp_hum = hk.nets.MLP(
            activation=nn.relu,
            activate_final=True,
            w_init=hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            b_init=hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            output_sizes=[150, 100], 
            name="embedding_mlp_hum"
        )
        self.hum_key_mlp = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50], 
            name="hum_key_mlp"
        )
        self.hum_query_mlp = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50], 
            name="hum_query_mlp"
        )
        self.hum_value_mlp = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50], 
            name="hum_value_mlp"
        )
        # Output MLP
        self.output_mlp = hk.nets.MLP(
            **mlp_params,
            output_sizes=[150, 100, 100, self.n_outputs], 
            name="output_mlp"
        )
        self.dirichlet = Dirichlet()

    def __call__(
            self, 
            x,
            **kwargs:dict,
        ) -> jnp.ndarray:
        """
        Self-attention based actor that maps GMM parameters to Dirichlet distribution over action space vertices.
        Obstacles GMM is passed through a standard MLP to extract features.
        Humans GMM (current and next) are passed through a self-attention mechanism to extract relationship between humans motions.
        The features of each GMM are weighted by the corresponding GMM weights to obtain a one-dimensional feature vector for obstacles and on for humans.
        This encodes the fact that the weight of each GMM component represents its probability of being a real object/human.
        Both features vectors are finally concatenated with robot state and passed through an MLP to compute Dirichlet distribution parameters.
        """
        ## Extract robot state and random key for sampling
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        robot_state = x[0,0, 7:]
        action_space_params = robot_state[:3]
        ## Compute obstcles attentive embedding
        obstacles_input = x[0,:,:6]  # Shape: (n_detectable_humans, 6)
        obstacles_weights = x[0,:,6] # Shape: (n_detectable_humans,)
        obstacles_features = self.features_mlp_obs(obstacles_input)  # Shape: (n_detectable_humans, obs embedding_size)
        weighted_obstacles_features = jnp.sum(jnp.multiply(obstacles_weights[:,None], obstacles_features), axis=0) # Shape: (obs feature_size,)
        ## Compute human attentive embedding
        humans_input = jnp.concatenate([x[1,:,:6], x[2,:,:6]], axis=-1)  # Shape: (n_detectable_humans, 7)
        humans_weights = x[1,:,6]  # Shape: (n_detectable_humans, 1)
        # next_humans_weights = x[2,:,6]  # Shape: (n_detectable_humans, 1) - Currently these are the same as humans_weights
        humans_embeddings = self.embedding_mlp_hum(humans_input)  # Shape: (n_detectable_humans, hum embedding_size)
        humans_keys = self.hum_key_mlp(humans_embeddings)  # Shape: (n_detectable_humans, key_size)
        humans_queries = self.hum_query_mlp(humans_embeddings)  # Shape: (n_detectable_humans, query_size)
        humans_values = self.hum_value_mlp(humans_embeddings)  # Shape: (n_detectable_humans, value_size)
        humans_attention_matrix = jnp.dot(humans_queries, humans_keys.T) / jnp.sqrt(humans_keys.shape[-1])  # Shape: (n_detectable_humans, n_detectable_humans)
        humans_attention_matrix = nn.softmax(humans_attention_matrix, axis=-1)  # Shape: (n_detectable_humans, n_detectable_humans)
        humans_features = jnp.dot(humans_attention_matrix, humans_values)  # Shape: (n_detectable_humans, value_size)
        weighted_humans_features = jnp.sum(jnp.multiply(humans_weights[:,None], humans_features), axis=0)  # Shape: (hum feature_size,)
        ## Concatenate weighted features
        weighted_features = jnp.concatenate((weighted_obstacles_features, weighted_humans_features), axis=-1) # Shape: (obs feature_size + hum feature_size,)
        ## Compute Dirichlet distribution parameters
        alphas = self.output_mlp(jnp.concatenate([robot_state, weighted_features], axis=0))
        alphas = nn.softplus(alphas) + 1
        ## Compute dirchlet distribution vetices
        vertices = jnp.array([
            [0., action_space_params[1] * self.wmax],
            [0., action_space_params[2] * self.wmin],
            [action_space_params[0] * self.vmax, 0.]
        ])
        distribution = {"alphas": alphas, "vertices": vertices}
        ## Sample action
        sampled_action = self.dirichlet.sample(distribution, random_key)
        return sampled_action, distribution
    
class Critic(hk.Module):
    def __init__(
            self,
            n_detectable_humans:int,
            mlp_params:dict={
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
        ) -> None:
        super().__init__(name="critic_network") 
        self.n_detectable_humans = n_detectable_humans
        self.n_inputs = 3 * 6 * self.n_detectable_humans + 5  # 6 outputs per GMM cell (mean_x, mean_y, sigma_x, sigma_y, correlation, weight) times  3 GMMs (obstacles, current humans, next humans)
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
        n_detectable_humans:int=10,
        max_humans_velocity:float=1.5,
        max_beam_range:float=10.0, # This is only used to normalize the LiDAR readings before feeding them to the encoder
    ) -> None:
        """
        JESSI (JAX-based E2E Safe Social Interpretable autonomous navigation).
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
        assert n_detectable_humans >= 2, "Number of detectable humans must be at least 2"
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
        self.n_detectable_humans = n_detectable_humans
        self.max_humans_velocity = max_humans_velocity
        self.max_beam_range = max_beam_range
        # Default attributes
        self.name = "JESSI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.dirichlet = Dirichlet()
        self.bivariate_gaussian = BivariateGaussian()
        # Initialize Perception network
        @hk.transform
        def perception_network(x):
            net = Perception(self.n_detectable_humans, self.max_humans_velocity, self.lidar_max_dist)
            return net(x)
        self.perception = perception_network
        # Initialize Actor
        @hk.transform
        def actor_network(x, **kwargs) -> jnp.ndarray:
            actor = Actor(self.n_detectable_humans, self.v_max, self.wheels_distance) 
            return actor(x, **kwargs)
        self.actor = actor_network
        # Initialize Critic
        @hk.transform
        def critic_network(x) -> jnp.ndarray:
            critic = Critic(self.n_detectable_humans) 
            return critic(x)
        self.critic = critic_network

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
    def init_nns(
        self, 
        key:random.PRNGKey, 
    ) -> tuple:
        perception_params = self.perception.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 7)))
        # actor_params = self.actor.init(key, jnp.zeros((self.n_detectable_humans, 11 + 9)))  # 7 params per GMM component + 9 robot goal, robot vmax, robot radius, robot wheels distance, action space params
        # critic_params = 
        return perception_params, {}, {}

    @partial(jit, static_argnames=("self"))
    def bound_action_space(self, lidar_point_cloud):
        """
        Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
        WARNING: Assumes LiDAR orientations is align with robot frame.
        """
        # Construct segments by connecting consecutive LiDAR points
        x1s = lidar_point_cloud[:,0]
        y1s = lidar_point_cloud[:,1]
        x2s = jnp.roll(lidar_point_cloud[:,0], -1)
        y2s = jnp.roll(lidar_point_cloud[:,1], -1)
        # Lower ALPHA
        _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
            x1s,
            y1s,
            x2s,
            y2s,
            # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
            0. + 1e-6, # xmin
            self.v_max * self.dt + self.robot_radius - 1e-6, # xmax
            -self.robot_radius + 1e-6, # ymin
            self.robot_radius - 1e-6, # ymax
        )
        intersection_points = jnp.vstack((intersection_points0, intersection_points1))
        min_x = jnp.nanmin(intersection_points[:,0])
        new_alpha = lax.cond(
            ~jnp.isnan(min_x),
            lambda _: jnp.max(jnp.array([0, min_x - self.robot_radius])) / (self.v_max * self.dt),
            lambda _: 1.,
            None,
        )
        @jit
        def _lower_beta_and_gamma(tup:tuple):
            x1s, y1s, x2s, y2s, new_alpha, vmax, wheels_distance, dt = tup
            # Lower BETA
            _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
                x1s,
                y1s,
                x2s,
                y2s,
                # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
                -self.robot_radius + 1e-6, # xmin
                new_alpha * vmax * dt + self.robot_radius - 1e-6, # xmax
                self.robot_radius + 1e-6, # ymin
                self.robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance)) - 1e-6, # ymax
            )
            intersection_points = jnp.vstack((intersection_points0, intersection_points1))
            min_y = jnp.nanmin(intersection_points[:,1])
            new_beta = lax.cond(
                ~jnp.isnan(min_y),
                lambda _: (min_y - self.robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            # Lower GAMMA
            _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
                x1s,
                y1s,
                x2s,
                y2s,
                # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
                -self.robot_radius + 1e-6, # xmin
                new_alpha * vmax * dt + self.robot_radius - 1e-6, # xmax
                -self.robot_radius - (new_alpha*dt**2*vmax**2/(4*wheels_distance)) + 1e-6, # ymin
                -self.robot_radius - 1e-6, # ymax
            )
            intersection_points = jnp.vstack((intersection_points0, intersection_points1))
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
            (x1s, y1s, x2s, y2s, new_alpha, self.v_max, self.wheels_distance, self.dt)
        )
        # Apply lower blound to new_alpha, new_beta, new_gamma
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
        - processed_obs (n_stack, lidar_num_rays * 2): aligned LiDAR stack. First information corresponds to the most recent observation.
        """
        ref_position = obs[0,:2]
        ref_orientation = obs[0,2]
        return vmap(JESSI._align_lidar_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)

    @partial(jit, static_argnames=("self"))
    def compute_actor_input(
        self,
        hcgs,
        action_space_params,
        robot_goal, # In cartesian coordinates (gx, gy) IN THE ROBOT FRAME
    ):
        # Compute ROBOT state inputs
        robot_goal_dist = jnp.linalg.norm(robot_goal)
        robot_goal_theta = jnp.arctan2(robot_goal[1], robot_goal[0])
        robot_goal_sin_theta = jnp.sin(robot_goal_theta)
        robot_goal_cos_theta = jnp.cos(robot_goal_theta)
        tiled_action_space_params = jnp.tile(action_space_params, (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        tiled_robot_params = jnp.tile(jnp.array([self.v_max, self.robot_radius, self.wheels_distance]), (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        tiled_robot_goals = jnp.tile(jnp.array([robot_goal_dist, robot_goal_sin_theta, robot_goal_cos_theta]), (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        actor_input = jnp.concatenate((
            hcgs,
            tiled_robot_goals,
            tiled_robot_params,
            tiled_action_space_params,
        ), axis=-1)  # Shape: (n_detectable_humans, 7 + 9)
        return actor_input

    @partial(jit, static_argnames=("self"))
    def compute_encoder_input(
        self,
        obs:jnp.ndarray,
    ):
        """
        Prepare the input for the encoder network.

        args:
        - obs (n_stack, lidar_num_rays + 6): Each stack [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        The first stack is the most recent one.

        output:
        - lidar_tokens (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
        7 features per token: [norm_dist, hit, x, y, distance, sin_fixed_theta, cos_fixed_theta, delta_t].
        - last LiDAR point cloud (lidar_num_rays, 2): in robot frame of the most recent observation.
        """
        # Align LiDAR scans - (x,y) coordinates of pointcloud in the robot frame, first information corresponds to the most recent observation.
        aligned_lidar_scans = self.align_lidar(obs)[0]  # Shape: (n_stack, lidar_num_rays, 2)
        last_lidar_point_cloud = aligned_lidar_scans[0,:, :]  # Shape: (lidar_num_rays, 2)
        # Compute LiDAR tokens
        @jit
        def compute_beam_token(
            scan_index:int,
            point:jnp.ndarray,
            fixed_theta:float,
        ) -> jnp.ndarray:
            # Extract point coordinates
            x, y = point
            # Compute beam features
            distance = jnp.linalg.norm(point)
            sin_fixed_theta = jnp.sin(fixed_theta)
            cos_fixed_theta = jnp.cos(fixed_theta)
            hit = jnp.where(distance < self.lidar_max_dist, 1.0, 0.0)
            # Compute stack index features
            delta_t = scan_index * self.dt
            return jnp.array([
                distance/self.max_beam_range,  # Normalize distance
                hit,
                x,
                y,
                sin_fixed_theta,
                cos_fixed_theta,
                delta_t,
            ])
        encoder_input = vmap(vmap(compute_beam_token, in_axes=(None, 0, 0)), in_axes=(0, 0, None))(
            jnp.arange(self.n_stack),
            aligned_lidar_scans,
            self.lidar_angles_robot_frame,
        )  # Shape: (n_stack, lidar_num_rays, 7)
        # Optionally select TOP K beams for each stack here to reduce computation

        return encoder_input, last_lidar_point_cloud

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
        # Compute encoder input and last lidar point cloud (for action bounding)
        encoder_input, last_lidar_point_cloud = self.compute_encoder_input(obs)
        # Compute GMMs (with encoder)
        encoder_distrs = self.perception.apply(
            encoder_params, 
            None, 
            encoder_input,
        )
        # Compute bounded action space parameters and add it to the input
        bounding_parameters = self.bound_action_space(
            last_lidar_point_cloud,  
        )
        # debug.print("Bounding parameters: {x}", x=bounding_parameters)
        # Prepare input for actor
        robot_goal = info["robot_goal"]  # Shape: (2,)
        robot_position = obs[0,:2]
        robot_orientation = obs[0,2]
        c, s = jnp.cos(-robot_orientation), jnp.sin(-robot_orientation)
        R = jnp.array([[c, -s],
                    [s,  c]])
        translated_position = robot_goal - robot_position
        rc_robot_goal = R @ translated_position
        # debug.print("Goal coords: {x}", x=rc_robot_goal)
        actor_input = self.compute_actor_input(
            encoder_distrs["obs_distr"],
            encoder_distrs["hum_distr"],
            encoder_distrs["next_hum_distr"],
            bounding_parameters,
            rc_robot_goal,
        )
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
    
    @partial(jit, static_argnames=("self"))
    def encoder_loss(
        self,
        current_params:dict,
        inputs:jnp.ndarray,
        targets:jnp.ndarray,
        lambda_pos_reg:float=5.0,
        lambda_vel_reg:float=5.0,
        lambda_cls:float=1.0,
        ) -> jnp.ndarray:
        # B: batch size, K: number of HCGs, M: max number of ground truth humans
        # Compute the prediction
        human_distrs = self.perception.apply(current_params, None, inputs)
        # Extract target data
        human_positions = targets["gt_poses"] # Shape: (B, M, 2)
        human_velocities = targets["gt_vels"] # Shape: (B, M, 2)
        human_mask = targets["gt_mask"] # Shape: (B, M) -> 1 if human exists, 0 otherwise
        # Extract dimensions
        B, K, _ = human_distrs['pos_distrs']['means'].shape
        _, M, _ = human_positions.shape
        ### Bipartite matching
        ## Cost matrix
        dist = jnp.linalg.norm( # (B, K, 1, 2) - (B, 1, M, 2)
            jnp.expand_dims(human_distrs['pos_distrs']['means'], 2) - jnp.expand_dims(human_positions, 1), 
            axis=-1
        ) # Shape (B, K, M)
        prob_cost = -jnp.log(jnp.expand_dims(human_distrs['weights'], 2) + 1e-6) # (B, K, 1)
        cost_matrix = lambda_pos_reg * dist + lambda_cls * prob_cost # (B, K, M)
        ## Matching
        assigned_query_idx, assigned_gt_idx = vmap(optax.assignment.hungarian_algorithm)(cost_matrix)
        sort_perm = jnp.argsort(assigned_gt_idx, axis=1) # Shape (B, M)
        best_pred_idx = jnp.take_along_axis(assigned_query_idx, sort_perm, axis=1) # Shape (B, M)
        # One-hot mask - shape: (B, K, M) -> 1 if k matches m, 0 otherwise
        matched_mask = nn.one_hot(best_pred_idx, K, axis=1) # Shape: (B, K, M)
        # Filter with GT mask
        valid_matches = matched_mask * jnp.expand_dims(human_mask, 1) # (B, K, M)
        matched_pos_means = jnp.einsum('bkm,bkd->bmd', valid_matches, human_distrs['pos_distrs']['means']) # (B, M, 2)
        matched_pos_logsigmas = jnp.einsum('bkm,bkd->bmd', valid_matches, human_distrs['pos_distrs']['logsigmas']) # (B, M, 2)
        matched_pos_correlations = jnp.einsum('bkm,bk->bm', valid_matches, human_distrs['pos_distrs']['correlation']) # (B, M)
        matched_pos_distrs = {
            "means": matched_pos_means,
            "logsigmas": matched_pos_logsigmas,
            "correlation": matched_pos_correlations,
        }
        matched_vel_means = jnp.einsum('bkm,bkd->bmd', valid_matches, human_distrs['vel_distrs']['means']) # (B, M, 2)
        matched_vel_logsigmas = jnp.einsum('bkm,bkd->bmd', valid_matches, human_distrs['vel_distrs']['logsigmas']) # (B, M, 2)
        matched_vel_correlations = jnp.einsum('bkm,bk->bm', valid_matches, human_distrs['vel_distrs']['correlation']) # (B, M)
        matched_vel_distrs = {
            "means": matched_vel_means,
            "logsigmas": matched_vel_logsigmas,
            "correlation": matched_vel_correlations,
        }
        ### REGRESSION LOSS
        ## NLL loss for position distribution
        pos_nll_losses = vmap(vmap(self.bivariate_gaussian.neglogp))(
            matched_pos_distrs, # Shape: (B, M, 1/2)
            human_positions, # Shape: (B, M, 2)
        )  # Shape: (B, M)
        # Mask invalid humans
        pos_reg_loss = jnp.sum(pos_nll_losses * human_mask) / (jnp.sum(human_mask) + 1e-6)
        ## NLL loss for velocity distribution
        vel_nll_losses = vmap(vmap(self.bivariate_gaussian.neglogp))(
            matched_vel_distrs, # Shape: (B, M, 1/2)
            human_velocities, # Shape: (B, M, 2)
        )  # Shape: (B, M)
        # Mask invalid humans
        vel_reg_loss = jnp.sum(vel_nll_losses * human_mask) / (jnp.sum(human_mask) + 1e-6)
        ### CLASSIFICATION LOSS
        ## Binary cross-entropy loss for classification
        target_cls = jnp.max(valid_matches, axis=2) # (B, K) -> 0 o 1
        bce = - (target_cls * jnp.log(human_distrs['weights'] + 1e-6) + (1 - target_cls) * jnp.log(1 - human_distrs['weights'] + 1e-6))
        cls_loss = jnp.mean(bce) 
        return lambda_pos_reg * pos_reg_loss + lambda_vel_reg * vel_reg_loss + lambda_cls * cls_loss

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
    ):
        """
        Update both actor and critic networks using the provided experiences in RL setting.
        """
        pass

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
                    actor_input = self.compute_actor_input(
                        input["obs_distrs"],
                        input["hum_distrs"],
                        input["next_hum_distrs"],
                        input["action_space_params"],
                        input["rc_robot_goals"],
                    )
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
        # Experiences: {"inputs":jnp.ndarray, "targets":dict{"gt_mask","gt_poses","gt_vels"}}
    ) -> tuple:
        @jit
        def _compute_loss_and_gradients(
            current_params:dict,  
            experiences:dict[str:jnp.ndarray],
        ) -> tuple:

            inputs = experiences["inputs"]
            targets = experiences["targets"]
            # Compute the loss and gradients
            loss, grads = value_and_grad(self.encoder_loss)(
                current_params, 
                inputs,
                targets,
            )
            return loss, grads
        # Compute loss and gradients
        loss, grads = _compute_loss_and_gradients(current_params, experiences)
        # Compute parameter updates
        updates, optimizer_state = encoder_optimizer.update(grads, optimizer_state)
        # Apply updates
        updated_params = optax.apply_updates(current_params, updates)
        return updated_params, optimizer_state, loss    

    def animate_trajectory(
        self,
        robot_poses, # x, y, theta
        robot_actions,
        robot_goals,
        observations,
        actor_distrs,
        humans_distrs,
        humans_poses, # x, y, theta
        humans_velocities, # vx, vy (in global frame)
        humans_radii,
        static_obstacles,
        p_visualization_threshold_hcgs:float=0.1,
        p_visualization_threshold_dir:float=0.05,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        # Validate input args
        assert \
            len(robot_poses) == \
            len(robot_actions) == \
            len(robot_goals) == \
            len(observations) == \
            len(actor_distrs['alphas']) == \
            len(humans_distrs['pos_distr']['means']) == \
            len(humans_poses) == \
            len(humans_velocities) == \
            len(humans_radii) == \
            len(static_obstacles), "All inputs must have the same length"
        # Set matplotlib fonts
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Compute informations for visualization
        n_steps = len(robot_poses)
        angs = jnp.linspace(0, 2*jnp.pi, 20, endpoint=False)
        dists = jnp.linspace(0, 1, 10)
        gauss_samples = jnp.array(jnp.meshgrid(angs, dists)).T.reshape(-1, 2)
        from socialjym.policies.cadrl import CADRL
        from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
        dummy_cadrl = CADRL(DummyReward(kinematics="unicycle"),kinematics="unicycle",v_max=self.v_max,wheels_distance=self.wheels_distance)
        test_action_samples = dummy_cadrl._build_action_space(unicycle_triangle_samples=35)
        # Animate trajectory
        fig = plt.figure(figsize=(16,8), layout='constrained')
        fig.subplots_adjust(left=0.02, right=0.99, wspace=0.08, hspace=0.2, top=0.95, bottom=0.07)
        gs = fig.add_gridspec(2, 3)
        axs = [
            fig.add_subplot(gs[0,0]), # Simulation + LiDAR ranges
            fig.add_subplot(gs[0,1]), # Simulation + LiDAR point cloud stack
            fig.add_subplot(gs[1,0]), # Human-centric Gaussians positions
            fig.add_subplot(gs[1,1]), # Human-centric Gaussians velocities
            fig.add_subplot(gs[:,2]), # Action space distribution/bounding + action taken
        ]
        def animate(frame):
            for ax in enumerate(axs[:-1]): # All except last (Action space)
                ax.clear()
                ax.set(xlim=x_lims if x_lims is not None else [-10,10], ylim=y_lims if y_lims is not None else [-10,10])
                ax.set_xlabel('X', labelpad=-5)
                ax.set_ylabel('Y', labelpad=-13)
                ax.set_aspect('equal', adjustable='box')
                # Plot humans
                for h in range(len(humans_poses[frame])):
                    head = plt.Circle((humans_poses[frame][h,0] + jnp.cos(humans_poses[frame][h,2]) * humans_radii[frame][h], humans_poses[frame][h,1] + jnp.sin(humans_poses[frame][h,2]) * humans_radii[frame][h]), 0.1, color='black', alpha=0.6, zorder=1)
                    ax.add_patch(head)
                    circle = plt.Circle((humans_poses[frame][h,0], humans_poses[frame][h,1]), humans_radii[frame][h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                    ax.add_patch(circle)
                # Plot human velocities
                for h in range(len(humans_poses[frame])):
                    ax.arrow(
                        humans_poses[frame][h,0],
                        humans_poses[frame][h,1],
                        humans_velocities[frame][h,0],
                        humans_velocities[frame][h,1],
                        head_width=0.15,
                        head_length=0.15,
                        fc="blue",
                        ec="blue",
                        alpha=0.6,
                        zorder=30,
                    )
                # Plot robot
                robot_position = robot_poses[frame,:2]
                head = plt.Circle((robot_position[0] + self.robot_radius * jnp.cos(robot_poses[frame,2]), robot_position[1] + self.robot_radius * jnp.sin(robot_poses[frame,2])), 0.1, color='black', zorder=1)
                ax.add_patch(head)
                circle = plt.Circle((robot_position[0], robot_position[1]), self.robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
                ax.add_patch(circle)
                # Plot robot goal
                ax.plot(
                    robot_goals[frame][0],
                    robot_goals[frame][1],
                    marker='*',
                    markersize=7,
                    color='red',
                    zorder=5,
                )
                # Plot static obstacles
                for o in static_obstacles[frame]:
                    for s in o:
                        ax.plot(s[:,0],s[:,1], color='black', linewidth=2, zorder=11, alpha=0.6, linestyle='solid')
            ### FIRST ROW AXS: SIMULATION + INPUT VISUALIZATION
            c, s = jnp.cos(robot_poses[frame,2]), jnp.sin(robot_poses[frame,2])
            rot = jnp.array([[c, -s], [s, c]])
            # AX 0,0: Simulation with LiDAR ranges
            lidar_scan = observations[frame,0,6:]
            for ray in range(len(lidar_scan)):
                axs[0].plot(
                    [robot_poses[frame,0], robot_poses[frame,0] + lidar_scan[ray] * jnp.cos(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    [robot_poses[frame,1], robot_poses[frame,1] + lidar_scan[ray] * jnp.sin(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    color="black", 
                    linewidth=0.5, 
                    zorder=0
                )
            axs[0].set_title("Trajectory")
            # AX 0,1: Simulation with LiDAR point cloud stack
            point_cloud = self.align_lidar(observations[frame])[1]
            for i, cloud in enumerate(point_cloud):
                # color/alpha fade with i (smaller i -> less faded)
                t = (1 - i / (self.n_stack - 1))  # in [0,1]
                axs[1].scatter(
                    cloud[:,0],
                    cloud[:,1],
                    c=0.3 + 0.7 * jnp.ones((self.lidar_num_rays,)) * t,
                    cmap='Reds',
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.3 + 0.7 * t,
                    zorder=20 + self.n_stack - i,
                )
            axs[1].set_title("Pointcloud")
            ### SECOND ROW AXS: PERCEPTION + ACTION VISUALIZATION
            frame_humans_distrs = tree_map(lambda x: x[frame], humans_distrs)
            pos_distrs = frame_humans_distrs["pos_distr"]
            vel_distrs = frame_humans_distrs["vel_distr"]
            probs = frame_humans_distrs["weights"]
            # AX 1,0 and 1,1: Human-centric Gaussians (HCGs) positions and velocities
            for h in range(len(humans_poses[frame])):
                if probs[h] > 0.5:
                    # Position HCG
                    test_p = self.bivariate_gaussian.batch_p(pos_distrs, gauss_samples)
                    points_high_p = gauss_samples[test_p > p_visualization_threshold_hcgs]
                    corresponding_colors = test_p[test_p > p_visualization_threshold_hcgs]
                    pos = pos_distrs["means"] @ rot + robot_poses[frame,:2]
                    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + pos
                    axs[2].scatter(pos[0], pos[1], c='red', s=10, marker='x', zorder=100)
                    axs[2].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
                    # Velocity HCG
                    test_p = self.bivariate_gaussian.batch_p(vel_distrs, gauss_samples)
                    points_high_p = gauss_samples[test_p > p_visualization_threshold_hcgs]
                    corresponding_colors = test_p[test_p > p_visualization_threshold_hcgs]
                    vel = vel_distrs["means"] @ rot + pos
                    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + pos
                    axs[3].scatter(vel[0], vel[1], c='red', s=10, marker='x', zorder=100)
                    axs[3].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[2].set_title("HCGs positions")
            axs[3].set_title("HCGs velocities")
            # AX :,2: Feasible and bounded action space + action space distribution and action taken
            axs[4].set_xlabel("$v$ (m/s)")
            axs[4].set_ylabel("$\omega$ (rad/s)")
            axs[4].set_xlim(-0.1, self.v_max + 0.1)
            axs[4].set_ylim(-2*self.v_max/self.wheels_distance - 0.3, 2*self.v_max/self.wheels_distance + 0.3)
            axs[4].set_xticks(jnp.arange(0, self.v_max+0.2, 0.2))
            axs[4].set_xticklabels([round(i,1) for i in jnp.arange(0, self.v_max, 0.2)] + [r"$\overline{v}$"])
            axs[4].set_yticks(jnp.arange(-2,3,1).tolist() + [2*self.v_max/self.wheels_distance,-2*self.v_max/self.wheels_distance])
            axs[4].set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
            axs[4].grid()
            axs[4].add_patch(
                plt.Polygon(
                    [   
                        [0,2*self.v_max/self.wheels_distance],
                        [0,-2*self.v_max/self.wheels_distance],
                        [self.v_max,0],
                    ],
                    closed=True,
                    fill=True,
                    edgecolor='red',
                    facecolor='lightcoral',
                    linewidth=2,
                    zorder=2,
                ),
            )
            bounded_action_space_vertices = actor_distrs["vertices"][frame]
            axs[4].add_patch(
                plt.Polygon(
                    [   
                        bounded_action_space_vertices[0],
                        bounded_action_space_vertices[1],
                        bounded_action_space_vertices[2],
                    ],
                    closed=True,
                    fill=True,
                    edgecolor='green',
                    facecolor='lightgreen',
                    linewidth=2,
                    zorder=3,
                ),
            )
            actor_distr = tree_map(lambda x: x[frame], actor_distrs)
            test_action_p = self.dirichlet.batch_p(actor_distr, test_action_samples)
            points_high_p = test_action_samples[test_action_p > p_visualization_threshold_dir]
            corresponding_colors = test_action_p[test_action_p > p_visualization_threshold_dir]
            axs[4].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[4].plot(robot_actions[frame,0], robot_actions[frame,1], marker='^',markersize=7,color='red',zorder=51)
            axs[4].set_title("Action space")
        anim = FuncAnimation(fig, animate, interval=self.dt*1000, frames=n_steps)
        if save_video:
            save_path = os.path.join(os.path.dirname(__file__), f'jessi_trajectory.mp4')
            writer_video = FFMpegWriter(fps=int(1/self.dt), bitrate=1800)
            anim.save(save_path, writer=writer_video, dpi=300)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()

    def animate_lasernav_trajectory(
        self,
        states,
        observations,
        actions,
        actor_distrs,
        humans_distrs,
        goals,
        static_obstacles,
        humans_radii,
        p_visualization_threshold_gmm:float=0.1,
        p_visualization_threshold_dir:float=0.05,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        robot_positions = states[:,-1,:2]
        robot_orientations = states[:,-1,4]
        robot_poses = jnp.hstack((robot_positions, robot_orientations.reshape(-1,1)))
        humans_positions = states[:,:-1,:2]
        humans_orientations = states[:,:-1,4]
        humans_poses = jnp.dstack((humans_positions, humans_orientations))
        humans_body_velocities = states[:,:-1,2:4]
        humans_velocities = vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(
            humans_orientations,
            humans_body_velocities,
        )
        self.animate_trajectory(
            robot_poses,
            actions,
            goals,
            observations,
            actor_distrs,
            humans_distrs,
            humans_poses,
            humans_velocities,
            humans_radii,
            static_obstacles,
            p_visualization_threshold_gmm,
            p_visualization_threshold_dir,
            x_lims,
            y_lims,
            save_video,
        )