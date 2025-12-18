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
from socialjym.utils.distributions.gaussian_mixture_model import BivariateGMM
from socialjym.policies.base_policy import BasePolicy
from jhsfm.hsfm import get_linear_velocity, vectorized_compute_obstacle_closest_point

class SpatioTemporalEncoder(hk.Module):
    def __init__(self, embed_dim=64, name=None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        
        # 1. Spatial Encoding: 1D Convolutions sono perfette per i LiDAR
        # Gestiscono numero di raggi variabile e catturano correlazioni locali.
        self.spatial_conv1 = hk.Conv1D(output_channels=embed_dim, kernel_shape=3, stride=1, padding="SAME")
        self.spatial_conv2 = hk.Conv1D(output_channels=embed_dim, kernel_shape=3, stride=1, padding="SAME")
        self.spatial_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # 2. Temporal Encoding: Self-Attention sul tempo
        self.temporal_attn = hk.MultiHeadAttention(num_heads=4, key_size=16, w_init_scale=1.0)
        self.temporal_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x):
        """
        x shape: [Batch, Time, Beams, Features] 
        Nota: Se non hai Batch dimensione, aggiungila con jnp.expand_dims(x, 0) prima di chiamare.
        """
        B, T, L, F = x.shape
        
        # --- A. Spatial Mixing (Conv1D) ---
        # Fondiamo Batch e Time per processare ogni scan indipendentemente
        x_flat = x.reshape(B * T, L, F) 
        
        # Proiezione iniziale e feature extraction locale
        h = nn.gelu(self.spatial_conv1(x_flat))
        h = self.spatial_conv2(h) # [B*T, L, embed_dim]
        
        # Residual + Norm
        # Nota: Qui assumiamo che x originale sia stato proiettato se F != embed_dim. 
        # Per semplicità, sommiamo l'output della conv direttamente (la conv cambia le feature size)
        h = self.spatial_norm(h)
        
        # Reshape back to separate Time: [B, T, L, embed_dim]
        h = h.reshape(B, T, L, self.embed_dim)

        # --- B. Temporal Mixing (Attention) ---
        # Vogliamo che ogni raggio (o feature spaziale) guardi alla sua storia.
        # Permutiamo per avere [Batch, Beam, Time, Dim] -> Time è la sequence dim
        h_transposed = h.transpose(0, 2, 1, 3) # [B, L, T, D]
        h_reshaped = h_transposed.reshape(B * L, T, self.embed_dim) # "Batch" per MHA è B*L
        
        # Self Attention temporale
        t_out = self.temporal_attn(query=h_reshaped, key=h_reshaped, value=h_reshaped)
        
        # Residual connection sul tempo + Norm
        h_reshaped = self.temporal_norm(h_reshaped + t_out)
        
        # Ritorniamo alla forma originale: [B, T, L, D]
        h_final = h_reshaped.reshape(B, L, T, self.embed_dim).transpose(0, 2, 1, 3)
        
        return h_final

class GMMQueryDecoder(hk.Module):
    def __init__(self, n_components, embed_dim, name=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.embed_dim = embed_dim
        
        # LEARNABLE QUERIES: Il cuore del sistema DETR-like.
        # Ogni query imparerà a rappresentare una parte della GMM (es. Query 0 = Ostacoli vicini, Query 1 = Ostacoli lontani...)
        self.query_embeddings = hk.get_parameter(
            "query_embeddings", 
            shape=[1, self.n_components, embed_dim], 
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        
        self.cross_attn = hk.MultiHeadAttention(num_heads=4, key_size=16, w_init_scale=1.0)
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.nets.MLP([embed_dim * 2, embed_dim], activation=nn.gelu, activate_final=False)
        self.norm_ffn = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, encoder_output):
        # encoder_output: [B, T, L, D]
        B, T, L, D = encoder_output.shape
        
        # 1. Flatten Spatio-Temporal dimensions for Cross Attention
        # Le query devono attendere su tutto lo spazio-tempo.
        # Key/Values: [B, T*L, D]
        kv = encoder_output.reshape(B, T * L, D)
        
        # 2. Prepare Queries
        # Espandiamo le query per la dimensione del batch: [B, n_components, D]
        q = jnp.tile(self.query_embeddings, (B, 1, 1))
        
        # 3. Cross Attention
        # Le Queries "leggono" l'input codificato per aggiornare il loro stato
        attn_out = self.cross_attn(query=q, key=kv, value=kv)
        q = self.norm(q + attn_out)
        
        # 4. FFN (Feed Forward per elaborare le informazioni)
        ffn_out = self.ffn(q)
        q = self.norm_ffn(q + ffn_out)
        
        return q # [B, n_components, D]

class Encoder(hk.Module):
    def __init__(
            self,
            mean_limits: jnp.ndarray, # Shape ((x_min, x_max), (y_min, y_max))
            n_gaussian_mixture_components: int,
            prediction_time: float,
            max_humans_velocity: float,
            embed_dim: int = 64
        ):
        super().__init__(name="lidar_gmm_encoder")
        self.mean_limits = mean_limits
        self.n_components = n_gaussian_mixture_components
        self.embed_dim = embed_dim
        self.prediction_time = prediction_time
        self.max_disp = max_humans_velocity * prediction_time

        # Modules
        self.input_proj = hk.Linear(embed_dim, name="input_projection")
        self.encoder = SpatioTemporalEncoder(embed_dim=embed_dim)
        
        # Decoders specifici per ogni task (Obs, Hum, NextHum)
        # Usiamo decoders separati perché le query per un muro sono diverse dalle query per un umano
        self.obs_decoder = GMMQueryDecoder(n_gaussian_mixture_components, embed_dim, name="obs_decoder")
        self.hum_decoder = GMMQueryDecoder(n_gaussian_mixture_components, embed_dim, name="hum_decoder")
        self.next_hum_decoder = GMMQueryDecoder(n_gaussian_mixture_components, embed_dim, name="next_hum_decoder")

        # Heads (MLP finali per trasformare i vettori latenti in parametri GMM)
        # Output sizes: 
        # Obs/Hum: 6 params (mu_x, mu_y, log_sig_x, log_sig_y, corr, logit_weight)
        # NextHum: 5 params (delta_mu_x, delta_mu_y, next_sig_x, next_sig_y, next_corr) -> weight is shared? Or separate? 
        # Assumiamo output full GMM per coerenza col tuo codice precedente
        self.head_obs = hk.Linear(6) 
        self.head_hum = hk.Linear(6)
        self.head_next = hk.Linear(5) # Delta Mu + params

    def _params_to_gmm(self, raw_params, is_delta=False, base_means=None):
        """Converte raw output MLP in parametri GMM validi."""
        # raw_params: [B, n_comp, 6] or [B, n_comp, 5]
        
        # 1. Means (Position)
        # Usiamo Sigmoid per vincolare strettamente dentro i limiti [0, 1] -> [min, max]
        # Se is_delta=True, calcoliamo lo spostamento
        if not is_delta:
            raw_x, raw_y = raw_params[..., 0], raw_params[..., 1]
            mu_x = nn.sigmoid(raw_x) * (self.mean_limits[0][1] - self.mean_limits[0][0]) + self.mean_limits[0][0]
            mu_y = nn.sigmoid(raw_y) * (self.mean_limits[1][1] - self.mean_limits[1][0]) + self.mean_limits[1][0]
            idx_offset = 2
        else:
            # Per Next Human, prediciamo spostamento relativo (velocity * time)
            # Limitiamo lo spostamento massimo con tanh
            d_x, d_y = raw_params[..., 0], raw_params[..., 1]
            dist = jnp.sqrt(d_x**2 + d_y**2 + 1e-6)
            scale = nn.tanh(dist) * self.max_disp
            delta_x = (d_x / dist) * scale
            delta_y = (d_y / dist) * scale
            
            mu_x = base_means[..., 0] + delta_x
            mu_y = base_means[..., 1] + delta_y
            idx_offset = 2

        # 2. Log Sigmas
        # Usiamo 5 * tanh per vincolare log_sigma in [-5, 5].
        # Sigma = exp(log_sigma) sarà in [0.0067, 148.4].
        # Questo è preferibile a softplus per evitare esplosioni o valori troppo piccoli.
        log_sig_x = 5.0 * nn.tanh(raw_params[..., idx_offset])
        log_sig_y = 5.0 * nn.tanh(raw_params[..., idx_offset+1])
        
        # 3. Correlation (Tanh -> [-0.99, 0.99])
        rho = nn.tanh(raw_params[..., idx_offset+2]) * 0.99
        
        # 4. Weights (Softmax)
        if not is_delta:
            # raw_params[..., 5] sono i logits
            w_logits = raw_params[..., 5]
            weights = nn.softmax(w_logits, axis=-1)
            
            return {
                "means": jnp.stack([mu_x, mu_y], axis=-1),
                "logsigmas": jnp.stack([log_sig_x, log_sig_y], axis=-1), # Export stddev instead of logsigma for easier usage
                "correlations": rho,
                "weights": weights
            }
        else:
            # Per next step, potremmo riutilizzare i pesi correnti o predirne di nuovi.
            # Qui ritorno solo i parametri geometrici, i pesi si prendono dallo step attuale solitamente
            return {
                "means": jnp.stack([mu_x, mu_y], axis=-1),
                "logsigmas": jnp.stack([log_sig_x, log_sig_y], axis=-1),
                "correlations": rho
            }

    def __call__(self, x):
        # x input: [n_stack, num_beams, 12] -> Assumiamo che l'utente gestisca il batch fuori o input sia single sample
        # Aggiungiamo dimensione batch fittizia se manca:
        has_batch = x.ndim == 4
        if not has_batch:
            x = jnp.expand_dims(x, 0) # [1, T, L, F]

        # 1. Feature Projection
        x_emb = self.input_proj(x) # [B, T, L, embed_dim]
        
        # 2. Spatio-Temporal Encoding
        encoded_features = self.encoder(x_emb) # [B, T, L, embed_dim]
        
        # 3. Decoding via Queries
        # Ogni decoder produce [B, n_components, embed_dim]
        obs_latents = self.obs_decoder(encoded_features)
        hum_latents = self.hum_decoder(encoded_features)
        next_hum_latents = self.next_hum_decoder(encoded_features)
        
        # 4. Heads & Parameter Transformation
        raw_obs = self.head_obs(obs_latents)
        raw_hum = self.head_hum(hum_latents)
        raw_next = self.head_next(next_hum_latents)
        
        obs_distr = self._params_to_gmm(raw_obs, is_delta=False)
        hum_distr = self._params_to_gmm(raw_hum, is_delta=False)
        
        # Per next humans, usiamo le medie correnti come base
        next_hum_distr = self._params_to_gmm(raw_next, is_delta=True, base_means=hum_distr["means"])
        # Assegniamo i pesi attuali anche al futuro (o aggiungi una head per delta weights)
        next_hum_distr["weights"] = hum_distr["weights"]

        # Rimuovi batch dimension se l'input non l'aveva
        if not has_batch:
            obs_distr = tree_map(lambda t: t[0], obs_distr)
            hum_distr = tree_map(lambda t: t[0], hum_distr)
            next_hum_distr = tree_map(lambda t: t[0], next_hum_distr)

        return {
            "obs_distr": obs_distr,
            "hum_distr": hum_distr,
            "next_hum_distr": next_hum_distr
        }

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
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.n_inputs = (3, self.n_components, 7+9) # (3, Number of GMM compoents, 7 params per GMM component + 9 robot goal, robot vmax, robot radius, robot wheels distance, action space params)
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
        obstacles_input = x[0,:,:6]  # Shape: (n_components, 6)
        obstacles_weights = x[0,:,6] # Shape: (n_components,)
        obstacles_features = self.features_mlp_obs(obstacles_input)  # Shape: (n_components, obs embedding_size)
        weighted_obstacles_features = jnp.sum(jnp.multiply(obstacles_weights[:,None], obstacles_features), axis=0) # Shape: (obs feature_size,)
        ## Compute human attentive embedding
        humans_input = jnp.concatenate([x[1,:,:6], x[2,:,:6]], axis=-1)  # Shape: (n_components, 12)
        humans_weights = x[1,:,6]  # Shape: (n_components, 1)
        # next_humans_weights = x[2,:,6]  # Shape: (n_components, 1) - Currently these are the same as humans_weights
        humans_embeddings = self.embedding_mlp_hum(humans_input)  # Shape: (n_components, hum embedding_size)
        humans_keys = self.hum_key_mlp(humans_embeddings)  # Shape: (n_components, key_size)
        humans_queries = self.hum_query_mlp(humans_embeddings)  # Shape: (n_components, query_size)
        humans_values = self.hum_value_mlp(humans_embeddings)  # Shape: (n_components, value_size)
        humans_attention_matrix = jnp.dot(humans_queries, humans_keys.T) / jnp.sqrt(humans_keys.shape[-1])  # Shape: (n_components, n_components)
        humans_attention_matrix = nn.softmax(humans_attention_matrix, axis=-1)  # Shape: (n_components, n_components)
        humans_features = jnp.dot(humans_attention_matrix, humans_values)  # Shape: (n_components, value_size)
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
        self.n_inputs = 3 * 6 * self.n_components + 5  # 6 outputs per GMM cell (mean_x, mean_y, sigma_x, sigma_y, correlation, weight) times  3 GMMs (obstacles, current humans, next humans)
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
        n_gmm_components:int=10,
        prediction_horizon:int=4,
        max_humans_velocity:float=1.5,
        gmm_means_limits:jnp.ndarray=jnp.array([[-10,10], [-10,10]]),
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
        assert n_gmm_components >= 2, "Number of GMM components must be at least 2"
        assert prediction_horizon >= 1, "Prediction horizon must be at least 1"
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
        self.n_gmm_components = n_gmm_components
        self.gmm_means_limits = gmm_means_limits
        self.prediction_horizon = prediction_horizon
        self.max_humans_velocity = max_humans_velocity
        # Default attributes
        self.name = "JESSI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.dirichlet = Dirichlet()
        self.gmm = BivariateGMM(self.n_gmm_components)
        # Initialize Encoder network
        @hk.transform
        def encoder_network(x):
            net = Encoder(self.gmm_means_limits, self.n_gmm_components, self.dt * self.prediction_horizon, self.max_humans_velocity)
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
    def _process_lidar_stack(self, obs_stack, ref_position, ref_orientation):
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
        encoder_params = self.encoder.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 12)))
        actor_params = self.actor.init(key, jnp.zeros((3, self.n_gmm_components, 7 + 9)))  # 7 params per GMM component + 9 robot goal, robot vmax, robot radius, robot wheels distance, action space params
        # critic_params = 
        return encoder_params, actor_params, {}

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
    def process_lidar(
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
        return vmap(JESSI._process_lidar_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)

    @partial(jit, static_argnames=("self"))
    def compute_actor_input(
        self,
        obs_gmm,
        hum_gmm,
        next_hum_gmm,
        action_space_params,
        robot_goal, # In cartesian coordinates (gx, gy) IN THE ROBOT FRAME
    ):
        # Compute ROBOT state inputs
        robot_goal_dist = jnp.linalg.norm(robot_goal)
        robot_goal_theta = jnp.arctan2(robot_goal[1], robot_goal[0])
        robot_goal_sin_theta = jnp.sin(robot_goal_theta)
        robot_goal_cos_theta = jnp.cos(robot_goal_theta)
        tiled_action_space_params = jnp.tile(action_space_params, (self.n_gmm_components,1)) # Shape: (n_components, 3)
        tiled_robot_params = jnp.tile(jnp.array([self.v_max, self.robot_radius, self.wheels_distance]), (self.n_gmm_components,1)) # Shape: (n_components, 3)
        tiled_robot_goals = jnp.tile(jnp.array([robot_goal_dist, robot_goal_sin_theta, robot_goal_cos_theta]), (self.n_gmm_components,1)) # Shape: (n_components, 3)
        # Compute OBSTACLES GMMs input
        obs_means_dist = jnp.linalg.norm(obs_gmm["means"], axis=-1)
        obs_means_theta = jnp.arctan2(obs_gmm["means"][:,1], obs_gmm["means"][:,0])
        obs_means_sin_theta = jnp.sin(obs_means_theta)
        obs_means_cos_theta = jnp.cos(obs_means_theta)
        hum_means_dist = jnp.linalg.norm(hum_gmm["means"], axis=-1)
        hum_means_theta = jnp.arctan2(hum_gmm["means"][:,1], hum_gmm["means"][:,0])
        hum_means_sin_theta = jnp.sin(hum_means_theta)
        hum_means_cos_theta = jnp.cos(hum_means_theta)
        next_hum_means_dist = jnp.linalg.norm(next_hum_gmm["means"], axis=-1)
        next_hum_means_theta = jnp.arctan2(next_hum_gmm["means"][:,1], next_hum_gmm["means"][:,0])
        next_hum_means_sin_theta = jnp.sin(next_hum_means_theta)
        next_hum_means_cos_theta = jnp.cos(next_hum_means_theta)
        obs_gmm = jnp.column_stack((
            obs_means_dist,
            obs_means_sin_theta,
            obs_means_cos_theta,
            obs_gmm["logsigmas"][:, 0],
            obs_gmm["logsigmas"][:, 1],
            obs_gmm["correlations"],
            obs_gmm["weights"],
            tiled_action_space_params[:, 0],
            tiled_action_space_params[:, 1],
            tiled_action_space_params[:, 2],
            tiled_robot_params[:, 0],
            tiled_robot_params[:, 1],
            tiled_robot_params[:, 2],
            tiled_robot_goals[:, 0],
            tiled_robot_goals[:, 1],
            tiled_robot_goals[:, 2],
        ))  # Shape: (n_components, 7 + 8)
        # Compute HUMANS GMMs input
        hum_gmm = jnp.column_stack((
            hum_means_dist,
            hum_means_sin_theta,
            hum_means_cos_theta,
            hum_gmm["logsigmas"][:, 0],
            hum_gmm["logsigmas"][:, 1],
            hum_gmm["correlations"],
            hum_gmm["weights"],
            tiled_action_space_params[:, 0],
            tiled_action_space_params[:, 1],
            tiled_action_space_params[:, 2],
            tiled_robot_params[:, 0],
            tiled_robot_params[:, 1],
            tiled_robot_params[:, 2],
            tiled_robot_goals[:, 0],
            tiled_robot_goals[:, 1],
            tiled_robot_goals[:, 2],
        ))  # Shape: (n_components, 7 + 8)
        # Compute NEXT HUMANS GMMs input
        next_hum_gmm = jnp.column_stack(( 
            next_hum_means_dist,
            next_hum_means_sin_theta,
            next_hum_means_cos_theta,
            next_hum_gmm["logsigmas"][:, 0],
            next_hum_gmm["logsigmas"][:, 1],
            next_hum_gmm["correlations"],
            next_hum_gmm["weights"],
            tiled_action_space_params[:, 0],
            tiled_action_space_params[:, 1],
            tiled_action_space_params[:, 2],
            tiled_robot_params[:, 0],
            tiled_robot_params[:, 1],
            tiled_robot_params[:, 2],
            tiled_robot_goals[:, 0],
            tiled_robot_goals[:, 1],
            tiled_robot_goals[:, 2],
        )) # Shape: (n_components, 7 + 8)
        # Concatenate all inputs
        actor_input = jnp.array([obs_gmm, hum_gmm, next_hum_gmm])  # Shape: (3, n_components, 7 + 8)
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
        - lidar_tokens (n_stack, lidar_num_rays, 12): aligned LiDAR tokens for transformer encoder.
        12 features per token: [hit, inside_box, distance, sin(theta), cos(theta), x, y, norm_stack_index, delta_x, delta_y, delta_theta, norm_delta_t]
        - last LiDAR point cloud (lidar_num_rays, 2): in robot frame of the most recent observation.
        """
        # Align LiDAR scans - (x,y) coordinates of pointcloud in the robot frame, first information corresponds to the most recent observation.
        aligned_lidar_scans = self.process_lidar(obs)[0]  # Shape: (n_stack, lidar_num_rays, 2)
        last_lidar_point_cloud = aligned_lidar_scans[0,:, :]  # Shape: (lidar_num_rays, 2)
        ego_delta_xs = obs[:,0] - obs[0,0]
        ego_delta_ys = obs[:,1] - obs[0,1]
        ego_delta_thetas = obs[:,2] - obs[0,2]
        # Compute LiDAR tokens
        @jit
        def compute_beam_token(
            scan_index:int,
            point:jnp.ndarray,
            ego_delta_x:float,
            ego_delta_y:float,
            ego_delta_theta:float,
        ) -> jnp.ndarray:
            # Extract point coordinates
            x, y = point
            # Check if point is inside the bounding box
            inside_box = jnp.where(
                (x >= self.gmm_means_limits[0,0]) & (x <= self.gmm_means_limits[0,1]) &
                (y >= self.gmm_means_limits[1,0]) & (y <= self.gmm_means_limits[1,1]),
                1.0,
                0.0
            )
            # Compute beam features
            distance = jnp.linalg.norm(point)
            angle = jnp.arctan2(y, x)
            sin_theta = jnp.sin(angle)
            cos_theta = jnp.cos(angle)
            hit = jnp.where(distance < self.lidar_max_dist, 1.0, 0.0)
            # Compute stack index features
            norm_stack_index = scan_index / (self.n_stack - 1)
            norm_delta_t = scan_index * self.dt / ((self.n_stack - 1) * self.dt)
            return jnp.array([
                hit,
                inside_box,
                distance,
                sin_theta,
                cos_theta,
                x,
                y,
                norm_stack_index,
                ego_delta_x,
                ego_delta_y,
                ego_delta_theta,
                norm_delta_t,
            ])
        encoder_input = vmap(vmap(compute_beam_token, in_axes=(None, 0, None, None, None)), in_axes=(0, 0, 0, 0, 0))(
            jnp.arange(self.n_stack),
            aligned_lidar_scans,
            ego_delta_xs,
            ego_delta_ys,
            ego_delta_thetas,
        )  # Shape: (n_stack, lidar_num_rays, 12)
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
        encoder_distrs = self.encoder.apply(
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
    def batch_loss_function_encoder(
        self,
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
            encoder_distrs = self.encoder.apply(current_params, None, input)
            obs_prediction = encoder_distrs["obs_distr"]
            humans_prediction = encoder_distrs["hum_distr"]
            next_humans_prediction = encoder_distrs["next_hum_distr"]
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
        # Experiences: {"inputs":jnp.ndarray, "obstacles_samples":jnp.ndarray, "humans_samples":jnp.ndarray, "next_humans_samples":jnp.ndarray}
    ) -> tuple:
        @jit
        def _compute_loss_and_gradients(
            current_params:dict,  
            experiences:dict[str:jnp.ndarray],
            # Experiences: {"inputs":jnp.ndarray, "obstacles_samples":jnp.ndarray, "humans_samples":jnp.ndarray, "next_humans_samples":jnp.ndarray}
        ) -> tuple:

            inputs = experiences["inputs"]
            obstacles_samples = experiences["obstacles_samples"]
            humans_samples = experiences["humans_samples"]
            next_humans_samples = experiences["next_humans_samples"]
            # Compute the loss and gradients
            loss, grads = value_and_grad(self.batch_loss_function_encoder)(
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

    @partial(jit, static_argnames=("self", "n_samples", "n_humans"))
    def build_frame_humans_samples(
        self,
        rc_humans_positions:jnp.ndarray,
        humans_radii:jnp.ndarray,
        humans_visibility:jnp.ndarray,
        key:random.PRNGKey,
        n_samples:int,
        n_humans:int,
        negative_samples_threshold:float=0.2,
    ) -> jnp.ndarray:
        """
        Sample points inside the local box around the robot, labeling them as positive/negative depending on whether they fall inside/outside any human.
        1. Sample points uniformly inside each visible human.
        2. Fill the remaining samples with random negative samples (outside all humans).
        3. Return samples and their labels.

        args:
        - rc_humans_positions (n_humans, 2): Humans positions IN THE ROBOT FRAME.
        - humans_radii (n_humans,): Humans radii.
        - humans_visibility (n_humans,): Boolean mask indicating which humans are visible.
        - key: JAX random key.
        - n_samples: Total number of samples to generate.
        - n_humans: Number of humans in the scene.
        - negative_samples_threshold: Minimum distance from human boundary for negative samples.

        output:
        - humans_samples: dict with:
            - position (n_samples, 2): Sampled points positions.
            - is_positive (n_samples,): Boolean mask indicating whether each sample is positive (inside any human) or negative (outside all humans).
        """
        ### Mask invisible humans and obstacles with NaNs
        humans = jnp.where(
            humans_visibility[:, None],
            jnp.concatenate((rc_humans_positions, humans_radii[:, None]), axis=-1),
            jnp.array([jnp.nan, jnp.nan, jnp.nan])
        )  # Shape: (n_humans, 3)
        ### Split n_samples evenly with respect to humans
        samples_per_object = n_samples // n_humans
        # Humans samples
        @partial(jit, static_argnames=['n_hum_samples'])
        def sample_from_human(human: jnp.ndarray, key: random.PRNGKey, n_hum_samples: int) -> jnp.ndarray:
            @jit
            def _not_nan_human(human):
                position, radius = human[0:2], human[2]
                angle_key, radius_key = random.split(key)
                angles = random.uniform(angle_key, (n_hum_samples,), minval=0.0, maxval=2*jnp.pi)
                rs = radius * jnp.sqrt(random.uniform(radius_key, (n_hum_samples,)))
                xs = position[0] + rs * jnp.cos(angles)
                ys = position[1] + rs * jnp.sin(angles)
                return jnp.stack((xs, ys), axis=-1)  # Shape: (n_hum_samples, 2)

            return lax.cond(
                jnp.any(jnp.isnan(human)),
                lambda _: jnp.full((n_hum_samples, 2), jnp.nan),
                _not_nan_human,
                human
            )
        humans_keys = random.split(key, n_humans)
        humans_samples = vmap(sample_from_human, in_axes=(0, 0, None))(humans, humans_keys, samples_per_object)  # Shape: (n_humans, samples_per_object, 2)
        humans_samples = humans_samples.reshape((n_humans * samples_per_object, 2))
        # Randomly fill nan samples with negative samples
        nan_mask = jnp.isnan(humans_samples).any(axis=1)
        total_nans = jnp.sum(nan_mask)
        aux_samples = jnp.nan_to_num(humans_samples, nan=jnp.inf)  # Temporary replace NaNs with large negative number for sorting
        idxs = jnp.argsort(aux_samples, axis=0)
        positive = vmap(lambda x: x < n_samples - total_nans)(jnp.arange(n_samples))
        humans_samples = { 
            "position": humans_samples[idxs[:,0]],  # Sort samples so that NaNs are at the end
            "is_positive": positive,
        }
        keys = random.split(key, n_samples)
        @jit
        def fill_nan_samples_with_negatives(sample: jnp.ndarray, key: random.PRNGKey, humans: jnp.ndarray) -> jnp.ndarray:
            @jit
            def find_negative_sample(val:tuple) -> jnp.ndarray:
                _, key, humans = val
                def _while_body(state):
                    key, _, is_positive, humans = state
                    key, subkey = random.split(key)
                    x_key, y_key = random.split(subkey)
                    x = random.uniform(x_key, minval=self.gmm_means_limits[0,0], maxval=self.gmm_means_limits[0,1])
                    y = random.uniform(y_key, minval=self.gmm_means_limits[1,0], maxval=self.gmm_means_limits[1,1])
                    pos = jnp.array([x, y])
                    is_positive = jnp.any(jnp.linalg.norm(pos - humans[:,:2], axis=1) < humans[:,2] + negative_samples_threshold)
                    return key, pos, is_positive, humans
                _, sample_position, _, _ = lax.while_loop(
                    lambda state: state[2],
                    _while_body,
                    (key, jnp.array([0.,0.]), True, humans)
                )
                return {"position": sample_position, "is_positive": False}
            return lax.cond(
                sample["is_positive"],
                lambda x: {"position": x[0]["position"], "is_positive": x[0]["is_positive"]},
                find_negative_sample,
                (sample, key, humans)
            )
        return vmap(fill_nan_samples_with_negatives, in_axes=(0, 0, None))(
            humans_samples,
            keys,
            humans,
        )

    @partial(jit, static_argnames=("self", "n_samples", "n_humans"))
    def batch_build_frame_humans_samples(
        self,
        batch_rc_humans_positions:jnp.ndarray,
        batch_humans_radii:jnp.ndarray,
        batch_humans_visibility:jnp.ndarray,
        keys:random.PRNGKey,
        n_samples:int,
        n_humans:int,
        negative_samples_threshold:float=0.2,
    ) -> jnp.ndarray:
        """
        Batch version of build_frame_humans_samples. Refers to the documentation of that method for details.
        """
        return vmap(JESSI.build_frame_humans_samples, in_axes=(None, 0, 0, 0, 0, None, None, None))(
            self,
            batch_rc_humans_positions,
            batch_humans_radii,
            batch_humans_visibility,
            keys,
            n_samples,
            n_humans,
            negative_samples_threshold,
        )

    @partial(jit, static_argnames=("self", "n_samples", "n_obstacles"))
    def build_frame_obstacles_samples(
        self,
        rc_static_obstacles:jnp.ndarray,
        obstacles_visibility:jnp.ndarray,
        key:random.PRNGKey,
        n_samples:int,
        n_obstacles:int,
        negative_samples_threshold:float,
    ) -> jnp.ndarray:
        """
        Sample points inside the local box around the robot, labeling them as positive/negative depending on whether they fall inside/outside any obstacle.
        1. Sample points uniformly inside each visible obstacle.
        2. Fill the remaining samples with random negative samples (outside all obstacles).
        3. Return samples and their labels.

        args:
        - rc_obstacles_positions (n_obstacles, 2): obstacles positions IN THE ROBOT FRAME.
        - obstacles_visibility (n_obstacles,): Boolean mask indicating which obstacles are visible.
        - key: JAX random key.
        - n_samples: Total number of samples to generate.
        - n_obstacles: Number of obstacles in the scene.
        - negative_samples_threshold: Minimum distance from obstacle boundary for negative samples.

        output:
        - obstacles_samples: dict with:
            - position (n_samples, 2): Sampled points positions.
            - is_positive (n_samples,): Boolean mask indicating whether each sample is positive (inside any obstacle) or negative (outside all obstacles).
        """
        ### WARNING: CURRENT IMPLEMENTATION CONSIDERS THE INSIDE OF POLIGONAL OBSTACLES AS FREE SPACE
        obstacles = jnp.where(
            obstacles_visibility[:,:, None, None],
            rc_static_obstacles,
            jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])
        )  # Shape: (n_obstacles, 1, 2, 2)
        ### Split n_samples evenly with respect to n_obstacles
        samples_per_object = n_samples // n_obstacles
        # Obstacles samples
        @partial(jit, static_argnames=['n_obs_samples'])
        def sample_from_obstacle(obstacle: jnp.ndarray, key: random.PRNGKey, n_obs_samples: int) -> jnp.ndarray:
            n_seg_samples = n_obs_samples // obstacle.shape[0]
            @jit
            def _segment_samples(segment):
                @jit
                def _not_nan_segment(segment):
                    p1, p2 = segment[0], segment[1]
                    # Sample uniformly on the segment
                    ts = random.uniform(key, (n_seg_samples,), minval=0.0, maxval=1.0)
                    xs = p1[0] + ts * (p2[0] - p1[0])
                    ys = p1[1] + ts * (p2[1] - p1[1])
                    return jnp.stack((xs, ys), axis=-1)  # Shape: (n_samples, 2)
                return lax.cond(
                    jnp.any(jnp.isnan(segment)),
                    lambda _: jnp.full((n_seg_samples, 2), jnp.nan),
                    _not_nan_segment,
                    segment
                )
            return jnp.concatenate(vmap(_segment_samples, in_axes=(0,))(obstacle), axis=0)  # Shape: (n_obs_samples, 2)
        obstacles_keys = random.split(key, n_obstacles)
        obstacles_samples = vmap(sample_from_obstacle, in_axes=(0, 0, None))(obstacles, obstacles_keys, samples_per_object)  # Shape: (n_obstacles, samples_per_object, 2)
        obstacles_samples = obstacles_samples.reshape((n_obstacles * samples_per_object, 2))
        # Randomly fill nan samples with negative samples
        nan_mask = jnp.isnan(obstacles_samples).any(axis=1)
        total_nans = jnp.sum(nan_mask)
        aux_samples = jnp.nan_to_num(obstacles_samples, nan=jnp.inf)  # Temporary replace NaNs with large negative number for sorting
        idxs = jnp.argsort(aux_samples, axis=0)
        positive = vmap(lambda x: x < n_samples - total_nans)(jnp.arange(n_samples))
        obstacles_samples = { 
            "position": obstacles_samples[idxs[:,0]],  # Sort samples so that NaNs are at the end
            "is_positive": positive,
        }
        keys = random.split(key, n_samples)
        @jit
        def fill_nan_samples_with_negatives(sample: jnp.ndarray, key: random.PRNGKey, obstacles: jnp.ndarray) -> jnp.ndarray:
            @jit
            def find_negative_sample(val:tuple) -> jnp.ndarray:
                _, key, obstacles = val
                def _while_body(state):
                    key, _, is_positive, obstacles = state
                    key, subkey = random.split(key)
                    x_key, y_key = random.split(subkey)
                    x = random.uniform(x_key, minval=self.gmm_means_limits[0,0], maxval=self.gmm_means_limits[0,1])
                    y = random.uniform(y_key, minval=self.gmm_means_limits[1,0], maxval=self.gmm_means_limits[1,1])
                    pos = jnp.array([x, y])
                    closest_points = vectorized_compute_obstacle_closest_point(pos, obstacles)
                    is_positive = jnp.any(jnp.linalg.norm(pos - closest_points, axis=1) < negative_samples_threshold)
                    return key, pos, is_positive, obstacles
                _, sample_position, _, _ = lax.while_loop(
                    lambda state: state[2],
                    _while_body,
                    (key, jnp.array([0.,0.]), True, obstacles)
                )
                return {"position": sample_position, "is_positive": False}
            return lax.cond(
                sample["is_positive"],
                lambda x: {"position": x[0]["position"], "is_positive": x[0]["is_positive"]},
                find_negative_sample,
                (sample, key, obstacles)
            )
        return vmap(fill_nan_samples_with_negatives, in_axes=(0, 0, None))(
            obstacles_samples,
            keys,
            obstacles,
        )

    @partial(jit, static_argnames=("self", "n_samples", "n_obstacles"))
    def batch_build_frame_obstacles_samples(
        self,
        batch_rc_static_obstacles:jnp.ndarray,
        batch_obstacles_visibility:jnp.ndarray,
        keys:random.PRNGKey,
        n_samples:int,
        n_obstacles:int,
        negative_samples_threshold:float,
    ) -> jnp.ndarray:
        """
        Batch version of build_frame_obstacles_samples. Refers to the documentation of that method for details.
        """
        return vmap(JESSI.build_frame_obstacles_samples, in_axes=(None, 0, 0, 0, None, None, None))(
            self,
            batch_rc_static_obstacles,
            batch_obstacles_visibility,
            keys,
            n_samples,
            n_obstacles,
            negative_samples_threshold,
        )

    def animate_trajectory(
        self,
        robot_poses, # x, y, theta
        robot_actions,
        robot_goals,
        observations,
        actor_distrs,
        encoder_distrs,
        humans_poses, # x, y, theta
        humans_velocities, # vx, vy (in global frame)
        humans_radii,
        static_obstacles,
        p_visualization_threshold_gmm:float=0.05,
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
            len(encoder_distrs['obs_distr']['means']) == \
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
        box_points = jnp.array([
            [self.gmm_means_limits[0,0], self.gmm_means_limits[1,0]],
            [self.gmm_means_limits[0,1], self.gmm_means_limits[1,0]],
            [self.gmm_means_limits[0,1], self.gmm_means_limits[1,1]],
            [self.gmm_means_limits[0,0], self.gmm_means_limits[1,1]],
        ])
        sx = jnp.linspace(self.gmm_means_limits[0, 0], self.gmm_means_limits[0, 1], num=60, endpoint=True)
        sy = jnp.linspace(self.gmm_means_limits[1, 0], self.gmm_means_limits[1, 1], num=60, endpoint=True)
        test_samples_x, test_samples_y = jnp.meshgrid(sx, sy)
        test_samples = jnp.stack((test_samples_x.flatten(), test_samples_y.flatten()), axis=-1)
        from socialjym.policies.cadrl import CADRL
        from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
        dummy_cadrl = CADRL(DummyReward(kinematics="unicycle"),kinematics="unicycle",v_max=self.v_max,wheels_distance=self.wheels_distance)
        test_action_samples = dummy_cadrl._build_action_space(unicycle_triangle_samples=35)
        # Animate trajectory
        fig, axs = plt.subplots(2,3,figsize=(24,8))
        fig.subplots_adjust(left=0.02, right=0.99, wspace=0.08, hspace=0.2, top=0.95, bottom=0.07)
        def animate(frame):
            for i, row in enumerate(axs):
                for j, ax in enumerate(row):
                    ax.clear()
                    if i == 1 and j == 2: continue # This is the ax for the action space
                    ax.set(xlim=x_lims if x_lims is not None else [-10,10], ylim=y_lims if y_lims is not None else [-10,10])
                    ax.set_xlabel('X', labelpad=-5)
                    ax.set_ylabel('Y', labelpad=-13)
                    ax.set_aspect('equal', adjustable='box')
                    # Plot box limits
                    c, s = jnp.cos(robot_poses[frame,2]), jnp.sin(robot_poses[frame,2])
                    rot = jnp.array([[c, -s], [s, c]])
                    rotated_box_points = jnp.einsum('ij,jk->ik', rot, box_points.T).T + robot_poses[frame,:2]
                    to_plot = jnp.vstack((rotated_box_points, rotated_box_points[0:1,:]))
                    ax.plot(to_plot[:,0], to_plot[:,1], color='grey', linewidth=2, alpha=0.5, zorder=1)
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
            ### FIRST ROW AXS: PERCEPTION
            obs_distr = tree_map(lambda x: x[frame], encoder_distrs["obs_distr"])
            hum_distr = tree_map(lambda x: x[frame], encoder_distrs["hum_distr"])
            next_hum_distr = tree_map(lambda x: x[frame], encoder_distrs["next_hum_distr"])
            # AX 0,0: Obstacles GMM
            test_p = self.gmm.batch_p(obs_distr, test_samples)
            points_high_p = test_samples[test_p > p_visualization_threshold_gmm]
            corresponding_colors = test_p[test_p > p_visualization_threshold_gmm]
            rotated_means = jnp.einsum('ij,jk->ik', rot, obs_distr["means"].T).T + robot_poses[frame,:2]
            rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + robot_poses[frame,:2]
            axs[0,0].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
            axs[0,0].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[0,0].set_title("Obstacles GMM")
            # AX 0,1: Humans GMM
            test_p = self.gmm.batch_p(hum_distr, test_samples)
            points_high_p = test_samples[test_p > p_visualization_threshold_gmm]
            corresponding_colors = test_p[test_p > p_visualization_threshold_gmm]
            rotated_means_hum = jnp.einsum('ij,jk->ik', rot, hum_distr["means"].T).T + robot_poses[frame,:2]
            rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + robot_poses[frame,:2]
            axs[0,1].scatter(rotated_means_hum[:,0], rotated_means_hum[:,1], c='red', s=10, marker='x', zorder=100)
            axs[0,1].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[0,1].set_title("Humans GMM")
            # AX 0,2: Next Humans GMM
            test_p = self.gmm.batch_p(next_hum_distr, test_samples)
            points_high_p = test_samples[test_p > p_visualization_threshold_gmm]
            corresponding_colors = test_p[test_p > p_visualization_threshold_gmm]
            rotated_means_next_hum = jnp.einsum('ij,jk->ik', rot, next_hum_distr["means"].T).T + robot_poses[frame,:2]
            rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + robot_poses[frame,:2]
            axs[0,2].scatter(rotated_means_next_hum[:,0], rotated_means_next_hum[:,1], c='red', s=10, marker='x', zorder=100)
            axs[0,2].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[0,2].set_title("Next Humans GMM")
            # Plot means displacement arrows between humans and next humans GMMs
            for c in range(len(hum_distr["means"])):
                axs[0,1].arrow(
                    rotated_means_hum[c,0],
                    rotated_means_hum[c,1],
                    rotated_means_next_hum[c,0] - rotated_means_hum[c,0],
                    rotated_means_next_hum[c,1] - rotated_means_hum[c,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc="red",
                    ec="red",
                    alpha=0.8,
                    zorder=50,
                )
            ### SECOND ROW AXS: SIMULATION, POINT CLOUD AND ACTIONS
            # AX 1,0: Simulation with LiDAR ranges
            lidar_scan = observations[frame,0,6:]
            for ray in range(len(lidar_scan)):
                axs[1,0].plot(
                    [robot_poses[frame,0], robot_poses[frame,0] + lidar_scan[ray] * jnp.cos(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    [robot_poses[frame,1], robot_poses[frame,1] + lidar_scan[ray] * jnp.sin(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    color="black", 
                    linewidth=0.5, 
                    zorder=0
                )
            axs[1,0].set_title("Trajectory")
            # AX 1,1: Simulation with LiDAR point cloud stack
            point_cloud = self.process_lidar(observations[frame])[1]
            for i, cloud in enumerate(point_cloud):
                # color/alpha fade with i (smaller i -> less faded)
                t = (1 - i / (self.n_stack - 1))  # in [0,1]
                axs[1,1].scatter(
                    cloud[:,0],
                    cloud[:,1],
                    c=0.3 + 0.7 * jnp.ones((self.lidar_num_rays,)) * t,
                    cmap='Reds',
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.3 + 0.7 * t,
                    zorder=20 + self.n_stack - i,
                )
            axs[1,1].set_title("Pointcloud")
            # AX 1,2: Feasible and bounded action space + action space distribution and action taken
            axs[1,2].set_xlabel("$v$ (m/s)")
            axs[1,2].set_ylabel("$\omega$ (rad/s)")
            axs[1,2].set_xlim(-0.1, self.v_max + 0.1)
            axs[1,2].set_ylim(-2*self.v_max/self.wheels_distance - 0.3, 2*self.v_max/self.wheels_distance + 0.3)
            axs[1,2].set_xticks(jnp.arange(0, self.v_max+0.2, 0.2))
            axs[1,2].set_xticklabels([round(i,1) for i in jnp.arange(0, self.v_max, 0.2)] + [r"$\overline{v}$"])
            axs[1,2].set_yticks(jnp.arange(-2,3,1).tolist() + [2*self.v_max/self.wheels_distance,-2*self.v_max/self.wheels_distance])
            axs[1,2].set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
            axs[1,2].grid()
            axs[1,2].add_patch(
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
            axs[1,2].add_patch(
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
            axs[1,2].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[1,2].plot(robot_actions[frame,0], robot_actions[frame,1], marker='^',markersize=7,color='red',zorder=51)
            axs[1,2].set_title("Action space")
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
        encoder_distrs,
        goals,
        static_obstacles,
        humans_radii,
        p_visualization_threshold_gmm:float=0.05,
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
            encoder_distrs,
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