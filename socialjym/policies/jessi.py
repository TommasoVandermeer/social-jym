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
from socialjym.utils.distributions.gaussian import BivariateGaussian
from socialjym.policies.base_policy import BasePolicy
from jhsfm.hsfm import get_linear_velocity
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics

class SinusoidalPositionalEncoding(hk.Module):
    def __init__(self, d_model, max_len=5000, name=None):
        super().__init__(name=name)
        position = jnp.arange(max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe_table = pe 

    def __call__(self, x):
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.shape[1]
        return x + self.pe_table[None, :seq_len, :]

class AngularLocalCrossAttention(hk.Module):
    def __init__(self, embed_dim, target_beams, max_angle_deg=18., name="angular_local_cross_attn", beam_dropout_rate=0.0):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.target_beams = target_beams 
        self.threshold = jnp.cos(jnp.deg2rad(max_angle_deg/2)) 
        self.beam_dropout_rate = beam_dropout_rate
    
        self.spatial_latents = hk.get_parameter("spatial_latents", [target_beams, embed_dim], init=hk.initializers.TruncatedNormal(stddev=0.02))
        query_angles = jnp.linspace(0, 2 * jnp.pi, target_beams, endpoint=False)

        self.latent_vecs = jnp.stack([jnp.sin(query_angles), jnp.cos(query_angles)], axis=-1)

        self.attn = hk.MultiHeadAttention(num_heads=4, key_size=embed_dim//4, w_init_scale=1.0, model_size=embed_dim)
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.ffn = hk.nets.MLP([embed_dim], activation=nn.gelu, activate_final=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def compute_angular_mask(self, input_sin_cos, key=None):
        # input_sin_cos: [B*T, Beams, 2]
        # latent_sin_cos: [Target_Beams, 2]
        cos_diff = jnp.einsum('id,mjd->mij', self.latent_vecs, input_sin_cos) # (B*T, Target_Beams, Beams)
        mask = cos_diff >= self.threshold # (B*T, Target_Beams, Beams)
        # debug.print("Mask example {x}", x=jnp.sum(mask[0,0,:]))
        
        if key is not None and self.beam_dropout_rate > 0.0:
            B_T, n_beams, _ = input_sin_cos.shape
            keep_prob = 1.0 - self.beam_dropout_rate
            random_mask = random.bernoulli(key, p=keep_prob, shape=(B_T, self.target_beams, n_beams))
            mask = mask & random_mask
        
        return mask[:, None, :, :] # [B*T, 1, Target_Beams, Beams]

    def __call__(self, x_emb, x_raw, key=None, external_mask=None):
        # x_emb: [B*T, Beams, D], x_raw: [B*T, Beams, 7]
        B_T = x_emb.shape[0]
        # 1. Latent Queries
        q = jnp.broadcast_to(self.spatial_latents[None, ...], (B_T, self.target_beams, self.embed_dim))
        # 2. Cross Attention
        if external_mask is not None:
            mask = external_mask.reshape(B_T, 1, self.target_beams, -1)
        else:
            input_sin_cos = x_raw[..., 4:6]
            mask = self.compute_angular_mask(input_sin_cos, key=key)
        attn_out = self.attn(query=q, key=x_emb, value=x_emb, mask=mask)
        q = self.norm1(q + attn_out)
        # 3. FFN
        ffn_out = self.ffn(q)
        return self.norm2(q + ffn_out), mask

class SpatioTemporalEncoder(hk.Module):
    def __init__(self, embed_dim, n_sectors, lidar_angles_robot_frame, name=None, beam_dropout_rate=0.0): 
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.n_sectors = n_sectors
        self.lidar_angles_robot_frame = lidar_angles_robot_frame # UNUSED
        # 1. Spatial Feature Extraction
        self.angular_spatial_attn = AngularLocalCrossAttention(embed_dim, target_beams=n_sectors, max_angle_deg=18., beam_dropout_rate=beam_dropout_rate)
        # 2. Temporal Positional Encodings
        self.pos_encoder_time = SinusoidalPositionalEncoding(embed_dim, max_len=100)
        # 3. Temporal Attention
        self.temporal_attn = hk.MultiHeadAttention(num_heads=4, key_size=embed_dim//4, w_init_scale=1.0)
        self.temporal_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.temporal_ffn = hk.nets.MLP([embed_dim * 2, embed_dim], activation=nn.gelu)
        self.temporal_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x, x_raw, key=None, external_mask=None):
        # x shape: [Batch, Time, Beams, Features]
        B, T, L, F = x.shape
        x_flat = x.reshape(B * T, L, self.embed_dim)
        x_raw_flat = x_raw.reshape(B * T, L, x_raw.shape[-1])
        h_spatial, mask = self.angular_spatial_attn(x_flat, x_raw_flat, key=key, external_mask=external_mask) # [B*T, L, embed_dim]
        L_new = h_spatial.shape[1]
        h_spatial = h_spatial.reshape(B, T, L_new, self.embed_dim)
        h_time_in = h_spatial.transpose(0, 2, 1, 3).reshape(B * L_new, T, self.embed_dim)
        h_time_in = self.pos_encoder_time(h_time_in)
        t_out = self.temporal_attn(query=h_time_in, key=h_time_in, value=h_time_in)
        h_time_mid = self.temporal_norm1(h_time_in + t_out)
        t_ffn_out = self.temporal_ffn(h_time_mid)
        h_time_out = self.temporal_norm2(h_time_mid + t_ffn_out)
        h_final = h_time_out.reshape(B, L_new, T, self.embed_dim).transpose(0, 2, 1, 3)
        return h_final, mask

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
        self.norm_cross = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.self_attn = hk.MultiHeadAttention(num_heads=4, key_size=embed_dim//4, w_init_scale=1.0, model_size=embed_dim)
        self.norm_self = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.nets.MLP([embed_dim * 2, embed_dim], activation=nn.gelu, activate_final=False)
        self.norm_ffn = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, encoder_output):
        # encoder_output: [B, T, L, D]
        B, T, L, D = encoder_output.shape
        kv = encoder_output.reshape(B, T * L, D)
        q = jnp.tile(self.query_embeddings, (B, 1, 1))
        # Cross Attention
        attn_out = self.cross_attn(query=q, key=kv, value=kv)
        q = self.norm_cross(q + attn_out)
        # Self Attention (avoiding interactions between queries for same human)
        self_out = self.self_attn(query=q, key=q, value=q)
        q = self.norm_self(q + self_out)
        # FFN + Add & Norm
        ffn_out = self.ffn(q)
        q = self.norm_ffn(q + ffn_out)
        return q # [B, n_detectable_humans, D]

class Perception(hk.Module):
    def __init__(
            self,
            name: str,
            n_detectable_humans: int,
            max_humans_velocity: float,
            max_lidar_distance: float,
            embed_dim: int,
            n_sectors: int,
            lidar_angles_robot_frame: jnp.ndarray,
            beam_dropout_rate: float = 0.0,
        ):
        super().__init__(name=name)
        self.n_detectable_humans = n_detectable_humans
        self.embed_dim = embed_dim
        self.max_humans_velocity = max_humans_velocity
        self.max_lidar_distance = max_lidar_distance
        self.lidar_angles_robot_frame = lidar_angles_robot_frame
        self.n_sectors = n_sectors
        # Modules
        self.input_proj = hk.Linear(embed_dim, name="input_projection")
        self.perception = SpatioTemporalEncoder(
            embed_dim=embed_dim, 
            n_sectors=n_sectors, 
            name="spatio_temporal_encoder", 
            lidar_angles_robot_frame=lidar_angles_robot_frame, 
            beam_dropout_rate=beam_dropout_rate
        )
        self.decoder = HCGQueryDecoder(n_detectable_humans, embed_dim, name="decoder")
        # Head: 11 params (weight, mu_x, mu_y, log_sig_x, log_sig_y, corr, mu_vx, mu_vy, log_sig_vx, log_sig_vy, corr_v)
        self.head_hum = hk.Linear(11, w_init=hk.initializers.VarianceScaling(0.01))

    def limit_vector_norm(self, raw_vector:jnp.ndarray, max_norm:float) -> jnp.ndarray:
        sq_sum = jnp.sum(jnp.square(raw_vector), axis=-1, keepdims=True)
        current_norm = jnp.sqrt(sq_sum + 1e-6)
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

    def __call__(self, x, stop_gradient=False, key=None, external_mask=None):
        # x input: [batch B, n_stack (T), num_beams (L), 7]
        has_batch = x.ndim == 4
        if not has_batch:
            x = jnp.expand_dims(x, 0) # [1, T, L, F]
        # 1. Feature Projection
        x_emb = self.input_proj(x) # [B, T, L, embed_dim]
        # 2. Spatio-Temporal Encoding
        encoded_features, mask = self.perception(x_emb, x, key=key, external_mask=external_mask) # [B, T, L, embed_dim]
        last_scan_embeddings = encoded_features[:, 0, :, :] # [B, N_sectors, embed_dim]
        # 3. Decoding via Queries
        latents = self.decoder(encoded_features) # [B, n_detectable_humans, embed_dim]        
        # 4. Heads & Parameter Transformation
        raw_hum = self.head_hum(latents)
        hum_distr = self._params_to_human_centric_gaussians(raw_hum)
        if not has_batch:
            hum_distr = tree_map(lambda t: t[0], hum_distr)
            last_scan_embeddings = last_scan_embeddings[0]
        if stop_gradient:
            hum_distr = tree_map(lambda t: lax.stop_gradient(t), hum_distr)
            last_scan_embeddings = lax.stop_gradient(last_scan_embeddings)
        return hum_distr, last_scan_embeddings, mask
    
class ActorCritic(hk.Module):
    def __init__(
            self,
            name: str,
            n_detectable_humans: int,
            v_max: float,
            wheels_distance: float,
            n_sectors: int,
            mlp_params: dict = {
                "activation": nn.relu,
                "activate_final": False,
                "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
                "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            },
            initial_concentration: float = 0.,
    ) -> None:
        super().__init__(name=name)
        self.n_detectable_humans = n_detectable_humans
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.initial_concentration = initial_concentration
        # Dimensions
        self.n_sectors = n_sectors
        self.n_outputs = 3  # Dirichlet distribution over 3 action vertices
        self.mlp_params = mlp_params
        # Scan embedding reducer
        self.scan_reducer = hk.Linear(1, name="scan_reducer")
        # 2. Self Attention Mechanism
        self.attention = hk.MultiHeadAttention(
            num_heads=2,
            key_size=(n_sectors + 20)//2,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            name="hcg_self_attention"
        )
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.att_ffn = hk.nets.MLP([n_sectors + 20], activation=nn.gelu, activate_final=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
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
            x: HCGs + Robot State. Shape (n_detectable_humans, 20) or (batch_size, n_detectable_humans, 20)
               - Index 0-10: HCG parameters (Mean, LogSigma, Corr, Weight)
               - Index 10: HCG Weight (Score)
               - Index 11-19: Tiled Robot Params
            y: LiDAR embedding. Shape (n_sectors, scan_embedding_dim,) or (batch_size, n_sectors, scan_embedding_dim)

        Returns:
            sampled_actions: Sampled actions from the policy. Shape (2,) or (batch_size, 2)
            distributions: Dict containing the Dirichlet distribution parameters.
            state_values: State value estimates from the critic. Shape (,) or (batch_size,)
        """
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        has_batch = x.ndim == 3
        if not has_batch:
            x = jnp.expand_dims(x, 0)
            y = jnp.expand_dims(y, 0)
        batch_size = x.shape[0]
        keys = random.split(random_key, batch_size)
        hcg_scores = x[..., 10:11] 
        global_robot_state = x[:, 0, 11:] 
        action_space_params = global_robot_state[:, :3]
        ### CONTEXT EXTRACTION
        scalar_scan = self.scan_reducer(y)  # (Batch, n_sectors, 1)
        y = jnp.squeeze(scalar_scan, axis=-1) # (Batch, n_sectors)
        y_tiled = jnp.broadcast_to(y[:, None, :], (batch_size, self.n_detectable_humans, y.shape[-1]))
        embeddings = jnp.concatenate([x, y_tiled], axis=-1) # [Batch, N_Humans, 20 + n_sectors]
        # SCENE-ATTENTION MECHANISM
        att_out = self.attention(embeddings, embeddings, embeddings) # (Batch, N, 20 + n_sectors)
        att_embeddings = self.layer_norm1(embeddings + att_out) # (Batch, N, 20 + n_sectors)
        ffn_out = self.att_ffn(att_embeddings)
        att_embeddings = self.layer_norm2(att_embeddings + ffn_out)
        summed_embeddings = jnp.sum(att_embeddings * hcg_scores, axis=1) # (Batch, 20 + n_sectors)
        sum_of_weights = jnp.sum(hcg_scores, axis=1) # (Batch, 1)
        base_mean = summed_embeddings / (sum_of_weights + 1e-5)  # (Batch, 20 + n_sectors)
        presence_gate = jnp.tanh(sum_of_weights) # (Batch, 1) Encodes if humans are present in the scene
        pooled_hcg_context = base_mean * presence_gate # (Batch, 20 + n_sectors)
        context = jnp.concatenate([
            pooled_hcg_context, 
            global_robot_state, 
            y
        ], axis=-1)  # (20 + n_sectors + 9 + n_sectors,)
        ### ACTOR
        ## Compute Dirichlet distribution parameters
        alphas = nn.softplus(self.actor_head(context)) + 1  # (Batch, 3)
        concentration = jnp.sum(alphas, axis=-1)  # (Batch,)
        ## Compute dirchlet distribution vetices
        zeros = jnp.zeros((batch_size,))
        v1 = jnp.stack([zeros, action_space_params[:, 1] * self.wmax], axis=-1)
        v2 = jnp.stack([zeros, action_space_params[:, 2] * self.wmin], axis=-1)
        v3 = jnp.stack([action_space_params[:, 0] * self.vmax, zeros], axis=-1)
        vertices = jnp.stack([v1, v2, v3], axis=1)  # Shape: (batch_size, 3, 2)
        distributions = {"alphas": alphas, "vertices": vertices}
        ## Sample action
        sampled_actions = vmap(self.dirichlet.sample)(distributions, keys)
        ### CRITIC
        state_values = self.critic_head(context) # (Batch, 1)
        state_values = jnp.squeeze(state_values, axis=-1) # (Batch,)
        if not has_batch:
            sampled_actions = sampled_actions[0]
            state_values = state_values[0]
            distributions = tree_map(lambda t: t[0], distributions)
        return sampled_actions, distributions, concentration, state_values

class E2E(hk.Module):
    def __init__(
        self,
        name: str,
        perception_name: str,
        controller_name: str,
        lidar_angles_robot_frame: jnp.ndarray,
        n_detectable_humans: int,
        max_humans_velocity: float,
        max_lidar_distance: float,
        v_max: float,
        wheels_distance: float,
        embed_dim: int,
        n_sectors: int,
        mlp_params: dict = {
            "activation": nn.relu,
            "activate_final": False,
            "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
            "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
        },
        initial_concentration: float = 0.,
        beam_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.n_detectable_humans = n_detectable_humans
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.initial_concentration = initial_concentration
        self.max_humans_velocity = max_humans_velocity
        self.max_lidar_distance = max_lidar_distance
        # Dimensions
        self.embed_dim = embed_dim
        self.n_sectors = n_sectors
        self.n_outputs = 3  # Dirichlet distribution over 3 action vertices
        self.mlp_params = mlp_params
        self.lidar_angles_robot_frame = lidar_angles_robot_frame
        # Initialize Perception module
        self.perception = Perception(
            perception_name,
            n_detectable_humans=n_detectable_humans,
            max_humans_velocity=max_humans_velocity,
            max_lidar_distance=max_lidar_distance,
            embed_dim=embed_dim,
            n_sectors=n_sectors,
            lidar_angles_robot_frame=lidar_angles_robot_frame,
            beam_dropout_rate=beam_dropout_rate,
        )
        # Initialize Actor-Critic module
        self.actor_critic = ActorCritic(
            controller_name,
            n_detectable_humans=n_detectable_humans,
            v_max=v_max,
            wheels_distance=wheels_distance,
            mlp_params=mlp_params,
            initial_concentration=initial_concentration,
            n_sectors=n_sectors,
        )

    def __call__(
        self,
        x: jnp.ndarray, # Perception input (B, n_stack, num_beams, 7)
        y: jnp.ndarray, # Additional actor-critic input (B, n_detectable_humans, 9)
        stop_perception_gradient: bool = False,
        only_perception: bool = False,
        **kwargs: dict,
    ) -> tuple:
        # Extract random key
        action_sample_key = kwargs.get("random_key", random.PRNGKey(0))
        perception_key = kwargs.get("perception_key", None)
        external_mask = kwargs.get("external_mask", None)
        ## PERCEPTION
        perception_output, scan_embedding, mask = self.perception(x, stop_gradient=stop_perception_gradient, key=perception_key, external_mask=external_mask) # perception_output: (B, N, 11), scan_embedding: (B, E)
        # Prepare actor-critic input
        hcgs = jnp.concatenate((
            perception_output["pos_distrs"]["means"],
            perception_output["pos_distrs"]["logsigmas"],
            perception_output["pos_distrs"]["correlation"][..., jnp.newaxis],
            perception_output["vel_distrs"]["means"],
            perception_output["vel_distrs"]["logsigmas"],
            perception_output["vel_distrs"]["correlation"][..., jnp.newaxis],
            perception_output["weights"][..., jnp.newaxis],
        ), axis=-1)  # Shape: (B, N, 11)
        # Concatenate all inputs
        actor_input = jnp.concatenate((
            hcgs,
            y,
        ), axis=-1)  # Shape: (B, N, 20)
        if only_perception:
            sampled_actions = None
            distributions = None
            concentration = None
            state_values = None
        else:
            ## CONTROL
            sampled_actions, distributions, concentration, state_values = self.actor_critic(
                actor_input,
                scan_embedding,
                random_key=action_sample_key
            )
        return perception_output, actor_input, sampled_actions, distributions, concentration, state_values, mask

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
        embedding_dim:int=96,
        n_sectors:int=60,
        n_stack_for_action_space_bounding:int=1,
        beam_dropout_rate:float=0.0,
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
        assert n_stack_for_action_space_bounding <= n_stack, "n_stack_for_action_space_bounding must be less than or equal to n_stack"
        # Configurable attributes
        super().__init__(discount=gamma)
        self.robot_radius = robot_radius
        self.v_max = v_max
        self.dt = dt
        self.wheels_distance = wheels_distance
        self.n_stack = n_stack
        self.n_stack_for_action_space_bounding = n_stack_for_action_space_bounding
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
        self.embedding_dim = embedding_dim
        self.n_sectors = n_sectors
        self.beam_dropout_rate = beam_dropout_rate
        # Default attributes
        self.name = "JESSI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.dirichlet = Dirichlet()
        self.bivariate_gaussian = BivariateGaussian()
        # Initialize Perception network
        self.perception_name = "lidar_perception"
        @hk.transform
        def perception_network(x, stop_gradient=False, key=None, external_mask=None) -> jnp.ndarray:
            net = Perception(
                self.perception_name, 
                self.n_detectable_humans, 
                self.max_humans_velocity, 
                self.lidar_max_dist, 
                embed_dim=self.embedding_dim, 
                n_sectors=n_sectors,
                lidar_angles_robot_frame=self.lidar_angles_robot_frame,
                beam_dropout_rate=self.beam_dropout_rate,
            )
            return net(x, stop_gradient=stop_gradient, key=key, external_mask=external_mask)
        self.perception = perception_network
        # Initialize Actor Critic network
        self.actor_critic_name = "actor_network"
        @hk.transform
        def actor_critic_network(x, y, **kwargs) -> jnp.ndarray:
            actor_critic = ActorCritic(
                self.actor_critic_name, 
                self.n_detectable_humans, 
                self.v_max, 
                self.wheels_distance, 
                n_sectors=self.n_sectors
            ) 
            return actor_critic(x, y, **kwargs)
        self.actor_critic = actor_critic_network
        # Initialize E2E Actor Critic network
        self.e2e_name = "e2e"
        @hk.transform
        def e2e_network(x, y, stop_perception_gradient=False, only_perception=False, **kwargs) -> jnp.ndarray:
            e2e = E2E(
                self.e2e_name,
                self.perception_name,
                self.actor_critic_name,
                self.lidar_angles_robot_frame,
                n_detectable_humans=self.n_detectable_humans,
                max_humans_velocity=self.max_humans_velocity,
                max_lidar_distance=self.lidar_max_dist,
                v_max=self.v_max,
                wheels_distance=self.wheels_distance,
                embed_dim=self.embedding_dim,
                n_sectors=self.n_sectors,
                beam_dropout_rate=self.beam_dropout_rate,
            ) 
            return e2e(x, y, stop_perception_gradient=stop_perception_gradient, only_perception=only_perception, **kwargs)
        self.e2e = e2e_network

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
    
    @partial(jit, static_argnames=("self"))
    def _perception_loss(
        self,
        human_distrs: dict,
        targets:jnp.ndarray,
        lambda_pos_reg:float=2.0,
        lambda_vel_reg:float=1.0,
        lambda_cls:float=1.0,
    ) -> jnp.ndarray:
        # Extract target data
        human_positions = targets["gt_poses"] # Shape: (B, M, 2)
        human_velocities = targets["gt_vels"] # Shape: (B, M, 2)
        human_mask = targets["gt_mask"] # Shape: (B, M) -> 1 if human exists, 0 otherwise
        # Extract dimensions
        B, K, _ = human_distrs['pos_distrs']['means'].shape
        _, M, _ = human_positions.shape
        ### Bipartite matching
        ## Cost matrix
        diff = jnp.expand_dims(human_distrs['pos_distrs']['means'], 2) - jnp.expand_dims(human_positions, 1) # (B, K, 1, 2) - (B, 1, M, 2)
        dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-6) # Shape (B, K, M)
        prob_cost = -jnp.log(jnp.expand_dims(human_distrs['weights'], 2) + 1e-6) # (B, K, 1)
        cost_matrix = lambda_pos_reg * dist + lambda_cls * prob_cost # (B, K, M)
        ## Matching
        assigned_query_idx, assigned_gt_idx = vmap(optax.assignment.hungarian_algorithm)(cost_matrix) # Shapes: (B, M), (B, M)
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

    @partial(jit, static_argnames=("self"))
    def _safety_loss(
        self,
        actor_distributions: dict,
        # {alphas (B, 3), vertices (B, 3, 2)}
        human_distrs: dict,
        # {
        #   pos_distrs: {means (B, M, 2), logsigmas (B, M, 2), correlation (B, M)}, 
        #   vel_distrs: {means (B, M, 2), logsigmas (B, M, 2), correlation (B, M)}, 
        #   weights (B, M)
        # }
        score_threshold: float = 0.5,
        dt: float = 0.5, # Time horizon to evaluate safety
        uncertainty_sensitivity: float = 10.0,
    ) -> jnp.ndarray:
        """
        Computes a safety loss that penalizes MEAN actions that bring the robot too close to a collision
        in the next time interval in the relative position space, considering the uncertainty in human positions.
        """
        ### Compute distance threshold
        distance_threshold = self.v_max * dt * 2
        ### Extract human distribution parameters (STOP GRADIENTS)
        raw_weights = lax.stop_gradient(human_distrs['weights']) # (B, M)
        if raw_weights.ndim == 3:
            human_weights = raw_weights[..., 0]
        else:
            human_weights = raw_weights
        h_pos_0 = lax.stop_gradient(human_distrs['pos_distrs']['means']) # (B, M, 2)
        h_vel = lax.stop_gradient(human_distrs['vel_distrs']['means'])   # (B, M, 2)
        def get_cov_matrix(logsig, corr):
            sig = jnp.exp(logsig) # (B, M, 2)
            sig_x = sig[..., 0]
            sig_y = sig[..., 1]
            if corr.ndim == 3:
                rho = corr[..., 0]
            else:
                rho = corr # (B, M)
            var_x = jnp.square(sig_x)
            var_y = jnp.square(sig_y)
            cov_xy = rho * sig_x * sig_y
            row1 = jnp.stack([var_x, cov_xy], axis=-1)
            row2 = jnp.stack([cov_xy, var_y], axis=-1)
            return jnp.stack([row1, row2], axis=-2)
        sigma_pos = get_cov_matrix(
            lax.stop_gradient(human_distrs['pos_distrs']['logsigmas']),
            lax.stop_gradient(human_distrs['pos_distrs']['correlation'])
        )
        sigma_vel = get_cov_matrix(
            lax.stop_gradient(human_distrs['vel_distrs']['logsigmas']),
            lax.stop_gradient(human_distrs['vel_distrs']['correlation'])
        )
        B, M, _ = h_pos_0.shape
        ### Filter HCGs by score threshold and distance threshold
        score_mask = human_weights >= score_threshold  # Shape: (B, M)
        distance_mask = jnp.linalg.norm(h_pos_0, axis=-1) <= distance_threshold
        final_mask = score_mask & distance_mask  # Shape: (B, M)
        ### Compute mean actions
        mean_actions = vmap(self.dirichlet.mean)(actor_distributions)  # Shape: (B, 2)
        v_cmd = mean_actions[:, 0]
        w_cmd = mean_actions[:, 1]
        ### Compute next positions
        theta = w_cmd * dt
        eps = 1e-6
        small_angle = jnp.abs(w_cmd) < eps
        safe_w_cmd = jnp.where(small_angle, 1.0, w_cmd) # JUST FOR NUMERICAL STABILITY OF GRADIENTS (the where later will discard the value 1.0 if small_angle is True)
        r_dx_linear = v_cmd * dt
        r_dy_linear = 0.0
        r_dx_curved = (v_cmd / safe_w_cmd) * jnp.sin(theta)
        r_dy_curved = (v_cmd / safe_w_cmd) * (1.0 - jnp.cos(theta))
        r_disp_x = jnp.where(small_angle, r_dx_linear, r_dx_curved)
        r_disp_y = jnp.where(small_angle, r_dy_linear, r_dy_curved) 
        r_disp = jnp.stack([r_disp_x, r_disp_y], axis=-1) # (B, 2)
        h_pos_next = h_pos_0 + (h_vel * dt) # (B, M, 2)
        ### Compute minimum distance between robot path and human predicted positions
        P_start = h_pos_0
        P_end = h_pos_next - jnp.expand_dims(r_disp, 1) # (B, M, 2)
        u = P_end - P_start
        u_sq_norm = jnp.sum(u**2, axis=-1)
        u_sq_norm = jnp.maximum(u_sq_norm, 1e-6)
        dot_prod = jnp.sum(P_start * u, axis=-1)
        valid_motion = u_sq_norm > 1e-4
        t_star = jnp.where(valid_motion, -dot_prod / u_sq_norm, 0.0) # If no relative motion, set t-star to 0
        t_clip = jnp.clip(t_star, 0.0, 1.0) # (B, M)
        closest_point = P_start + jnp.expand_dims(t_clip, -1) * u # (B, M, 2)
        min_dist_sq = jnp.sum(closest_point**2, axis=-1) # (B, M)
        min_dist = jnp.sqrt(min_dist_sq + 1e-6)
        ### Compute uncertainty at time t-clip
        time_real = t_clip * dt  # (B, M)
        time_sq = jnp.square(time_real) # (B, M)
        time_sq_bd = time_sq[..., None, None]
        sigma_tot = sigma_pos + (time_sq_bd * sigma_vel) # (B, M, 2, 2)
        d = closest_point # (B, M, 2)
        d_norm_sq = min_dist_sq # (B, M)
        safe_denominator = jnp.maximum(d_norm_sq, 1e-6)
        sigma_proj_sq_unnormalized = jnp.einsum('bmi,bmij,bmj->bm', d, sigma_tot, d)
        variance_along_collision = sigma_proj_sq_unnormalized / safe_denominator
        confidence_weight = 1.0 / (1.0 + uncertainty_sensitivity * variance_along_collision)
        ### Compute alignement factor (if collision is frontal, penalize more)
        safe_den = lax.stop_gradient(min_dist) + 1e-3 # To avoid division by zero (it is just a normalization factor)
        cos_impact = closest_point[..., 0] / safe_den
        alignment_factor = 1.0 + 5.0 * jnp.maximum(0.0, cos_impact)
        ### Compute safety loss
        required_safety_dist = self.robot_radius * 2 + 0.05
        violations = jnp.maximum(0.0, required_safety_dist - min_dist) # (B, M)
        weighted_loss = jnp.square(violations) * alignment_factor * confidence_weight * final_mask # (B, M)
        frame_safety_loss = jnp.sum(weighted_loss, axis=-1) # (B,)
        scalar_loss = jnp.mean(frame_safety_loss) # ()
        return scalar_loss
    
    # Public methods

    def merge_nns_params(
        self, 
        perception_params, 
        actor_critic_params
    ) -> dict:
        """
        Convert modular network parameters to end-to-end network parameters.
        """
        merged = hk.data_structures.merge(perception_params, actor_critic_params)
        return {f"{self.e2e_name}/~/{k}": v for k, v in merged.items()}
    
    def split_nns_params(
        self, 
        e2e_params: dict
    ) -> tuple:
        """
        Convert end-to-end network parameters to modular network parameters.
        """
        perception_params, actor_critic_params = hk.data_structures.partition(lambda m, n, p: m.startswith(f"{self.e2e_name}/~/{self.perception_name}"), e2e_params)
        perception_params = {k.replace(f"{self.e2e_name}/~/", ""): v for k, v in perception_params.items()}
        actor_critic_params = {k.replace(f"{self.e2e_name}/~/", ""): v for k, v in actor_critic_params.items()}
        return perception_params, actor_critic_params

    @partial(jit, static_argnames=("self"))
    def init_nns(
        self, 
        key:random.PRNGKey, 
    ) -> tuple:
        # Input is shaped (self.n_stack, self.lidar_num_rays, 7)
        perception_params = self.perception.init(key, jnp.zeros((2, 2, 7))) # Cardinality invariant for n_stack and lidar_num_rays
        actor_critic_params = self.actor_critic.init(key, jnp.zeros((self.n_detectable_humans, 20)), jnp.zeros((self.n_sectors, self.embedding_dim)))
        e2e_params = self.e2e.init(key, jnp.zeros((2, 2, 7)), jnp.zeros((self.n_detectable_humans, 9))) # Cardinality invariant for n_stack and lidar_num_rays
        return perception_params, actor_critic_params, e2e_params

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
        - processed_obs (n_stack, lidar_num_rays * 2): aligned LiDAR stack. First information corresponds to the most recent observation.
        """
        ref_position = obs[0,:2]
        ref_orientation = obs[0,2]
        return vmap(JESSI._align_lidar_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)

    @partial(jit, static_argnames=("self"))
    def compute_robot_state_input(
        self,
        action_space_params,
        robot_goal, # In cartesian coordinates (gx, gy) IN THE ROBOT FRAME
    ):
        robot_goal_dist = jnp.linalg.norm(robot_goal)
        robot_goal_theta = jnp.arctan2(robot_goal[1], robot_goal[0])
        robot_goal_sin_theta = jnp.sin(robot_goal_theta)
        robot_goal_cos_theta = jnp.cos(robot_goal_theta)
        tiled_action_space_params = jnp.tile(action_space_params, (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        tiled_robot_params = jnp.tile(jnp.array([self.v_max, self.robot_radius, self.wheels_distance]), (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        tiled_robot_goals = jnp.tile(jnp.array([robot_goal_dist, robot_goal_sin_theta, robot_goal_cos_theta]), (self.n_detectable_humans,1)) # Shape: (n_detectable_humans, 3)
        robot_state_input = jnp.concatenate((
            tiled_action_space_params,
            tiled_robot_goals,
            tiled_robot_params,
        ), axis=-1)  # Shape: (n_detectable_humans, 9)
        return robot_state_input

    @partial(jit, static_argnames=("self"))
    def compute_actor_input(
        self,
        hcgs,
        action_space_params,
        robot_goal, # In cartesian coordinates (gx, gy) IN THE ROBOT FRAME
    ):
        # Compute ROBOT state inputs
        robot_state_input = self.compute_robot_state_input(action_space_params, robot_goal)
        # HCGs from dict to jnp.array
        hcgs = jnp.concatenate((
            hcgs["pos_distrs"]["means"],
            hcgs["pos_distrs"]["logsigmas"],
            hcgs["pos_distrs"]["correlation"][..., jnp.newaxis],
            hcgs["vel_distrs"]["means"],
            hcgs["vel_distrs"]["logsigmas"],
            hcgs["vel_distrs"]["correlation"][..., jnp.newaxis],
            hcgs["weights"][..., jnp.newaxis],
        ), axis=-1)  # Shape: (n_detectable_humans, 11)
        # Concatenate all inputs
        actor_input = jnp.concatenate((
            hcgs,
            robot_state_input,
        ), axis=-1)  # Shape: (n_detectable_humans, 11 + 9)
        return actor_input

    @partial(jit, static_argnames=("self"))
    def compute_perception_input(
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
        7 features per token: [norm_dist, hit, x, y, sin_fixed_theta (theta of beam in the robot frame), cos_fixed_theta (theta of beam in the robot frame), delta_t (time difference from the most recent scan)].
        - last LiDAR point cloud (lidar_num_rays, 2): in robot frame of the most recent observation.
        """
        # Align LiDAR scans - (x,y) coordinates of pointcloud in the robot frame, first information corresponds to the most recent observation.
        aligned_lidar_scans = self.align_lidar(obs)[0]  # Shape: (n_stack, lidar_num_rays, 2)
        point_cloud_for_bounding = aligned_lidar_scans[:self.n_stack_for_action_space_bounding,:, :]  # Shape: (n_stack_for_action_space_bounding, lidar_num_rays, 2)
        point_cloud_for_bounding = jnp.reshape(
            point_cloud_for_bounding,
            (self.n_stack_for_action_space_bounding * self.lidar_num_rays, 2)
        )  # Shape: (n_stack_for_action_space_bounding * lidar_num_rays, 2)
        # Compute LiDAR tokens
        @jit
        def compute_beam_token(
            scan_index:int,
            point:jnp.ndarray,
        ) -> jnp.ndarray:
            # Extract point coordinates
            x, y = point
            # Compute beam features
            distance = jnp.linalg.norm(point)
            current_theta = jnp.arctan2(y, x)
            sin_current_theta = jnp.sin(current_theta)
            cos_current_theta = jnp.cos(current_theta)
            hit = jnp.where(distance < self.lidar_max_dist, 1.0, 0.0)
            # Compute stack index features
            delta_t = scan_index * self.dt
            return jnp.array([
                distance/self.max_beam_range,  # Normalize distance
                hit,
                x,
                y,
                sin_current_theta,
                cos_current_theta,
                delta_t,
            ])
        encoder_input = vmap(vmap(compute_beam_token, in_axes=(None, 0)), in_axes=(0, 0))(
            jnp.arange(self.n_stack),
            aligned_lidar_scans,
        )  # Shape: (n_stack, lidar_num_rays, 7)
        # Optionally select TOP K beams for each stack here to reduce computation
        # First stack is the most recent one!!!
        return encoder_input, point_cloud_for_bounding

    @partial(jit, static_argnames=("self"))
    def compute_e2e_input(
        self,
        obs:jnp.ndarray,
        robot_goal:jnp.ndarray,
    ) -> jnp.ndarray:
        # Compute encoder input and last lidar point cloud (for action bounding)
        perception_input, point_cloud_for_bounding = self.compute_perception_input(obs)
        # Compute bounded action space parameters and add it to the input
        bounding_parameters = self.bound_action_space(
            point_cloud_for_bounding,  
        )
        # Prepare input for network
        robot_position = obs[0,:2]
        robot_orientation = obs[0,2]
        c, s = jnp.cos(-robot_orientation), jnp.sin(-robot_orientation)
        R = jnp.array([[c, -s],
                    [s,  c]])
        translated_position = robot_goal - robot_position
        rc_robot_goal = R @ translated_position
        robot_state_input = self.compute_robot_state_input(
            bounding_parameters,
            rc_robot_goal,
        )
        return perception_input, robot_state_input

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict,
        e2e_network_params:dict,
        sample:bool = False,
    ) -> jnp.ndarray:
        # Compute encoder input and last lidar point cloud (for action bounding)
        perception_input, robot_state_input = self.compute_e2e_input(
            obs,
            info["robot_goal"],
        )
        # Compute action
        key, subkey = random.split(key)
        perception_output, actor_input, sampled_action, actor_distr, concentration, state_value, mask = self.e2e.apply(
            e2e_network_params, 
            None, 
            perception_input,
            robot_state_input,
            random_key=subkey
        )
        action = lax.cond(sample, lambda _: sampled_action, lambda _: self.dirichlet.mean(actor_distr), None)
        return action, key, perception_input, robot_state_input, actor_input, sampled_action, perception_output, actor_distr, state_value, mask
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        e2e_network_params,
        sample,
    ):
        return vmap(JESSI.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos,
            e2e_network_params,
            sample,
        )   
    
    @partial(jit, static_argnames=("self"))
    def perception_loss(
        self,
        current_params:dict,
        inputs:jnp.ndarray,
        targets:jnp.ndarray,
        lambda_pos_reg:float=2.0,
        lambda_vel_reg:float=1.0,
        lambda_cls:float=1.0,
        key=None,
        ) -> jnp.ndarray:
        # B: batch size, K: number of HCGs, M: max number of ground truth humans
        # Compute the prediction
        human_distrs, _, _= self.perception.apply(current_params, None, inputs, key=key)
        return self._perception_loss(
            human_distrs,
            targets,
            lambda_pos_reg,
            lambda_vel_reg,
            lambda_cls,
        )

    @partial(jit, static_argnames=("self","actor_critic_optimizer"))
    def update_il(
        self, 
        actor_critic_params:dict,
        actor_critic_optimizer:optax.GradientTransformation, 
        actor_critic_opt_state: jnp.ndarray, 
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
                expert_actions:jnp.ndarray,
                returns:jnp.ndarray,
                ) -> jnp.ndarray:
                
                @partial(vmap, in_axes=(None, 0, 0, 0))
                def _loss_function(
                    current_actor_params:dict,
                    input:jnp.ndarray,
                    expert_action:jnp.ndarray,
                    returnn:jnp.ndarray,
                    ) -> jnp.ndarray:
                    # Compute the prediction (here we should input a key but for now we work only with mean actions)
                    _, predicted_distr, _, predicted_state_value = self.actor_critic.apply(
                        current_actor_params, 
                        None, 
                        input['actor_input'], 
                        input['scan_embedding']
                    )                    
                    ## Compute actor loss (MSE between expert action and predicted mean action)
                    predicted_action = self.dirichlet.mean(predicted_distr)
                    actor_loss = jnp.mean(jnp.square(predicted_action - expert_action))
                    # Compute critic loss
                    critic_loss = jnp.square(predicted_state_value - returnn)
                    # Compute total loss
                    total_loss = actor_loss + .5 *critic_loss
                    return total_loss, (actor_loss, critic_loss)
                
                total_loss, (actor_losses, critic_losses) = _loss_function(
                    current_actor_params,
                    inputs,
                    expert_actions,
                    returns,
                )

                return jnp.mean(total_loss), (jnp.mean(actor_losses), jnp.mean(critic_losses))

            inputs = {
                "actor_input": experiences["actor_inputs"],
                "scan_embedding": experiences["scan_embeddings"],
            }
            expert_actions = experiences["actor_actions"]
            returns = experiences["returns"]
            # Compute the loss and gradients
            (loss, (actor_loss, critic_loss)), grads = value_and_grad(_batch_loss_function, has_aux=True)(
                current_actor_params, 
                inputs,
                expert_actions,
                returns,
            )
            return loss, actor_loss, critic_loss, grads
        # Compute loss and gradients for actor and critic
        actor_critic_loss, actor_loss, critic_loss, actor_critic_grads = _compute_loss_and_gradients(actor_critic_params,experiences)
        # Compute parameter updates
        actor_critic_updates, actor_critic_opt_state = actor_critic_optimizer.update(actor_critic_grads, actor_critic_opt_state)
        # Apply updates
        updated_actor_critic_params = optax.apply_updates(actor_critic_params, actor_critic_updates)
        return (
            updated_actor_critic_params, 
            actor_critic_opt_state, 
            actor_critic_loss, 
            actor_loss,
            critic_loss,
        )
    
    @partial(jit, static_argnames=("self","encoder_optimizer"))
    def update_perception(
        self,
        current_params:dict, 
        encoder_optimizer:optax.GradientTransformation, 
        optimizer_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray],
        key:random.PRNGKey,
        # Experiences: {"inputs":jnp.ndarray, "targets":dict{"gt_mask","gt_poses","gt_vels"}}
    ) -> tuple:
        # Compute loss and gradients
        loss, grads = value_and_grad(self.perception_loss)(
            current_params, 
            experiences["inputs"],
            experiences["targets"],
            key=key,
        )
        grad_norm = optax.global_norm(grads)
        # Compute parameter updates
        updates, optimizer_state = encoder_optimizer.update(grads, optimizer_state, current_params)
        # Apply updates
        updated_params = optax.apply_updates(current_params, updates)
        return updated_params, optimizer_state, loss, grad_norm

    @partial(jit, static_argnames=("self"))
    def perception_metrics(
        self,
        predicted_distrs:dict, 
        # {
        #   "pos_distrs": {"means", "logsigmas", "correlation"},
        #   "vel_distrs": {"means", "logsigmas", "correlation"},
        #   "weights"
        # }
        targets:dict, 
        # {"gt_mask","gt_poses","gt_vels"}
        distance_thresholds:jnp.ndarray = jnp.arange(0.1, 1.02, 0.02), # Distance threshold to consider a correct detection
        score_thresholds:jnp.ndarray = jnp.arange(0.05, 0.97, 0.02), # Minimum score to consider a detection
    ) -> dict:
        """
        Compute perception metrics for a batch of predictions and targets on the threshold grid given by 
        intersection of distance_thresholds (on the position) and score_thresholds.

        args:
        - predicted_distrs: dict containing predicted human distributions
        - targets: dict containing ground truth masks, positions and velocities

        returns:
        - metrics: dict containing for each threshold pair the:
            - true positives
            - false positives
            - false negatives
            - precision
            - recall
            - average displacement error (ADE) for positions (RMSE over true positives)
            - average velocity error (AVE) for velocities (RMSE over true positives)
            - Mahlanobis distance for positions (average over true positives)
            - Mahlanobis distance for velocities (average over true positives)
        """
        ## EXTRACT DATA 
        # Predictions
        pred_means = predicted_distrs['pos_distrs']['means']       # (B, K, 2)
        pred_logsig_pos = predicted_distrs['pos_distrs']['logsigmas']
        pred_corr_pos = predicted_distrs['pos_distrs']['correlation']
        pred_vels = predicted_distrs['vel_distrs']['means']       # (B, K, 2)
        pred_logsig_vel = predicted_distrs['vel_distrs']['logsigmas']
        pred_corr_vel = predicted_distrs['vel_distrs']['correlation']
        raw_weights = predicted_distrs['weights']
        if raw_weights.ndim == 3:
            pred_scores = raw_weights[..., 0] # (B, K, 1) -> (B, K)
        else:
            pred_scores = raw_weights
        # Targets
        gt_pos = targets["gt_poses"]    # (B, M, 2)
        gt_vel = targets["gt_vels"]     # (B, M, 2)
        raw_mask = targets["gt_mask"]
        if raw_mask.ndim == 3:
            gt_mask = raw_mask[..., 0] 
        else:
            gt_mask = raw_mask
        B, K, _ = pred_means.shape
        _, M, _ = gt_pos.shape

        ## MATCHING
        # Cost matrix: Distance (B, K, M)
        diff_matrix = jnp.expand_dims(pred_means, 2) - jnp.expand_dims(gt_pos, 1)
        dist_matrix = jnp.sqrt(jnp.sum(jnp.square(diff_matrix), axis=-1) + 1e-6)
        cost_matrix = dist_matrix 
        assigned_query_idx, assigned_gt_idx = vmap(optax.assignment.hungarian_algorithm)(cost_matrix)
        sort_perm = jnp.argsort(assigned_gt_idx, axis=1) 
        best_pred_idx = jnp.take_along_axis(assigned_query_idx, sort_perm, axis=1) # (B, M)
        def gather_k_to_m(data_k):
            ndim_diff = data_k.ndim - best_pred_idx.ndim
            if ndim_diff > 0:
                expansion = (1,) * ndim_diff
                indices = best_pred_idx.reshape(best_pred_idx.shape + expansion)
            else:
                indices = best_pred_idx
            return jnp.take_along_axis(data_k, indices, axis=1)
        matched_pos_means = gather_k_to_m(pred_means)
        matched_pos_logsig = gather_k_to_m(pred_logsig_pos)
        matched_pos_corr = gather_k_to_m(pred_corr_pos)
        matched_vel_means = gather_k_to_m(pred_vels)
        matched_vel_logsig = gather_k_to_m(pred_logsig_vel)
        matched_vel_corr = gather_k_to_m(pred_corr_vel)
        matched_scores = jnp.take_along_axis(pred_scores, best_pred_idx, axis=1) # (B, M)
        ## COMPUTE RAW ERRORS (B, M) ---
        # Position & Velocity Euclidean Errors
        pos_diff = matched_pos_means - gt_pos
        pos_error = jnp.linalg.norm(pos_diff, axis=-1)
        vel_diff = matched_vel_means - gt_vel
        vel_error = jnp.linalg.norm(vel_diff, axis=-1)
        pos_distr = {
            "means": matched_pos_means,       # (B, M, 2)
            "logsigmas": matched_pos_logsig,  # (B, M, 2)
            "correlation": matched_pos_corr   # (B, M, 1)
        }
        vel_distr = {
            "means": matched_vel_means,
            "logsigmas": matched_vel_logsig,
            "correlation": matched_vel_corr
        }
        md_pos = vmap(self.bivariate_gaussian.batch_mahalanobis)(pos_distr, gt_pos)
        md_vel = vmap(self.bivariate_gaussian.batch_mahalanobis)(vel_distr, gt_vel)

        ## BROADCASTING OVER THRESHOLDS
        # Expand Data: (B, M, 1, 1)
        e_pos_error = jnp.expand_dims(pos_error, axis=(-1, -2)) # (B, M, 1, 1)
        e_scores = jnp.expand_dims(matched_scores, axis=(-1, -2))
        e_gt_mask = jnp.expand_dims(gt_mask, axis=(-1, -2))     # (B, M, 1, 1)
        e_vel_error = jnp.expand_dims(vel_error, axis=(-1, -2))
        e_md_pos = jnp.expand_dims(md_pos, axis=(-1, -2))
        e_md_vel = jnp.expand_dims(md_vel, axis=(-1, -2))
        # Expand Thresholds: (1, 1, N_dist, N_score)
        t_dist = distance_thresholds[None, None, :, None]
        t_score = score_thresholds[None, None, None, :]
        
        ## CALCULATE METRICS ---
        # TRUE POSITIVES (D, S)
        is_tp = (e_gt_mask == 1) & (e_pos_error < t_dist) & (e_scores > t_score)
        tp_count = jnp.sum(is_tp, axis=(0, 1))
        # FALSE NEGATIVES (D, S)
        total_gt = jnp.sum(gt_mask)
        fn_count = total_gt - tp_count
        # FALSE POSITIVES (D, S)
        # Total predictions passing score threshold (B, K, 1, S) -> sum -> (S)
        all_scores_expand = pred_scores[:, :, None, None] # (B, K, 1, 1)
        pred_above_thresh = (all_scores_expand > t_score) # (B, K, 1, S)
        total_pred_positive = jnp.sum(pred_above_thresh, axis=(0, 1)) # (1, S) broadcastable to (D, S)
        fp_count = total_pred_positive - tp_count
        # PRECISION & RECALL
        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / (tp_count + fn_count + 1e-6)
        
        ## AGGREGATE REGRESSION METRICS (over TPs only)
        def compute_mean_metric(metric_map, mask):
            # metric_map: (B, M, 1, 1)
            # mask: (B, M, D, S)
            numerator = jnp.sum(metric_map * mask, axis=(0, 1))
            denominator = jnp.sum(mask, axis=(0, 1)) + 1e-6
            return numerator / denominator
        ade = compute_mean_metric(e_pos_error, is_tp)
        ave = compute_mean_metric(e_vel_error, is_tp)
        mean_md_pos = compute_mean_metric(e_md_pos, is_tp)
        mean_md_vel = compute_mean_metric(e_md_vel, is_tp)
        # Format output
        metrics = {
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "precision": precision,
            "recall": recall,
            "ADE": ade,
            "AVE": ave,
            "mahalanobis_pos": mean_md_pos,
            "mahalanobis_vel": mean_md_vel,
            # Utilities useful for debugging
            "total_gt_objects": total_gt,
            "distance_thresholds": distance_thresholds,
            "score_thresholds": score_thresholds,
        }
        return metrics

    def evaluate(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
        e2e_network_params:dict,
    ) -> dict:
        """
        Test the trained policy over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "JESSI policy can only be evaluated on unicycle kinematics"
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
                action, policy_key, _, _, _, _, _, _, _, _ = self.act(policy_key, obs, info, e2e_network_params, sample=False)
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

    def evaluate_perception(
        self,
        n_steps:int,
        random_seed:int,
        env:LaserNav,
        e2e_network_params:dict,
    ) -> dict:
        """
        Test the trained policy over n_steps steps and compute relative  perception metrics.
        """
        # Assertions
        assert env.scenario != SCENARIOS.index('circular_crossing_with_static_obstacles'), "Perception evaluation is not supported in environments with static obstacles."
        assert SCENARIOS.index('circular_crossing_with_static_obstacles') not in env.hybrid_scenario_subset, "Perception evaluation is not supported in environments with static obstacles."
        # Initialize storage variables
        perception_distrs = {
            "pos_distrs": {
                "means": jnp.full((n_steps, self.n_detectable_humans, 2), jnp.nan),
                "logsigmas": jnp.full((n_steps, self.n_detectable_humans, 2), jnp.nan),
                "correlation": jnp.full((n_steps, self.n_detectable_humans), jnp.nan),
            },
            "vel_distrs": {
                "means": jnp.full((n_steps, self.n_detectable_humans, 2), jnp.nan),
                "logsigmas": jnp.full((n_steps, self.n_detectable_humans, 2), jnp.nan),
                "correlation": jnp.full((n_steps, self.n_detectable_humans), jnp.nan),
            },
            "weights": jnp.full((n_steps, self.n_detectable_humans), jnp.nan),
        }
        gt_targets = {
            "gt_mask": jnp.full((n_steps, env.n_humans), jnp.nan),
            "gt_poses": jnp.full((n_steps, env.n_humans, 2), jnp.nan),
            "gt_vels": jnp.full((n_steps, env.n_humans, 2), jnp.nan),
        }
        @loop_tqdm(n_steps)
        @jit
        def _fori_body(i:int, for_val:tuple):
            # Retrieve data from the tuple
            state, obs, info, outcome, policy_key, reset_key, env_key, steps, perception_distrs, gt_targets = for_val
            # Compute action and perception out
            action, policy_key, _, _, _, _, perception_out, _, _, _ = self.act(policy_key, obs, info, e2e_network_params, sample=False)
            # Save perception outputs and ground truth
            rc_humans_positions, _, rc_humans_velocities, rc_obstacles, _ = env.robot_centric_transform(
                state[:-1,:2], 
                state[:-1,4], 
                vmap(get_linear_velocity)(state[:-1,4], state[:-1,2:4]),
                info["static_obstacles"][-1], 
                state[-1,:2], 
                state[-1,4], 
                info["robot_goal"],
            )
            humans_visibility, _ = env.object_visibility(
                rc_humans_positions, info["humans_parameters"][:,0], rc_obstacles
            )
            humans_in_range = env.humans_inside_lidar_range(
                rc_humans_positions, info["humans_parameters"][:,0]
            )
            perception_distrs = tree_map(
                lambda x, y: x.at[steps].set(y),
                perception_distrs,
                perception_out,
            )
            gt_targets = tree_map(
                lambda x, y: x.at[steps].set(y),
                gt_targets,
                {
                    "gt_mask": humans_visibility & humans_in_range,
                    "gt_poses": rc_humans_positions,
                    "gt_vels": rc_humans_velocities,
                },
            )
            # Step the environment and update step counter
            state, obs, info, _, outcome, (reset_key, env_key) = env.step(state,info,action,test=True,env_key=env_key,reset_if_done=True,reset_key=reset_key)    
            steps += 1
            return state, obs, info, outcome, policy_key, reset_key, env_key, steps, perception_distrs, gt_targets
        # Execute n_steps steps
        print(f"\nExecuting {n_steps} steps with {env.n_humans} humans and {env.n_obstacles} obstacles to evaluate perception...")
        policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed) # We don't care if we generate two identical keys, they operate differently
        env_key = random.PRNGKey(random_seed + 1_000_000)
        state, reset_key, obs, info, init_outcome = env.reset(reset_key)
        for_val_init = (state, obs, info, init_outcome, policy_key, reset_key, env_key, 0, perception_distrs, gt_targets)
        _, _, _, _, _, _, _, _, perception_distrs, gt_targets = lax.fori_loop(0, n_steps, _fori_body, for_val_init)
        # Compute perception metrics
        perception_metrics = self.perception_metrics(
            perception_distrs,
            gt_targets,
        )
        return perception_metrics

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
        humans_visibility_mask,
        static_obstacles,
        p_visualization_threshold_hcgs,
        p_visualization_threshold_dir,
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
            len(humans_distrs['pos_distrs']['means']) == \
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
        gauss_samples = gauss_samples.at[:].set(jnp.stack((gauss_samples[:,1] * jnp.cos(gauss_samples[:,0]), gauss_samples[:,1] * jnp.sin(gauss_samples[:,0])), axis=-1))
        from socialjym.policies.cadrl import CADRL
        from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
        dummy_cadrl = CADRL(DummyReward(kinematics="unicycle"),kinematics="unicycle",v_max=self.v_max,wheels_distance=self.wheels_distance)
        test_action_samples = dummy_cadrl._build_action_space(unicycle_triangle_samples=35)
        # Animate trajectory
        fig = plt.figure(figsize=(21.43,13.57))
        fig.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        outer_gs = fig.add_gridspec(1, 2, width_ratios=[2, 0.4], wspace=0.09)
        gs_left = outer_gs[0].subgridspec(2, 2, wspace=0.0, hspace=0.0)
        axs = [
            fig.add_subplot(gs_left[0, 0]), # Simulation + LiDAR ranges (Top-Left)
            fig.add_subplot(gs_left[0, 1]), # Simulation + Point cloud (Top-Right)
            fig.add_subplot(gs_left[1, 0]), # HCG positions (Bottom-Left)
            fig.add_subplot(gs_left[1, 1]), # HCG velocities (Bottom-Right)
            fig.add_subplot(outer_gs[1]),   # Action space (Right, tall)
        ]
        def animate(frame):
            for i, ax in enumerate(axs):
                ax.clear()
                if i == len(axs) - 1: continue
                ax.set(xlim=x_lims if x_lims is not None else [-10,10], ylim=y_lims if y_lims is not None else [-10,10])
                if i >= 2:
                    ax.set_xlabel('X', labelpad=-5)
                else:
                    ax.set_xticks([])
                if i % 2 == 0:
                    ax.set_ylabel('Y', labelpad=-13)
                else:
                    ax.set_yticks([])
                ax.set_aspect('equal', adjustable='datalim')
                # Plot humans
                for h in range(len(humans_poses[frame])):
                    color = 'blue' if ((humans_visibility_mask[frame][h] == 1) and (i >= 2)) or (i < 2) else 'grey'
                    alpha = 0.6 if ((humans_visibility_mask[frame][h] == 1) and (i >= 2)) or (i < 2) else 0.3
                    head = plt.Circle((humans_poses[frame][h,0] + jnp.cos(humans_poses[frame][h,2]) * humans_radii[frame][h], humans_poses[frame][h,1] + jnp.sin(humans_poses[frame][h,2]) * humans_radii[frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                    ax.add_patch(head)
                    circle = plt.Circle((humans_poses[frame][h,0], humans_poses[frame][h,1]), humans_radii[frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
                    ax.add_patch(circle)
                # Plot human velocities
                for h in range(len(humans_poses[frame])):
                    color = 'blue' if ((humans_visibility_mask[frame][h] == 1) and (i >= 2)) or (i < 2) else 'grey'
                    alpha = 0.6 if ((humans_visibility_mask[frame][h] == 1) and (i >= 2)) or (i < 2) else 0.3
                    ax.arrow(
                        humans_poses[frame][h,0],
                        humans_poses[frame][h,1],
                        humans_velocities[frame][h,0],
                        humans_velocities[frame][h,1],
                        head_width=0.15,
                        head_length=0.15,
                        fc=color,
                        ec=color,
                        alpha=alpha,
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
                if static_obstacles[frame].shape[1] > 1: # Polygon obstacles
                    for o in static_obstacles[frame]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
                else: # One segment obstacles
                    for o in static_obstacles[frame]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
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
            pos_distrs = frame_humans_distrs["pos_distrs"]
            vel_distrs = frame_humans_distrs["vel_distrs"]
            probs = frame_humans_distrs["weights"]
            # AX 1,0 and 1,1: Human-centric Gaussians (HCGs) positions and velocities
            for h in range(self.n_detectable_humans):
                human_pos_distr = tree_map(lambda x: x[h], pos_distrs)
                human_vel_distr = tree_map(lambda x: x[h], vel_distrs)
                pos = rot @ human_pos_distr["means"] + robot_poses[frame,:2]
                vel = rot @ human_vel_distr["means"] + pos
                if probs[h] > 0.5:
                    # Position HCG
                    test_p = self.bivariate_gaussian.batch_p(human_pos_distr, human_pos_distr["means"] + gauss_samples)
                    points_high_p = gauss_samples[test_p > p_visualization_threshold_hcgs]
                    corresponding_colors = test_p[test_p > p_visualization_threshold_hcgs]
                    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + pos
                    axs[2].scatter(pos[0], pos[1], c='red', s=10, marker='x', zorder=100)
                    axs[2].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
                    # Velocity HCG
                    test_p = self.bivariate_gaussian.batch_p(human_vel_distr, human_vel_distr["means"] + gauss_samples)
                    points_high_p = gauss_samples[test_p > p_visualization_threshold_hcgs]
                    corresponding_colors = test_p[test_p > p_visualization_threshold_hcgs]
                    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + vel
                    axs[3].scatter(vel[0], vel[1], c='red', s=10, marker='x', zorder=100)
                    axs[3].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
                else:
                    axs[2].scatter(pos[0], pos[1], c='grey', s=10, marker='x', zorder=99, alpha=0.5)
                    axs[3].scatter(vel[0], vel[1], c='grey', s=10, marker='x', zorder=99, alpha=0.5)
            axs[2].text(
                0.5, -0.13,
                "HCGs positions",
                transform=axs[2].transAxes,
                rotation=0,
                va="center",
                ha="center"
            )
            axs[3].text(
                0.5, -0.13,
                "HCGs velocities",
                transform=axs[3].transAxes,
                rotation=0,
                va="center",
                ha="center"
            )
            # AX :,2: Feasible and bounded action space + action space distribution and action taken
            axs[4].set_xlabel("$v$ (m/s)")
            axs[4].set_ylabel("$\omega$ (rad/s)", labelpad=-15)
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
            samples = test_action_samples[self.dirichlet.batch_is_in_support(actor_distr, test_action_samples)]
            test_action_p = self.dirichlet.batch_p(actor_distr, samples)
            points_high_p = samples[test_action_p > p_visualization_threshold_dir]
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
        lasernav_env:LaserNav,
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
        humans_velocities = lax.cond(
            lasernav_env.humans_policy == HUMAN_POLICIES.index('hsfm'),
            lambda: vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(
                    humans_orientations,
                    humans_body_velocities,
                ),
            lambda: humans_body_velocities,
        )
        rc_humans_positions, _, _, rc_static_obstacles, _ = lasernav_env.batch_robot_centric_transform(
            humans_positions,
            humans_orientations,
            humans_velocities,
            static_obstacles,
            robot_positions,
            robot_orientations,
            goals,
        )
        humans_visibility_mask, _ = lasernav_env.batch_object_visibility(
            rc_humans_positions, 
            humans_radii, 
            rc_static_obstacles
        )
        humans_in_range = lasernav_env.batch_humans_inside_lidar_range(
            rc_humans_positions,
            humans_radii,
        )
        humans_visibility_mask = humans_visibility_mask & humans_in_range
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
            humans_visibility_mask,
            static_obstacles,
            p_visualization_threshold_gmm,
            p_visualization_threshold_dir,
            x_lims,
            y_lims,
            save_video,
        )