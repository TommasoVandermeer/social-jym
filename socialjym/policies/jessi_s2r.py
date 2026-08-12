import jax.numpy as jnp
from jax import nn, random, vmap, jit, lax, value_and_grad, debug
from jax.tree_util import tree_map, tree_leaves
from functools import partial
import haiku as hk
from typing import Sequence
import optax

from socialjym.policies.jessi import JESSI, E2E, MultiHeadAttention
from socialjym.utils.distributions.logistic_normal import LogisticNormal
from jhsfm.hsfm import step as humans_step
from jhsfm.utils import get_standard_humans_parameters as get_standard_humans_parameters
from jhsfm.hsfm import get_linear_velocity

class Actor(hk.Module):
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
            ablation_mode: str = None,
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
        self.n_outputs = 3 # Logistic-normal (3 alphas, 1 per vertex)
        self.mlp_params = mlp_params
        # Scan embedding reducer
        self.scan_reducer = hk.Linear(1, name="scan_reducer")
        # 2. Self Attention Mechanism
        add_size = 22
        self.attention = MultiHeadAttention(
            num_heads=2,
            key_size=(n_sectors + add_size)//2,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            name="hcg_self_attention"
        )
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.att_ffn = hk.nets.MLP([n_sectors + add_size], activation=nn.gelu, activate_final=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # 3. Final Output MLP
        self.actor_head = hk.nets.MLP(
            **mlp_params,
            output_sizes=[100, 50, self.n_outputs], 
            name="actor_head"
        )
        self.action_distribution = LogisticNormal()

    def __call__(
            self, 
            x: jnp.ndarray,
            y: jnp.ndarray,
            **kwargs: dict,
    ) -> tuple:
        """
        Args:
            x: HCGs + Robot State. Shape (n_detectable_humans, 22 + 2*n_actions_history) or (batch_size, n_detectable_humans, 22 + 2*n_actions_history)
               - Index 0-10: HCG parameters (Mean, LogSigma, Corr, Weight)
               - Index 10: HCG Weight (Score)
               - Index 11-21: Tiled Robot Params
               - Index 22-22+2*n_actions_history: Actions history
            y: LiDAR embedding. Shape (n_sectors, scan_embedding_dim,) or (batch_size, n_sectors, scan_embedding_dim)

        Returns:
            sampled_actions: Sampled actions from the policy. Shape (2,) or (batch_size, 2)
            distributions: Dict containing the Dirichlet distribution parameters.
        """
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        has_batch = x.ndim == 3
        if not has_batch:
            x = jnp.expand_dims(x, 0)
            y = jnp.expand_dims(y, 0)
        # Ablation 2: no human uncertainties
        batch_size = x.shape[0]
        keys = random.split(random_key, batch_size)
        hcg_scores = x[..., 10:11] 
        global_robot_state = x[:, 0, 11:22] 
        action_space_params = global_robot_state[:, :3]
        action_history = x[:, 0, 22:]
        x = x[:, :, :22]
        ### CONTEXT EXTRACTION
        scalar_scan = self.scan_reducer(y)  # (Batch, n_sectors, 1)
        y = jnp.squeeze(scalar_scan, axis=-1) # (Batch, n_sectors)
        # SCENE-ATTENTION MECHANISM
        y_tiled = jnp.broadcast_to(y[:, None, :], (batch_size, self.n_detectable_humans, y.shape[-1]))
        embeddings = jnp.concatenate([x, y_tiled], axis=-1) # [Batch, N_Humans, 22 + n_sectors]
        att_out, att_mtrx = self.attention(embeddings, embeddings, embeddings) # (Batch, N, 22 + n_sectors)
        # debug.print("Attention matrix shape: {s}", s=att_mtrx.shape) # (Batch, N, N)
        att_embeddings = self.layer_norm1(embeddings + att_out) # (Batch, N, 22 + n_sectors)
        ffn_out = self.att_ffn(att_embeddings)
        att_embeddings = self.layer_norm2(att_embeddings + ffn_out)
        summed_embeddings = jnp.sum(att_embeddings * hcg_scores, axis=1) # (Batch, 22 + n_sectors)
        sum_of_weights = jnp.sum(hcg_scores, axis=1) # (Batch, 1)
        base_mean = summed_embeddings / (sum_of_weights + 1e-5)  # (Batch, 22 + n_sectors)
        presence_gate = jnp.tanh(sum_of_weights) # (Batch, 1) Encodes if humans are present in the scene
        pooled_hcg_context = base_mean * presence_gate # (Batch, 22 + n_sectors)
        # Attention (wighted by HCG scores) computation for visualization
        mean_att_mtrx = jnp.mean(att_mtrx, axis=1)
        norm_hcg_scores = hcg_scores / (sum_of_weights[:, None, :] + 1e-5)
        human_attention = jnp.sum(mean_att_mtrx * norm_hcg_scores, axis=1) # (Batch, N)
        context = jnp.concatenate([
            pooled_hcg_context, 
            global_robot_state, 
            action_history,
            y
        ], axis=-1)  # (Batch, 31 + 2*n_sectors + 2*n_actions_history,)
        ### ACTOR
        ## Compute dirchlet distribution vetices
        zeros = jnp.zeros((batch_size,))
        v1 = jnp.stack([zeros, action_space_params[:, 1] * self.wmax], axis=-1)
        v2 = jnp.stack([zeros, action_space_params[:, 2] * self.wmin], axis=-1)
        v3 = jnp.stack([action_space_params[:, 0] * self.vmax, zeros], axis=-1)
        vertices = jnp.stack([v1, v2, v3], axis=1)  # Shape: (batch_size, 3, 2)
        distributions = {"vertices": vertices}
        locs = self.actor_head(context)
        raw_logscales_param = hk.get_parameter("raw_logscales", shape=[3], init=hk.initializers.Constant(jnp.arctanh(9/11)))
        logscales_bounded = jnp.tanh(raw_logscales_param) * 11 - 9 # Bound logscales between [-20,2]
        logscales = jnp.broadcast_to(logscales_bounded, locs.shape)
        distributions["locs"] = locs
        distributions["log_scales"] = logscales
        ## Sample action
        sampled_actions = vmap(self.action_distribution.sample)(distributions, keys)
        if not has_batch:
            sampled_actions = sampled_actions[0]
            distributions = tree_map(lambda t: t[0], distributions)
        return sampled_actions, distributions, 0., 0.,  human_attention

class Critic(hk.Module):
    def __init__(
            self,
            embed_dim: int = 64,
            num_heads: int = 4,
            mlp_hidden_dims: Sequence[int] = (256, 128, 64),
            name: str = None,
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dims = mlp_hidden_dims

    def __call__(
            self,
            robot_data: jnp.ndarray, # (B, 12 + 2*n_actions_history,) --> e.g., n_actions_history=5 --> (B, 12 + 10,) = (B, 22,)
            humans_data: jnp.ndarray, # (B, n_humans, 11 + horizon_length*6) --> e.g., n_humans=5, horizon_length=20 --> (B, 5, 11 + 120) = (B, 5, 131)
            static_obstacles_data: jnp.ndarray, # (B, n_obstacles * n_edges, 2, 2) --> e.g., n_obstacles=5, n_edges=4 --> (B, 20, 2, 2)
            **kwargs: dict,
    ) -> jnp.ndarray:
        # START: deal with batch dimension
        has_batch = robot_data.ndim == 2
        if not has_batch:
            robot_data = jnp.expand_dims(robot_data, 0)
            humans_data = jnp.expand_dims(humans_data, 0)
            static_obstacles_data = jnp.expand_dims(static_obstacles_data, 0)

        # MAIN BODY
        # 1. Obstacles processing: Pure Deep Sets (MLP + Max-Pooling)
        obs_flat = jnp.reshape(static_obstacles_data, (static_obstacles_data.shape[0], static_obstacles_data.shape[1], -1))  # (B, n_obstacles * n_edges, 4)
        valid_obs_mask = jnp.all(jnp.isfinite(obs_flat), axis=-1)  
        safe_obs_flat = jnp.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        obs_emb = hk.nets.MLP(
            [self.embed_dim, self.embed_dim], 
            activation=nn.silu,
            name="obstacles_mlp"
        )(safe_obs_flat)  # (B, n_obstacles * n_edges, embed_dim)
        valid_obs_mask_exp = jnp.expand_dims(valid_obs_mask, axis=-1)  # Shape: (B, N_e, 1)
        masked_obs_emb = jnp.where(valid_obs_mask_exp, obs_emb, -1e9)
        obstacles_feat = jnp.max(masked_obs_emb, axis=1)  # (B, embed_dim)
        obstacles_feat = jnp.where(obstacles_feat == -1e9, 0.0, obstacles_feat)
        # 2. Humans processing: Robot-Human Cross-Attention (Q=Robot, K/V=Humans)
        valid_humans_mask = jnp.all(jnp.isfinite(humans_data), axis=-1)  
        safe_humans_data = jnp.nan_to_num(humans_data, nan=0.0, posinf=0.0, neginf=0.0)
        robot_query = hk.Linear(self.embed_dim)(robot_data)[:, None, :]  # (B, 1, embed_dim)
        humans_kv = hk.Linear(self.embed_dim)(safe_humans_data)  # (B, n_humans, embed_dim)
        humans_kv = nn.silu(humans_kv)  # (B, n_humans, embed_dim)
        norm_q = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(robot_query)  # (B, 1, embed_dim)
        norm_kv = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(humans_kv)  # (B, n_humans, embed_dim)
        attn_mask = jnp.expand_dims(valid_humans_mask, axis=(1, 2))
        humans_attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.embed_dim // self.num_heads,
            model_size=self.embed_dim,
            w_init=hk.initializers.VarianceScaling(),
        )(query=norm_q, key=norm_kv, value=norm_kv, mask=attn_mask)  # (B, 1, embed_dim)
        humans_attn = jnp.nan_to_num(humans_attn, nan=0.0)
        humans_feat = jnp.squeeze(robot_query + humans_attn, axis=1)  # (B, embed_dim)
        # 3. Multimodal Fusion & Value Estimation
        fused_features = jnp.concatenate([robot_data, humans_feat, obstacles_feat], axis=-1)  # (B, D_robot + 2 * embed_dim)
        value = hk.nets.MLP(
            output_sizes=list(self.mlp_hidden_dims) + [1],
            activation=nn.silu,
            activate_final=False,
        )(fused_features)  # (B, 1)
        value = jnp.squeeze(value, axis=-1)
        # END: deal with batch dimension
        if not has_batch:
            value = value[0]  # (1,)
        return value

class JESSI_S2R(JESSI):
    """
    A state augmented version of JESSI specialized for SIM2REAL designed to operate on systems with delayed actuation.
    This version also implements a separated and asynchronous actor critic.
    """
    def __init__(
        self, 
        robot_radius:float=0.3,
        v_max:float=1., 
        dt:float=0.25, 
        humans_dt:float=0.01,
        humans_prediction_horizon:int=20,
        humans_trajectory_noise_std:float=0.1,
        wheels_distance:float=0.7, 
        n_stack:int=5,
        n_actions_history:int=5,
        lidar_angular_range=2*jnp.pi,
        lidar_max_dist=10.,
        lidar_num_rays=100,
        lidar_angles_robot_frame=None, # If not specified, rays are evenly distributed in the angular range
        n_detectable_humans:int=10,
        max_humans_velocity:float=1.5,
        max_beam_range:float=10.0, # This is only used to normalize the LiDAR readings before feeding them to the encoder
        embedding_dim:int=32,
        n_sectors:int=60,
        angular_sectors_width_deg:float=18.0, # This is the width of the attention sectors in degrees. It determines how many rays are attended to by each sector.
        n_stack_for_action_space_bounding:int=1,
        beam_dropout_rate:float=0.0,
    ) -> None:
        assert n_actions_history <= n_stack, "The length of the actions history must be greater than the length of the observation stack"
        self.n_actions_history = n_actions_history
        self.humans_dt = humans_dt
        self.humans_prediction_horizon = humans_prediction_horizon
        self.humans_trajectory_noise_std = humans_trajectory_noise_std
        super().__init__(
            robot_radius=robot_radius,
            v_max=v_max, 
            dt=dt, 
            wheels_distance=wheels_distance, 
            n_stack=n_stack,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            lidar_angles_robot_frame=lidar_angles_robot_frame,
            n_detectable_humans=n_detectable_humans,
            max_humans_velocity=max_humans_velocity,
            max_beam_range=max_beam_range, 
            embedding_dim=embedding_dim,
            n_sectors=n_sectors,
            angular_sectors_width_deg=angular_sectors_width_deg, 
            n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
            beam_dropout_rate=beam_dropout_rate,
            ablation_mode=6,
            legit=True, 
        )
        # Initialize Actor network
        self.actor_name = "actor_network"
        @hk.transform
        def actor_network(x, y, **kwargs) -> jnp.ndarray:
            actor = Actor(
                self.actor_name, 
                self.n_detectable_humans, 
                self.v_max, 
                self.wheels_distance, 
                n_sectors=self.n_sectors,
            ) 
            return actor(x, y, **kwargs)
        self.actor = actor_network
        # Initialize Critic network
        self.critic_name = "critic_network"
        @hk.transform
        def critic_network(robot_data, humans_data, static_obstacles_data, **kwargs) -> jnp.ndarray:
            critic = Critic(
            ) 
            return critic(robot_data, humans_data, static_obstacles_data, **kwargs)
        self.critic = critic_network
        # Initialize E2E network
        self.e2e_name = "e2e"
        @hk.transform
        def e2e_network(x, y, stop_perception_gradient=False, only_perception=False, **kwargs) -> jnp.ndarray:
            e2e = E2E(
                Actor,
                self.e2e_name,
                self.perception_name,
                self.actor_name,
                self.lidar_angles_robot_frame,
                n_detectable_humans=self.n_detectable_humans,
                max_humans_velocity=self.max_humans_velocity,
                max_lidar_distance=self.lidar_max_dist,
                v_max=self.v_max,
                wheels_distance=self.wheels_distance,
                embed_dim=self.embedding_dim,
                n_sectors=self.n_sectors,
                beam_dropout_rate=self.beam_dropout_rate,
                ablation_mode=self.ablation_mode,
                legit=self.legit,
            ) 
            return e2e(x, y, stop_perception_gradient=stop_perception_gradient, only_perception=only_perception, **kwargs)
        self.e2e = e2e_network

    @partial(jit, static_argnames=("self"))
    def init_nns(
        self, 
        key:random.PRNGKey, 
    ) -> tuple:
        # Perception input is shaped (self.n_stack, self.lidar_num_rays, 7)
        # Actor input is shaped (n_detectable_humans, 22 + 2*n_actions_history) and (self.n_sectors, self.embedding_dim))
        # Critic input is (12 + 2*n_actions_history,) and (n_humans, 11 + horizon_length*6) and (n_obstacles * n_edges, 2, 2)
        # E2E input is shaped (self.n_stack, self.lidar_num_rays, 7) and (self.n_detectable_humans, 11 + 2*n_actions_history)
        perception_params = self.perception.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 7))) # Cardinality invariant for n_stack and lidar_num_rays
        actor_params = self.actor.init(key, jnp.zeros((self.n_detectable_humans, 22+2*self.n_actions_history)), jnp.zeros((self.n_sectors, self.embedding_dim)))
        critic_params = self.critic.init(key, jnp.zeros((12 + 2*self.n_actions_history,)), jnp.zeros((1, 11 + self.humans_prediction_horizon*6)), jnp.zeros((1, 2, 2))) # Cardinality invariant for n_obstacles and n_edges
        e2e_params = self.e2e.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 7)), jnp.zeros((self.n_detectable_humans, 11+2*self.n_actions_history))) # Cardinality invariant for n_stack and lidar_num_rays
        return perception_params, actor_params, critic_params, e2e_params

    @partial(jit, static_argnames=("self"))
    def compute_robot_state_input(
        self,
        robot_obs_stack, # (N_stack, 11): : Each stack [rx,ry,r_theta,r_radius,r_vx, r_wz,r_a1,r_a2,lidar_timestamp,odom_timestamp,control_timestamp,lidar_measurements]. The first stack is the most recent one.
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
        tiled_robot_velocity = jnp.tile(robot_obs_stack[0,4:6], (self.n_detectable_humans,1))
        tiled_robot_actions = jnp.tile(jnp.reshape(robot_obs_stack[:self.n_actions_history,6:8],(-1,)), (self.n_detectable_humans,1))
        robot_state_input = jnp.concatenate((
            tiled_action_space_params,
            tiled_robot_goals,
            tiled_robot_params,
            tiled_robot_velocity,
            tiled_robot_actions
        ), axis=-1)  # Shape: (n_detectable_humans, 11 + 2*n_actions_history)
        return robot_state_input

    @partial(jit, static_argnames=("self"))
    def next_humans_state(
        self,
        key,
        humans_state,
        humans_goal,
        humans_parameters,
        obstacles,
    ):
        """
        This functions makes a step in time (of length dt) for the humans' state using the Headed Social Force Model (HSFM) with 
        global force guidance for torque and sliding component on the repulsive forces.

        args:
        - key: shape is (1,) random key to inject noise in velocities bvx, bvy, omega (helpful to not overfit over deterministic trajectories)
        - humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
        - humans_parameters: shape is (n_humans, 19) where each row is (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
        - obstacles: shape is (n_humans, n_obstacles, n_edges, 2, 2) where each human can be assigned a different set of obstacles. Each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
        
        output:
        - new_humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        """
        def scan_step(state, _):
            new_state = humans_step(state, humans_goal, humans_parameters, obstacles, self.humans_dt)
            return new_state, new_state
        new_humans_state, _ = lax.scan(
            f=scan_step,
            init=humans_state,
            xs=None,
            length=int(self.dt/self.humans_dt)
        )
        # Inject noise to velocities (safer)
        key, subkey1, subkey2 = random.split(key, 3)
        noised_new_humans_state = new_humans_state.at[:,2:4].set(
            new_humans_state[:,2:4] + random.normal(subkey1, (humans_state.shape[0],2)) * self.humans_trajectory_noise_std
        )
        noised_new_humans_state = noised_new_humans_state.at[:,4].set(
            noised_new_humans_state[:,4] + random.normal(subkey2, (humans_state.shape[0],)) * self.humans_trajectory_noise_std
        )
        # Inject noise to the whole state
        # noised_new_humans_state = new_humans_state + random.normal(subkey1, (humans_state.shape[0],6)) * self.humans_trajectory_noise_std
        # Bound the linear velocity
        # for h in range(len(noised_new_humans_state)):
        #     speed = jnp.linalg.norm(noised_new_humans_state[h,2:4])
        #     noised_new_humans_state = lax.cond(
        #         speed > humans_parameters[h,2],
        #         lambda: noised_new_humans_state.at[h,2:4].set((noised_new_humans_state[h,2:4] / speed) * humans_parameters[h,2]),
        #         lambda: noised_new_humans_state,
        #     )
        return noised_new_humans_state, new_humans_state, key

    @partial(jit, static_argnames=("self"))
    def predict_humans_trajectory(
        self,
        key,
        humans_state,
        humans_goal,
        humans_parameters,
        obstacles,
    ):
        """
        This functions predicts humans' trajectories over a predefined horizon using the Headed Social Force Model (HSFM) with 
        global force guidance for torque and sliding component on the repulsive forces.

        args:
        - key: shape is (1,) random key to inject noise in velocities bvx, bvy, omega (helpful to not overfit over deterministic trajectories)
        - humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
        - humans_parameters: shape is (n_humans, 19) where each row is (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
        - obstacles: shape is (n_humans, n_obstacles, n_edges, 2, 2) where each human can be assigned a different set of obstacles. Each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
        
        output:
        - noised_trajectory: shape is (horizon_length, n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        - denoised_trajectory: shape is (horizon_length, n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        """
        def scan_step(carry, _):
            state, key = carry
            new_state, denoised_state, new_key = self.next_humans_state(key, state, humans_goal, humans_parameters, obstacles)
            return (new_state, new_key), (new_state, denoised_state)
        _, (humans_trajectory, denoised_humans_trajectory) = lax.scan(
            f=scan_step,
            init=(humans_state, key),
            xs=None,
            length=self.humans_prediction_horizon
        )
        return humans_trajectory, denoised_humans_trajectory

    @partial(jit, static_argnames=("self"))
    def critic_forward(
        self,
        random_key:random.PRNGKey,
        critic_params:dict,
        state:jnp.ndarray,
        actions_history:jnp.ndarray,
        env_params:dict,
        robot_params:dict,
        action_space_params:jnp.ndarray,
    ):
        """
        Forward pass of the critic network.

        Args:
            random_key: PRNG key for random number generation. Used for predicting noised humans' trajectory.
            critic_params: Parameters of the critic network.
            state: Current state of the environment. (full state humans + robot)
            robot_goal: Goal position of the robot.
            actions_history: History of actions taken by the robot.
            env_params: Parameters of the environment. A dictionary containing:
                - "static_obstacles": Array of static obstacles in the environment.
                - "humans_goal": Array of goal positions for each human in the environment.
                - "humans_parameters": Array of the HMM parameters for each human in the environment.
            robot_params: Parameters of the robot. A dictionary containing:
                - "robot_goal": Goal position of the robot.
                - "robot_radius": Radius of the robot.
                - "v_max": Maximum linear velocity of the robot.
                - "wheels_distance": Distance between the wheels of the robot.
                - "wheels_max_linear_acceleration": Maximum linear acceleration of the robot's wheels.
                - "robot_delay": Current actuation delay of the robot.
            action_space_params: Parameters alpha, beta, gamma of the restricted safe action space.

        Returns:
            value: Estimated value of the current state.
        """
        ### Preliminaries: predict humans' trajectories and build inputs for the critic network
        # Predict humans' trajectories
        humans_trajectory, _ = self.predict_humans_trajectory(
            random_key,
            state[:-1],
            env_params["humans_goal"],
            env_params["humans_parameters"],
            env_params["static_obstacles"][:-1]
        ) 
        ### Transformations (robot-centric parameterization)
        robot_state = state[-1]
        robot_position = robot_state[:2]
        robot_orientation = robot_state[4]
        rotation_matrix = jnp.array([
            [jnp.cos(robot_orientation), -jnp.sin(robot_orientation)],
            [jnp.sin(robot_orientation),  jnp.cos(robot_orientation)]
        ])
        robot_goal = (robot_params["robot_goal"] - robot_position) @ rotation_matrix # Transform goal to robot-centric coordinates
        humans_state = state[:-1] # Shape: (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
        humans_state = humans_state.at[:, :2].set((humans_state[:, :2] - robot_position) @ rotation_matrix) # Transform humans' positions to robot-centric coordinates
        humans_state = humans_state.at[:, 2:4].set(vmap(get_linear_velocity)(humans_state[:, 4],humans_state[:, 2:4]) @ rotation_matrix) # Transform humans' velocities to robot-centric coordinates
        humans_state = humans_state.at[:, 4].set(humans_state[:, 4] - robot_orientation) # Transform humans' orientations to robot-centric coordinates
        humans_goal = (env_params["humans_goal"] - robot_position) @ rotation_matrix # Transform humans' goals to robot-centric coordinates
        humans_trajectory = humans_trajectory.at[:, :, :2].set((humans_trajectory[:, :, :2] - robot_position) @ rotation_matrix) # Transform humans' trajectories to robot-centric coordinates
        humans_trajectory = humans_trajectory.at[:, :, 2:4].set((vmap(vmap(get_linear_velocity))(humans_trajectory[:, :, 4],humans_trajectory[:, :, 2:4]) @ rotation_matrix)) # Transform humans' velocities to robot-centric coordinates
        humans_trajectory = humans_trajectory.at[:, :, 4].set(humans_trajectory[:, :, 4] - robot_orientation) # Transform humans' orientations to robot-centric coordinates
        static_obstacles = (env_params["static_obstacles"][-1] - robot_position) @ rotation_matrix # Transform static obstacles to robot-centric coordinates
        ### Build Critic inputs: robot_data, humans_data, static_obstacles_data       
        # Build robot_data input. (robot_velocity + robot_goal + robot_params + action_space_params + actions_history)
        robot_data = jnp.concatenate((
            robot_state[2:4], # robot_velocity (v, w)
            robot_goal, # robot_goal
            jnp.array([robot_params["robot_radius"], robot_params["v_max"], robot_params["wheels_distance"], robot_params["wheels_max_linear_acceleration"], robot_params["robot_delay"]]), # robot_params
            action_space_params, # action_space_params
            actions_history.flatten() # actions_history
        ), axis=-1) # Shape: (2 + 2 + 5 + 3 + 2*n_actions_history,) --> e.g., n_actions_history=5 --> (2 + 2 + 5 + 3 + 10,) = (22,)
        # Build humans_data input. (humans_state + humans_trajectory + humans_goal + humans_parameters)
        humans_data = jnp.concatenate((
            humans_state, # humans_state
            jnp.reshape(humans_trajectory, (humans_state.shape[0], -1)), # humans_trajectory
            humans_goal, # humans_goal
            env_params["humans_parameters"][:,:3] # humans_parameters (radius, mass, v_max)
        ), axis=-1) # Shape: (n_humans, 6 + horizon_length*6 + 2 + 3) --> e.g., n_humans=5, horizon_length=20 --> (5, 6 + 120 + 2 + 3) = (5, 131)
        # Build static_obstacles_data input. (static_obstacles)
        static_obstacles_data = jnp.reshape(static_obstacles, (-1, 2, 2)) # Shape: (n_obstacles * n_edges, 2, 2) --> e.g., n_obstacles=5, n_edges=4 --> (20, 2, 2)
        ### Critic forward pass
        value = self.critic.apply(critic_params, None, robot_data, humans_data, static_obstacles_data)
        return value

    @partial(jit, static_argnames=("self"))
    def batch_critic_forward(
        self,
        random_keys:random.PRNGKey,
        critic_params:dict,
        states:jnp.ndarray,
        actions_histories:jnp.ndarray,
        env_params:dict,
        robot_params:dict,
        action_space_params:jnp.ndarray,
    ) -> jnp.ndarray:
        return vmap(JESSI_S2R.critic_forward, in_axes=(None, 0, None, 0, 0, 0, 0, 0))(
            self, 
            random_keys, 
            critic_params, 
            states, 
            actions_histories, 
            env_params, 
            robot_params, 
            action_space_params
        )

    @partial(jit, static_argnames=("self","actor_optimizer","critic_optimizer"))
    def update_il(
        self, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        critic_key:random.PRNGKey,
        critic_params:dict,
        critic_optimizer:optax.GradientTransformation, 
        critic_opt_state: jnp.ndarray, 
        experiences:dict[str:jnp.ndarray], 
        beta_entropy:float=0.01,
    ) -> tuple:
        def _compute_actor_loss_and_gradients(
            current_actor_params:dict,  
            experiences:dict,
        ) -> tuple:
            def _batch_loss_function(
                current_actor_params:dict,
                inputs:jnp.ndarray,
                expert_actions:jnp.ndarray,
                ) -> jnp.ndarray:
                
                @partial(vmap, in_axes=(None, 0, 0))
                def _loss_function(
                    current_actor_params:dict,
                    input:jnp.ndarray,
                    expert_action:jnp.ndarray,
                    ) -> jnp.ndarray:
                    # Compute the prediction (here we should input a key but for now we work only with mean actions)
                    _, predicted_distr, _, _, _ = self.actor.apply(
                        current_actor_params, 
                        None, 
                        input['actor_input'], 
                        input['scan_embedding']
                    )                    
                    ## Compute actor loss (MSE between expert action and predicted mean action)
                    predicted_action = self.action_distribution.mean(predicted_distr)
                    actor_loss = jnp.mean(jnp.square(predicted_action - expert_action))
                    ## Compute actor loss (NLL of expert action uneder current predicted distribution + entropy regularization)
                    # actor_loss = self.action_distribution.neglogp(predicted_distr, expert_action)
                    # # Entropy and final loss computation
                    # entropy = - beta_entropy * self.action_distribution.entropy(predicted_distr)
                    # loss = actor_loss + entropy
                    return actor_loss
                
                total_loss = _loss_function(
                    current_actor_params,
                    inputs,
                    expert_actions,
                )

                return jnp.mean(total_loss)

            inputs = {
                "actor_input": experiences["actor_inputs"],
                "scan_embedding": experiences["scan_embeddings"],
            }
            expert_actions = experiences["actor_actions"]
            # Compute the loss and gradients
            actor_loss, grads = value_and_grad(_batch_loss_function, has_aux=False)(
                current_actor_params, 
                inputs,
                expert_actions,
            )
            return actor_loss, grads
        def _compute_critic_loss_and_gradients(
            current_critic_params:dict,  
            experiences:dict,
        ) -> tuple:
            def _batch_loss_function(
                current_critic_params:dict,
                experiences:dict
                ) -> jnp.ndarray:
            
                predicted_values = self.batch_critic_forward(
                    random.split(critic_key, experiences["states"].shape[0]),
                    current_critic_params,
                    experiences["states"],
                    experiences["actions_history"],
                    {
                        "humans_goal": experiences["humans_goal"],
                        "humans_parameters": experiences["humans_parameters"],
                        "static_obstacles": experiences["static_obstacles"]
                    },
                    {
                        "robot_goal": experiences["robot_goal"],
                        "robot_radius": experiences["robot_radius"],
                        "v_max": experiences["v_max"],
                        "wheels_distance": experiences["wheels_distance"],
                        "wheels_max_linear_acceleration": experiences["wheels_max_linear_acceleration"],
                        "robot_delay": experiences["robot_delay"]
                    },
                    experiences["action_space_params"],
                )
                total_loss = jnp.square(predicted_values - experiences["returns"])
                pred_has_nan = jnp.isnan(predicted_values).any()
                lax.cond(
                    pred_has_nan,
                    lambda: debug.print("CRITICAL: NaNs trovati nelle PREDIZIONI del Critic!"),
                    lambda: None
                )
                return jnp.mean(total_loss)
            # Compute the loss and gradients
            critic_loss, grads = value_and_grad(_batch_loss_function, has_aux=False)(
                current_critic_params, 
                experiences
            )
            return critic_loss, grads
        ## ACTOR
        # Compute loss and gradients for actor and critic
        actor_loss, actor_grads = _compute_actor_loss_and_gradients(actor_params, experiences)
        # Compute parameter updates
        actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
        # Apply updates
        updated_actor_params = optax.apply_updates(actor_params, actor_updates)
        ## CRITIC
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads = _compute_critic_loss_and_gradients(critic_params, experiences)
        # Compute parameter updates
        critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
        # Apply updates
        updated_critic_params = optax.apply_updates(critic_params, critic_updates)
        return (
            updated_actor_params, 
            actor_opt_state, 
            actor_loss, 
            updated_critic_params,
            critic_opt_state,
            critic_loss,
        )