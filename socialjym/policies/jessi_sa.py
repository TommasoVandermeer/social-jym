import jax.numpy as jnp
from jax import nn, random, vmap, jit
from jax.tree_util import tree_map
from functools import partial
import haiku as hk
from typing import Optional

from socialjym.policies.jessi import JESSI, E2E, MultiHeadAttention
from socialjym.utils.distributions.dirichlet import Dirichlet
from socialjym.utils.distributions.gaussian import Gaussian
from socialjym.utils.distributions.logistic_normal import LogisticNormal

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
            ablation_mode: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        self.n_detectable_humans = n_detectable_humans
        self.wheels_distance = wheels_distance
        self.vmax = v_max
        self.wmax = 2 * v_max / wheels_distance
        self.wmin = -2 * v_max / wheels_distance
        self.initial_concentration = initial_concentration
        self.ablation_mode = ablation_mode
        # Dimensions
        self.n_sectors = n_sectors
        if self.ablation_mode == 4: self.n_outputs = 2 # Gaussian (2 means, 1 per action)
        elif self.ablation_mode == 6: self.n_outputs = 3 # Logistic-Normal (3 means, 1 per vertex)
        else: self.n_outputs = 3 # Dirichlet (3 alphas, 1 per vertex)
        self.mlp_params = mlp_params
        # Scan embedding reducer
        self.scan_reducer = hk.Linear(1, name="scan_reducer")
        # 2. Self Attention Mechanism
        add_size = 22 if self.ablation_mode != 2 else 16 # Ablation: No human uncertainty params, so input size is smaller
        self.attention = MultiHeadAttention(
            num_heads=2,
            key_size=(n_sectors + add_size)//2,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            name="hcg_self_attention"
        )
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.att_ffn = hk.nets.MLP([n_sectors + add_size], activation=nn.gelu, activate_final=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # 2.5 Simple MLP (in place of scene-attention for ablation)
        if self.ablation_mode == 3: # Ablation: No scene attention, simple MLP instead
            self.simple_mlp = hk.nets.MLP(
                **mlp_params,
                output_sizes=[200, 100, n_sectors + 20], 
                name="simple_mlp"
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
        if ablation_mode == 4: self.action_distribution = Gaussian()
        elif ablation_mode == 6: self.action_distribution = LogisticNormal()
        else: self.action_distribution = Dirichlet()

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
            state_values: State value estimates from the critic. Shape (,) or (batch_size,)
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
        if self.ablation_mode == 2:
            x = x[:,:,[0,1,5,6,10,11,12,13,14,15,16,17,18,19,20,21]] # Keep only mean positions, mean velocities and weights            
        ### CONTEXT EXTRACTION
        scalar_scan = self.scan_reducer(y)  # (Batch, n_sectors, 1)
        y = jnp.squeeze(scalar_scan, axis=-1) # (Batch, n_sectors)
        # SCENE-ATTENTION MECHANISM
        if self.ablation_mode == 3: # Ablation: No scene attention, simple MLP instead
            humans_state = x[..., :11] # (Batch, N, 11)
            robot_state = x[:, 0, 11:22] # (Batch, 11)
            mlp_input = jnp.concatenate([humans_state.reshape(batch_size, -1), robot_state, y], axis=-1) # (Batch, (N+1)*11 + n_sectors)
            pooled_hcg_context = self.simple_mlp(mlp_input) # (Batch, 22 + n_sectors)
            human_attention = jnp.full((batch_size, self.n_detectable_humans), jnp.nan) # No attention in this ablation
        else:
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
        if self.ablation_mode == 4:
            ## Compute Gaussian distribution parameters
            scaled_vmax = action_space_params[:, 0] * self.vmax
            scaled_wmax = action_space_params[:, 1] * self.wmax
            scaled_wmin = action_space_params[:, 2] * self.wmin
            actor_out = self.actor_head(context) # [mu_v, mu_w]
            raw_mu_v = actor_out[..., 0]
            raw_mu_w = actor_out[..., 1]
            mu_v = nn.sigmoid(raw_mu_v) * scaled_vmax
            w_scale = (scaled_wmax - scaled_wmin) / 2.0
            w_shift = (scaled_wmax + scaled_wmin) / 2.0
            mu_w = nn.tanh(raw_mu_w) * w_scale + w_shift
            means = jnp.stack([mu_v, mu_w], axis=-1)  # Shape: (batch_size, 2)
            raw_logsigmas_param = hk.get_parameter("raw_logsigmas", shape=[2], init=hk.initializers.Constant(jnp.arctanh(9/11)))
            logsigmas_bounded = jnp.tanh(raw_logsigmas_param) * 11 - 9 # Bound logsigmas between [-20,2]
            logsigmas = jnp.broadcast_to(logsigmas_bounded, means.shape)
            distributions["means"] = means
            distributions["logsigmas"] = logsigmas
            # Dummy
            concentration = jnp.zeros((batch_size,))
        elif self.ablation_mode == 6:
            locs = self.actor_head(context)
            raw_logscales_param = hk.get_parameter("raw_logscales", shape=[3], init=hk.initializers.Constant(jnp.arctanh(9/11)))
            logscales_bounded = jnp.tanh(raw_logscales_param) * 11 - 9 # Bound logscales between [-20,2]
            logscales = jnp.broadcast_to(logscales_bounded, locs.shape)
            distributions["locs"] = locs
            distributions["log_scales"] = logscales
            # Dummy
            concentration = jnp.zeros((batch_size,))
        else:
            ## Compute Dirichlet distribution parameters
            alphas = nn.softplus(self.actor_head(context)) + 1  # (Batch, 3)
            concentration = jnp.sum(alphas, axis=-1)  # (Batch,)
            distributions["alphas"] = alphas
        ## Sample action
        sampled_actions = vmap(self.action_distribution.sample)(distributions, keys)
        ### CRITIC
        state_values = self.critic_head(context) # (Batch, 1)
        state_values = jnp.squeeze(state_values, axis=-1) # (Batch,)
        if not has_batch:
            sampled_actions = sampled_actions[0]
            state_values = state_values[0]
            distributions = tree_map(lambda t: t[0], distributions)
        return sampled_actions, distributions, concentration, state_values, human_attention

class E2E(E2E):
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
        ablation_mode: Optional[int] = None,
        legit: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            perception_name=perception_name,
            controller_name=controller_name,
            lidar_angles_robot_frame=lidar_angles_robot_frame,
            n_detectable_humans=n_detectable_humans,
            max_humans_velocity=max_humans_velocity,
            max_lidar_distance=max_lidar_distance,
            v_max=v_max,
            wheels_distance=wheels_distance,
            embed_dim=embed_dim,
            n_sectors=n_sectors,
            mlp_params=mlp_params,
            initial_concentration=initial_concentration,
            beam_dropout_rate=beam_dropout_rate,
            ablation_mode=ablation_mode,
            legit=legit,
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
            ablation_mode=ablation_mode,
        )

class JESSI_SA(JESSI):
    """
    A state augmented version of JESSI designed to operate on systems with delayed actuation.
    """
    def __init__(
        self, 
        robot_radius:float=0.3,
        v_max:float=1., 
        dt:float=0.25, 
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
        ablation_mode: Optional[int] = None, # Options: 1 (no bounding), 2 (no humans uncertainty), 3 (no scene attention)
    ) -> None:
        assert n_actions_history <= n_stack, "The length of the actions history must be greater than the length of the observation stack"
        self.n_actions_history = n_actions_history
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
            ablation_mode=ablation_mode,
            legit=True, 
        )
        # Initialize Actor Critic network
        self.actor_critic_name = "actor_network"
        @hk.transform
        def actor_critic_network(x, y, **kwargs) -> jnp.ndarray:
            actor_critic = ActorCritic(
                self.actor_critic_name, 
                self.n_detectable_humans, 
                self.v_max, 
                self.wheels_distance, 
                n_sectors=self.n_sectors,
                ablation_mode=self.ablation_mode,
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
        # E2E input is shaped (self.n_stack, self.lidar_num_rays, 7) and (self.n_detectable_humans, 11 + 2*n_actions_history)
        perception_params = self.perception.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 7))) # Cardinality invariant for n_stack and lidar_num_rays
        actor_critic_params = self.actor_critic.init(key, jnp.zeros((self.n_detectable_humans, 22+2*self.n_actions_history)), jnp.zeros((self.n_sectors, self.embedding_dim)))
        e2e_params = self.e2e.init(key, jnp.zeros((self.n_stack, self.lidar_num_rays, 7)), jnp.zeros((self.n_detectable_humans, 11+2*self.n_actions_history))) # Cardinality invariant for n_stack and lidar_num_rays
        return perception_params, actor_critic_params, e2e_params

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