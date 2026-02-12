from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial

from socialjym.policies.dwa import DWA
from socialjym.envs.base_env import wrap_angle

class DWB(DWA):
    def __init__(
        self,
        # DWB Critics weights
        base_obstacle_weight=.02,
        goal_align_weight=24.0,
        goal_distance_weight=24.0,
        oscillation_weight=1.0,
        path_align_weight=32.0,
        path_distance_weight=32.0,
        #prefer_forward_weight,
        rotate_to_goal_weight=32.0,
        #twirling_weight,
        # DWB Critics other parameters
        scaling_factor_obstacle=10.0,
        goal_align_forward_point_distance=0.1,
        path_align_forward_point_distance=0.1,
        # DWA hyperparameters
        actions_discretization = 15,
        predict_time_horizon = 1.75,
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
        lidar_n_stack_to_use=1, # Number of lidar scans to use to compute the action. If 1, only the most recent scan is used. If >1, the most recent n_stack_to_use scans are used and stacked together (e.g. if n_stack_to_use=3 and lidar_num_rays=100, the input point cloud will have 300 points).
        use_box_action_space=False, # If True, the action space for which the DWA cost is computed is the box action space (i.e. all combinations of linear and angular speeds up to the maximum) but then the action is bounded in the triangle, otherwise it is the discretized action space (i.e. a subset of the box action space). Using the box action space is more computationally expensive, but allows to find better actions.
    ):
        """
        NOT COMPLETED YET! DO NOT USE!

        Dynamic Window Approach by David Lu! (DWB) policy for navigation. Equivalent to DWB in NAV2 ROS2.
        Implementation inspired by https://github.com/locusrobotics/robot_navigation/tree/master/dwb_local_planner
        """
        #TODO: FINISH IMPLEMENTATION OF THE CRITICS AND TEST THE POLICY
        # Initialize parent class
        super().__init__(
            actions_discretization = actions_discretization,
            predict_time_horizon = predict_time_horizon,
            robot_radius=robot_radius,
            v_max=v_max, 
            gamma=gamma, 
            dt=dt, 
            wheels_distance=wheels_distance, 
            n_stack=n_stack,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            lidar_angles_robot_frame=lidar_angles_robot_frame, 
            lidar_n_stack_to_use=lidar_n_stack_to_use, 
            use_box_action_space=use_box_action_space,
        )
        # Initialize DWB-specific attributes
        self.scaling_factor_obstacle = scaling_factor_obstacle
        self.goal_align_forward_point_distance = goal_align_forward_point_distance
        self.path_align_forward_point_distance = path_align_forward_point_distance
        # Default attributes
        self.name = "DWB"
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, point_cloud)
            # Refer to https://github.com/locusrobotics/robot_navigation/tree/master/dwb_critics
            'base_obstacle': lambda p, a, aidx, g, pc: self._base_obstacle_critic(p, aidx, pc), # BaseObstacleCritic
            'goal_align':   lambda p, a, aidx, g, pc: self._goal_align_critic(p, a, g), # GoalAlignCritic
            'goal_distance': lambda p, a, aidx, g, pc: self._goal_distance_critic(p, a, g), # GoalDistCritic
            'oscillation': lambda p, a, aidx, g, pc: self._oscillation_critic(p, a), # OscillationCritic
            'path_align': lambda p, a, aidx, g, pc: self._path_align_critic(p, a, g), # PathAlignCritic
            'path_distance': lambda p, a, aidx, g, pc: self._path_distance_critic(p, a, g), # PathDistCritic 
            #'prefer_forward': lambda p, a, aidx, g, pc: self._prefer_forward_critic(p, a), # PreferForwardCritic
            'rotate_to_goal': lambda p, a, aidx, g, pc: self._rotate_to_goal_critic(p, a, g), # RotateToGoalCritic
            #'twirling': lambda p, a, aidx, g, pc: self._twirling_critic(p, a), # TwirlingCritic
        }
        self.weights = {
            'base_obstacle': base_obstacle_weight,
            'goal_align': goal_align_weight,
            'goal_distance': goal_distance_weight,
            'oscillation': oscillation_weight,
            'path_align': path_align_weight,
            'path_distance': path_distance_weight,
            #'prefer_forward': prefer_forward_weight,
            'rotate_to_goal': rotate_to_goal_weight,
            #'twirling': twirling_weight,
        }
    
    @partial(jit, static_argnames=("self"))
    def _base_obstacle_critic(self, robot_pose, action_idx, point_cloud):
        # Predict robot trajectory for the given action
        c, s = jnp.cos(robot_pose[2]), jnp.sin(robot_pose[2])
        rot = jnp.array([[c, -s], [s, c]])
        final_robot_pose =  robot_pose[:2] + jnp.dot(self.delta_trajectories[action_idx,-1,:2], rot.T)
        # Compute distance from predicted trajectory to each point in the point cloud
        distances = jnp.linalg.norm(final_robot_pose - point_cloud, axis=1)
        min_distance = jnp.min(distances)
        clearance_cost = lax.cond(
            min_distance - self.robot_radius <= 0,
            lambda: jnp.inf,  # Collision, assign infinite cost
            lambda: 253 * jnp.exp(-self.scaling_factor_obstacle * (min_distance - self.robot_radius)),  # Prefer larger clearance (i.e. smaller cost)
        )
        return clearance_cost

    @partial(jit, static_argnames=("self"))
    def _goal_align_critic(self, robot_pose, action, robot_goal):
        pass

    @partial(jit, static_argnames=("self"))
    def _goal_distance_critic(self, robot_pose, action, robot_goal):
        pass

    @partial(jit, static_argnames=("self"))
    def _oscillation_critic(self, robot_pose, action):
        pass

    @partial(jit, static_argnames=("self"))
    def _path_align_critic(self, robot_pose, action, robot_goal):
        pass

    @partial(jit, static_argnames=("self"))
    def _path_distance_critic(self, robot_pose, action, robot_goal):
        pass

    @partial(jit, static_argnames=("self"))
    def _prefer_forward_critic(self, robot_pose, action):
        pass

    @partial(jit, static_argnames=("self"))
    def _rotate_to_goal_critic(self, robot_pose, action, robot_goal):
        pass

    @partial(jit, static_argnames=("self"))
    def _twirling_critic(self, robot_pose, action):
        pass