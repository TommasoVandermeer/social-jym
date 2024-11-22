from socialjym.utils.rewards.socialnav_rewards.reward2 import RewardAblation

# Define reward parameters
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': 'unicycle',
    'target_reached_reward': True,
    'collision_penalty_reward': True,
    'discomfort_penalty_reward': True,
    'progress_to_goal_reward': True,
    'time_penalty_reward': True,
    'high_rotation_penalty_reward': True,
    'heading_deviation_from_goal_penalty_reward': True,
    'pass_right_penalty_reward': True,
}
reward_function = RewardAblation(**reward_params)
print(reward_function.type)