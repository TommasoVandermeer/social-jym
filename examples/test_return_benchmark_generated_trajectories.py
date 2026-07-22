import os
import pickle

from socialjym.utils.aux_functions import animate_trajectory
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rewards.lasernav_rewards.reward3 import Reward3

trajectory_1_name = "active_avoidance"  
trajectory_2_name = "last_moment_avoidance"  
gamma = 0.9

dir = os.path.join(os.path.dirname(__file__), f"{trajectory_1_name}.pkl")
with open(dir, "rb") as f:
    trajectory_1_data = pickle.load(f)
   
# animate_trajectory(
#     trajectory_1_data["all_states"],
#     trajectory_1_data["humans_parameters"][:,0],
#     trajectory_1_data["robot_radius"],
#     'hsfm',
#     trajectory_1_data["robot_goal"],
#     -1,
#     static_obstacles=trajectory_1_data["static_obstacles"][-1],
#     kinematics='unicycle',
# )

dir = os.path.join(os.path.dirname(__file__), f"{trajectory_2_name}.pkl")
with open(dir, "rb") as f:
    trajectory_2_data = pickle.load(f)

# animate_trajectory(
#     trajectory_2_data["all_states"],
#     trajectory_2_data["humans_parameters"][:,0],
#     trajectory_2_data["robot_radius"],
#     'hsfm',
#     trajectory_2_data["robot_goal"],
#     -1,
#     static_obstacles=trajectory_2_data["static_obstacles"][-1],
#     kinematics='unicycle',
# )

# reward_function = Reward1(
#     robot_radius=0.3,
#     collision_with_humans_penalty=-.5,
#     progress_to_goal_reward=True,
#     discomfort_penalty_reward=True,
#     high_rotation_penalty_reward=True,
# )

reward_function = Reward3(
    robot_radius=0.3,
    collision_with_humans_penalty=-.5,
    progress_to_goal_reward=True,
    discomfort_penalty_reward=True,
    high_rotation_penalty_reward=True,
    discomfort_k_front=2.,
    discomfort_weight=0.05,
)

return_1 = 0.0
dt = trajectory_1_data["robot_dt"]
for t, state in enumerate(trajectory_1_data["all_states"][:-1]):
    info = {}
    info['time'] = t * dt
    info['humans_parameters'] = trajectory_1_data["humans_parameters"]
    info['static_obstacles'] = trajectory_1_data["static_obstacles"]
    info['robot_goal'] = trajectory_1_data["robot_goal"]
    action = trajectory_1_data["all_actions"][t]
    reward, _, _ = reward_function(state, action, info, dt)
    return_1 += gamma**(info['time']) * reward
print(f"Return of trajectory 1 ({trajectory_1_name}): {return_1:.4f}")

return_2 = 0.0
dt = trajectory_2_data["robot_dt"]
for t, state in enumerate(trajectory_2_data["all_states"][:-1]):
    info = {}
    info['time'] = t * dt
    info['humans_parameters'] = trajectory_2_data["humans_parameters"]
    info['static_obstacles'] = trajectory_2_data["static_obstacles"]
    info['robot_goal'] = trajectory_2_data["robot_goal"]
    action = trajectory_2_data["all_actions"][t]
    reward, _, _ = reward_function(state, action, info, dt)
    return_2 += gamma**(info['time']) * reward
print(f"Return of trajectory 2 ({trajectory_2_name}): {return_2:.4f}")