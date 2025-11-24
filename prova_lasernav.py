import jax.numpy as jnp
from jax import random, vmap
import numpy as np
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import animate_trajectory

def main():
    env_params = {
        'n_stack': 5, 'lidar_num_rays': 100, 'lidar_angular_range': 2*jnp.pi,
        'lidar_max_dist': 10., 'n_humans': 7, 'n_obstacles': 5,
        'robot_radius': 0.3, 'robot_dt': 0.25, 'humans_dt': 0.01,
        'robot_visible': False, 'scenario': 'hybrid_scenario',
        'reward_function': DummyReward(robot_radius=0.3), 'kinematics': 'unicycle',
    }

    print("Inizializzazione ambiente...")
    env= LaserNav(**env_params)

    key = random.PRNGKey(0)
    reset_key, subkey = random.split(key) 
    state, reset_key, obs, info, outcome = env.reset(reset_key)

    all_states = np.array([state]) # inizializza array numpy con stato iniziale
    all_observations = np.array([obs]) # inizializza array numpy con osservazione iniziale  

    print("inizio simulazione 50 passi...")
    for i in range(50):
        action = jnp.array([1.0, 0.5])  # azione fissa: avanti

        state, obs, info, reward, outcome, _ = env.step(state, info, action, test=True) 

        all_states = np.vstack((all_states, [state])) # aggiungi nuovo stato
        all_observations = np.vstack((all_observations, [obs])) # aggiungi nuova osservazione

        if not outcome['nothing']:
            print(f"Episode ended at step {i+1} with outcome: {outcome}")
            break

    print("Simulazione terminata. Inizio animazione...")

    robot_yaws = all_states[:, -1, 4]
    def get_angles(yaw):
        return jnp.linspace(
            yaw - env.lidar_angular_range/2,
            yaw + env.lidar_angular_range/2,
            env.lidar_num_rays
        )
    angles = vmap(get_angles)(robot_yaws)
    
    lidar_dists = all_observations[:, 0, 6:]
    lidar_measurements = vmap(lambda d, a: jnp.stack((d, a), axis=-1))(lidar_dists, angles)

    animate_trajectory(
        states=all_states,
        humans_radiuses=info['humans_parameters'][:, 0],
        robot_radius=env.robot_radius,
        humans_policy='hsfm', # Policy simulata degli umani
        robot_goal=info['robot_goal'],
        scenario=info['current_scenario'],
        static_obstacles=info['static_obstacles'][-1], # Ultimo set di ostacoli
        robot_dt=env.robot_dt,
        lidar_measurements=lidar_measurements,
        kinematics=env.kinematics,
        figsize=(10, 10) # Dimensione finestra
    )

if __name__ == "__main__":
    main()