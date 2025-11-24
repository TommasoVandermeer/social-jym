import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap

# Import Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

# Import Environment e Utils
from wrapper_gym_lasernav import LaserNavGym
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
from socialjym.utils.rewards.lasernav_rewards.reward1_lasernav import Reward1LaserNav
from socialjym.utils.aux_functions import animate_trajectory

def main():
    # ==========================================
    # 1. CONFIGURAZIONE PARAMETRI
    # ==========================================
    env_params = {
        'n_stack': 5, 
        'lidar_num_rays': 108, 
        'lidar_angular_range': 2*jnp.pi,
        'lidar_max_dist': 10., 
        'n_humans': 7, 
        'n_obstacles': 5,
        'robot_radius': 0.3, 
        'robot_dt': 0.25, 
        'humans_dt': 0.01,
        'robot_visible': False, 
        'scenario': 'hybrid_scenario',
        'reward_function': Reward1LaserNav(
            robot_radius=0.3,
            goal_reward=10.0,      # Aumentato per incentivare il goal
            collision_penalty=-10.0, # Penalità forte
            discomfort_distance=0.5
        ),
        'kinematics': 'unicycle',
    }

    # ==========================================
    # 2. TRAINING PARALLELO (Multi-Process)
    # ==========================================
    # Numero di ambienti paralleli (es. 4 o 8 in base alla tua CPU)
    N_ENVS = 32
    TOTAL_TIMESTEPS = 1_000_000  

    print(f"Creazione di {N_ENVS} ambienti paralleli...")
    # Crea gli ambienti paralleli passando i parametri
    vec_env = make_vec_env(LaserNavGym, n_envs=N_ENVS, env_kwargs={'env_params': env_params})
    # Monitora i log (reward, lunghezza episodi)
    vec_env = VecMonitor(vec_env)

    print("Inizio training PPO...")
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda") # Usa device="cuda" se hai GPU configurata
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("Training completato. Salvataggio modello...")
    model.save("ppo_lasernav_parallel")
    
    # Chiudiamo gli ambienti paralleli per liberare risorse
    vec_env.close()

    # ==========================================
    # 3. TEST E VISUALIZZAZIONE (Single Env)
    # ==========================================
    print("\nAvvio test di visualizzazione su ambiente singolo...")
    
    # Creiamo un ambiente singolo pulito per il test
    eval_env = LaserNavGym(env_params)
    obs, info = eval_env.reset()
    
    # Inizializziamo le liste per salvare la storia (necessario per animate_trajectory)
    # Nota: eval_env._state è lo stato interno JAX salvato dal wrapper
    all_states = np.array([eval_env._state])
    all_observations = np.array([obs])

    # Loop di simulazione
    for i in range(60): # Massimo 60 step di test
        # Predici l'azione usando il modello addestrato (deterministic=True rimuove la casualità)
        action, _ = model.predict(obs, deterministic=True)
        
        # Esegui lo step
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # Salva stato e osservazione
        all_states = np.vstack((all_states, [eval_env._state]))
        all_observations = np.vstack((all_observations, [obs]))
        
        if terminated or truncated:
            print(f"Episodio terminato al passo {i+1}")
            break

    # ==========================================
    # 4. CALCOLI PER VISUALIZZAZIONE (Lidar)
    # ==========================================
    print("Preparazione dati per l'animazione...")
    
    # Estrae le yaw del robot dalla storia degli stati
    robot_yaws = all_states[:, -1, 4]
    
    # Funzione JAX vettorizzata per calcolare gli angoli dei raggi laser
    def get_angles(yaw):
        return jnp.linspace(
            yaw - env_params['lidar_angular_range']/2, 
            yaw + env_params['lidar_angular_range']/2, 
            env_params['lidar_num_rays']
        )
    angles = vmap(get_angles)(robot_yaws)

    # Estrae le distanze lidar dalle osservazioni (dal 6° indice in poi)
    lidar_dists = all_observations[:, 0, 6:] 
    
    # Combina distanze e angoli per il plotter: shape (T, 100, 2)
    lidar_measurements = vmap(lambda d, a: jnp.stack((d, a), axis=-1))(lidar_dists, angles)

    # ==========================================
    # 5. LANCIO ANIMAZIONE
    # ==========================================
    print("Apertura finestra grafica...")
    animate_trajectory(
        states=all_states,
        humans_radiuses=info['humans_parameters'][:, 0],
        robot_radius=env_params['robot_radius'],
        humans_policy='hsfm',
        robot_goal=info['robot_goal'],
        scenario=info['current_scenario'],
        static_obstacles=info['static_obstacles'][-1],
        robot_dt=env_params['robot_dt'],
        lidar_measurements=lidar_measurements,
        kinematics=env_params['kinematics'],
        figsize=(10, 10)
    )

if __name__ == "__main__":
    main()