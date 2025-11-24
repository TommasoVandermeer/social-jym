import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from socialjym.envs.lasernav import LaserNav

class LaserNavGym(gym.Env):
    def __init__(self, env_params, seed=0):
        self.env = LaserNav(**env_params)
        self.key = jax.random.PRNGKey(seed)
        self._state = None
        self._info = None
        self.max_dist = env_params['lidar_max_dist']
        
        # Definisce lo spazio delle azioni: [v, w]
        self.action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Definisce lo spazio delle osservazioni: (n_stack, features)
        obs_dim = env_params['lidar_num_rays'] + 6 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_params['n_stack'], obs_dim), dtype=np.float32)

    def _sanitize_obs(self, obs):
        """Sostituisce inf e nan con valori sicuri"""
        obs = np.array(obs)
        # Sostituisci infiniti con max_dist (o poco più)
        obs[obs == np.inf] = self.max_dist
        obs[obs == -np.inf] = 0.0
        # Sostituisci NaN con 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.max_dist, neginf=0.0)
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None: self.key = jax.random.PRNGKey(seed)
        self.key, reset_key = jax.random.split(self.key)
        
        self._state, _, obs, self._info, _ = self.env.reset(reset_key)
        
        # Sanifica l'osservazione
        clean_obs = self._sanitize_obs(obs)
        
        return clean_obs, self._info.copy()

    def step(self, action):
        # Clip dell'azione per sicurezza (evita NaN se la rete impazzisce)
        action = np.clip(action, -1.0, 1.0)
        
        self._state, obs, self._info, reward, outcome, _ = self.env.step(self._state, self._info, jnp.array(action))
        
        terminated = bool(outcome['success'] | outcome['collision_with_human'] | outcome['collision_with_obstacle'])
        truncated = bool(outcome['timeout'])
        
        # Sanifica l'osservazione
        clean_obs = self._sanitize_obs(obs)
        
        # Sanifica anche il reward (non si sa mai)
        reward = 0.0 if np.isnan(reward) else float(reward)

        return clean_obs, reward, terminated, truncated, self._info.copy()