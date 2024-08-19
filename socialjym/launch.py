from jax import random, debug
from envs.socialnav import SocialNav

env = SocialNav(robot_radius=0.5, robot_dt=0.1, humans_dt=0.1, scenario='circular_crossing', n_humans=5)
env_state, obs = env.reset(random.key(0))

# DEBUG
debug.print("\n")
debug.print("jax.debug.print(env_state[0]) -> {x}", x=env_state[0])
debug.print("jax.debug.print(obs) -> {x}", x=obs)