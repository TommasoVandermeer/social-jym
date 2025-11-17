# social-jym
An environment based on JAX to train mobile robots within crowded environments. Includes several human motion models, several RL algorithms for social navigation and implements fast training and computing thanks to JAX.

## Installation
Create a virtual environment.
```
virtualenv socialjym
```
Activate the virtual environment.
```
source socialjym/bin/activate
```
Clone the repository and its submodules.
```
git clone --recurse-submodules https://github.com/TommasoVandermeer/social-jym.git
```
Install the submodules and the main package.
```
pip install -e social-jym social-jym/JHSFM social-jym/JSFM social-jym/JORCA
```

## Project structure
The source code of the project can be found in the folder socialjym which includes all the python modules. It is structured as follows:
```bash
├── socialjym
│   ├── envs
│   │   ├── __init__.py
│   │   ├── base_env.py
│   │   ├── lasernav.py
│   │   ├── socialnav.py
│   ├── policies
│   │   ├── __init__.py
│   │   ├── base_policy.py
│   │   ├── cadrl.py
│   │   ├── dir_safe.py
│   │   ├── sarl_ppo.py
│   │   ├── sarl_star.py
│   │   ├── sarl.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── aux_functions.py
│   │   ├── cell_decompositions
│   │   │   ├── __init__.py
│   │   │   ├── grid.py
│   │   │   ├── quadtree.py
│   │   │   ├── utils.py
│   │   ├── distributions
│   │   │   ├── __init__.py
│   │   │   ├── base_distribution.py
│   │   │   ├── gaussian_mixture_model.py
│   │   │   ├── gaussian.py
│   │   ├── global_planners
│   │   │   ├── __init__.py
│   │   │   ├── base_global_planner.py
│   │   │   ├── a_star.py
│   │   │   ├── dijkstra.py
│   │   ├── replay_buffers
│   │   │   ├── __init__.py
│   │   │   ├── base_act_cri_buffer.py
│   │   │   ├── base_vnet_replay_buffer.py
│   │   │   ├── ppo_replay_buffer.py
│   │   ├── rewards
│   │   │   ├── __init__.py
│   │   │   ├── base_reward.py
│   │   │   ├── socialnav_rewards
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dummy_reward.py
│   │   │   │   ├── reward1.py
│   │   │   │   ├── reward2.py
│   │   ├── rollouts
│   │   │   ├── __init__.py
│   │   │   ├── act_cri_rollouts.py
│   │   │   ├── ppo_rollouts.py
│   │   │   ├── vnet_rollouts.py
│   ├── __init__.py
```
### Envs
Includes all the available Reinforcement Learning environments developed in an open AI gymnasium style (step, reset, _get_obs, ecc..). BaseEnv serves as a base class defining the methods and attributes each environment should have. In BaseEnv also the available scenarios are listed. [SocialNav](socialjym/envs/README.md) and [LaserEnv](socialjym/envs/README.md) are complete environments that can be used to train and test RL policies for navigation (click on the links to see more).

### Policies
Includes all the available Reinforcement Learning policies that can be used in the environments. BasePolicy serves as a base class defining the abstract methods each policy should have. Here is a comprehensive list:
- CADRL [[1]](#cadrl).
- SARL [[2]](#sarl).
- SARL* [[3]](#sarl-star).
- SARL-PPO: an actor-critic version of SARL, trained with PPO.
- DIR-SAFE: discover more on the <a href="https://github.com/TommasoVandermeer/social-jym/tree/dir-safe">dedicated branch</a>.

### Utils


## Get started


## References
<ul>
    <li id="cadrl">[1] Chen, Y. F., Liu, M., Everett, M., & How, J. P. (2017, May). Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA) (pp. 285-292). IEEE.</li>
    <li id="sarl">[2] Chen, C., Liu, Y., Kreiss, S., & Alahi, A. (2019, May). Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning. In 2019 international conference on robotics and automation (ICRA) (pp. 6015-6022). IEEE.</li>
    <li id="sarl-star">[3] Li, K., Xu, Y., Wang, J., & Meng, M. Q. H. (2019, December). SARL: Deep reinforcement learning based human-aware navigation for mobile robot in indoor environments. In 2019 IEEE International Conference on Robotics and Biomimetics (ROBIO) (pp. 688-694). IEEE.</li>
</ul>