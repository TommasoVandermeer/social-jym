# social-jym
An environment based on JAX to train mobile robots within crowded environments. Includes several human motion models, several RL algorithms for social navigation and implements fast training and computing thanks to JAX.

## Installation
Create a virtual environment.
```
python3 -m venv socialjym
```
Activate the virtual environment.
```
source socialjym/bin/activate
```
First, configure Git to swap the protocols.
```
git config --global url."https://github.com/".insteadOf "git@github.com:"
```
Now, you can clone the repository and its submodules.
```
git clone --recurse-submodules https://github.com/otr-ebla/albe-social-jym.git
```
Install the submodules and the main package.
```
pip install -e .
```

## Project structure
The source code of the project can be found in the folder socialjym which includes all the python modules. It is structured as follows:
```bash
├── socialjym
│   ├── __init__.py
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
│   │   │   ├── lasernav_rewards
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dummy_reward.py
│   │   ├── rollouts
│   │   │   ├── __init__.py
│   │   │   ├── act_cri_rollouts.py
│   │   │   ├── ppo_rollouts.py
│   │   │   ├── vnet_rollouts.py
│   │   ├── terminations
│   │   │   ├── __init__.py
│   │   │   ├── base_termination.py
│   │   │   ├── robot_human_collision.py
│   │   │   ├── robot_obstacle_collision.py
│   │   │   ├── robot_reached_goal.py
│   │   │   ├── timeout.py
```
### Envs ([envs readme](socialjym/envs/README.md))
Includes all the available Reinforcement Learning environments developed in an open AI gymnasium style (step, reset, _get_obs, ecc..) but with functional programming (required for JAX). BaseEnv serves as a base class defining the methods and attributes each environment should have. In BaseEnv also the available scenarios, human motion models and robot kinematic models are listed. [SocialNav](socialjym/envs/README.md) and [LaserEnv](socialjym/envs/README.md) are complete environments that can be used to train and test RL policies for navigation (click on the links to see more). 

<b>Human motion models</b>
- ORCA (Optimal Reciprocal Collision Avoidance) [[1]](#orca).
- SFM (Social Force Model) [[2]](#sfm).
- HSFM (Headed Social Force Model)  [[3]](#hsfm).

<b>Scenarios</b> [[4]](#soc-guidelines)
- CC (Circular Crossing): <br>
<img src=".media/cc.gif" alt="Circular Crossing" width="350"/>
- PaT (Parallel Traffic): <br>
<img src=".media/pat.gif" alt="Parallel Traffic" width="350"/>
- PeT (Perpendicular Traffic): <br>
<img src=".media/pet.gif" alt="Perpendicular Traffic" width="350"/>
- RC (Robot Crowding): <br>
<img src=".media/rc.gif" alt="Robot Crowding" width="350"/>
- DCC (Delayed Circular Crossing): <br>
<img src=".media/dcc.gif" alt="Delayed Circular Crossing" width="350"/>
- CCSO (Circular Crossing with Static Obstacles): <br>
<img src=".media/ccso.gif" alt="Circular Crossing with Static Obstacles" width="350"/>
- CN (Crowd Navigation): <br>
<img src=".media/cn.gif" alt="Crowd Navigation" width="350"/>
- CT (Corner Traffic): <br>
<img src=".media/ct.gif" alt="Corner Traffic" width="350"/>

<b>Robot kinematic models</b>
- Holonomic
- Differential Drive

### Policies
Includes all the available Reinforcement Learning policies that can be used in the environments. BasePolicy serves as a base class defining the abstract methods each policy should have. Here is a comprehensive list:
- CADRL [[5]](#cadrl).
- SARL [[6]](#sarl).
- SARL* [[7]](#sarl-star).
- SARL-PPO: an actor-critic version of SARL, trained with PPO.
- DIR-SAFE: discover more on the <a href="https://github.com/TommasoVandermeer/social-jym/tree/dir-safe">dedicated branch</a>.

### Utils
Includes all the necessary implementations to train RL-based policies:
- <b>Cell decompositions</b>: contain various method to discretize the 2D space for global planning.
- <b>Distributions</b>: implementations of various stochastic distributions (used to parameterize the action space in continuos control for actor-critic policies).
- <b>Global planners</b>: contain various algorithms for the shortest path problem.
- <b>Replay buffers</b>: implementations of different buffers for the offline RL.
- <b>Rewards</b>: contain the reward functions for each environment.
- <b>Rollouts</b>: implement the learning cycle for each method (Actor-Critic and Value Network Learning).
- <b>Terminations</b>: implementation of termination conditions for RL episodes.

## Get started
The notebooks folder contains examples on how to use this package with a step-by-step guide. Get started by looking at the example in there.

You might find other useful examples in the tests folder. However, be aware that the scripts in this folder are not maintained (they might not be updated to latest features of the package. Hence, they might not work).

## References
<ul>
    <li id="orca">[1] Van Den Berg, J., Snape, J., Guy, S. J., & Manocha, D. (2011, May). Reciprocal collision avoidance with acceleration-velocity obstacles. In 2011 IEEE International Conference on Robotics and Automation (pp. 3475-3482). IEEE.</li>
    <li id="sfm">[2] Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic. Nature, 407(6803), 487-490.</li>
    <li id="hsfm">[3] Farina, F., Fontanelli, D., Garulli, A., Giannitrapani, A., & Prattichizzo, D. (2017). Walking ahead: The headed social force model. PloS one, 12(1), e0169734.</li>
    <li id="soc-guidelines">[4] Francis, A., Pérez-d’Arpino, C., Li, C., Xia, F., Alahi, A., Alami, R., ... & Martín-Martín, R. (2025). Principles and guidelines for evaluating social robot navigation algorithms. ACM Transactions on Human-Robot Interaction, 14(2), 1-65.</li>
    <li id="cadrl">[5] Chen, Y. F., Liu, M., Everett, M., & How, J. P. (2017, May). Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA) (pp. 285-292). IEEE.</li>
    <li id="sarl">[6] Chen, C., Liu, Y., Kreiss, S., & Alahi, A. (2019, May). Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning. In 2019 international conference on robotics and automation (ICRA) (pp. 6015-6022). IEEE.</li>
    <li id="sarl-star">[7] Li, K., Xu, Y., Wang, J., & Meng, M. Q. H. (2019, December). SARL: Deep reinforcement learning based human-aware navigation for mobile robot in indoor environments. In 2019 IEEE International Conference on Robotics and Biomimetics (ROBIO) (pp. 688-694). IEEE.</li>
</ul>
