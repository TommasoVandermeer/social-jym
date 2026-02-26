# DEPLOYMENT ON TURTLEBOT4

WARNING: Always make sure to be connected to the same WiFi of the turtlebot before running the scripts. 

The action inference will be run on the remote device used to run the scripts and sent to the Turtlebot4 through a ROS2 interface.

## JESSI (JAX-based E2E Safe Social Interpretable navigation)

Instructions to deploy JESSI on the Turtlebot:

- Save your trained policy in the ```turtlebot``` folder.
- Turn on the Turtlebot4 and wait for it to become fully operative (all the LEDs over the display should be green).
- Launch the following script ```turtlebot_jessi_controller.py``` on your remote device using the following command on terminal. 
```
python3 jessi_controller.py -x REPLACE_WITH_GOAL_X -y REPLACE_WITH_GOAL_Y -n REPLACE_WITH_NETWORK_NAME --patrol -s REPLACE_WITH_EXPERIMENT_NAME
```
The ```x``` and ```y``` flags indicate the position of the goal in the robot frame (<b>positive x axis is on the front of the robot, positive y axis is on the left of the robot</b>). The trained network used for control will be the one indicated after the ```n``` flag (include .pkl at the end). The ```patrol``` flag is a boolean indicating whether the robot should go back and forth from its initial position to its goal (True, keep the flag), or if it should just reach its goal once and then stop (False, remove the flag). The trajectory data will be saved in the ```turtlebot``` folder under the name indicated after the ```s``` flag (include .pkl at the end).

To animate the recorded trajectory run:
```
python3 turtlebot_jessi_animate_recorded_trajectory.py -s REPLACE_WITH_EXPERIMENT_NAME
```