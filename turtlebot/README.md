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

# SYSTEM IDENTIFICATION ON TURTLEBOT4

The script `turtlebot_sysid.py` collects raw sensor and actuator data from the TurtleBot4 for offline analysis. It covers three independent acquisition modes, each targeting a specific source of sim-to-real gap.

---

## Mode 1 — LiDAR Noise & Dropout (`lidar`)

**Purpose:** Characterise per-ray Gaussian noise and correlated dropout of the LiDAR sensor.

**Setup:** Place the robot in a static environment with surfaces of varying reflectance (smooth wall, glass, dark fabric). Switch the motors off to eliminate vibration.

```bash
python3 turtlebot_sysid.py lidar -n 1000 -s sysid_lidar.pkl
```

| Flag                  | Default           | Description                                |
|-----------------------|-------------------|--------------------------------------------|
| `-n` / `--n_scans`    | `1000`            | Number of full-resolution scans to collect |
| `-s` / `--save_file`  | `sysid_lidar.pkl` | Output pickle filename                     |

**Saved data:** for each scan — full `ranges` array (native resolution, typically 1080 rays), `intensities`, scan geometry (`angle_min`, `angle_max`, `angle_increment`), and both ROS and wall-clock timestamps.

**Offline analysis:**

- Compute per-ray mean µᵢ and std σᵢ → fit linear noise model σ(d) ≈ a·d + b
- Analyse NaN/inf dropout sequences → estimate Markov chain transition probabilities p_start and p_continue

---

## Mode 2 — Velocity Step-Response (`velocity`)

**Purpose:** Identify the time constant τ of the low-level wheel velocity controller (first-order system).

**Setup:** Place the robot on the surface you want to characterise (smooth floor, carpet, etc.) with enough free space in front.

```bash
python3 turtlebot_sysid.py velocity -v 0.5 --hold 4.0 --rest 2.0 -n 5 -s sysid_velocity.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--v_target` | `0.5` m/s | Step target velocity |
| `--hold` | `4.0` s | Duration of each step at v_target |
| `--rest` | `2.0` s | Rest duration at 0 between steps |
| `-n` / `--n_steps` | `5` | Number of step repetitions |
| `-s` / `--save_file` | `sysid_velocity.pkl` | Output pickle filename |

**Saved data:** full odometry buffer (vx, vy, wz, px, py, yaw) sampled at ~100 Hz with phase and trial annotations, plus per-trial command timestamps.

**Offline analysis:** For each trial, extract the vx(t) response after the step onset and fit a first-order model to estimate τ. Average across trials to get τ_base and σ_τ.

---

## Mode 3 — Actuator Latency & Differential Drift (`latency`)

**Purpose:** Measure the communication dead-time T_delay between command publication and first encoder response, and quantify the lateral drift due to asymmetric motor efficiencies.

**Setup:** Ensure a clear straight corridor of at least `--dist` metres in front of the robot.

```bash
python3 turtlebot_sysid.py latency -v 0.5 -d 5.0 -n 10 -s sysid_latency.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--v_cruise` | `0.5` m/s | Velocity used for both sub-tests |
| `-d` / `--dist` | `5.0` m | Straight-line distance for the drift test |
| `-n` / `--n_trials` | `10` | Number of latency trials |
| `-s` / `--save_file` | `sysid_latency.pkl` | Output pickle filename |

The script runs two sub-tests **automatically in sequence**:

1. **Dead-time trials** (`n_trials` repetitions): issues a step command and detects the first odom sample where |vx| exceeds 0.02 m/s. T_delay is reported in milliseconds for each trial.
2. **Drift run**: resets odometry, then commands v = v_cruise, ω = 0 for `dist` metres. Records the full trajectory; lateral deviation at the end quantifies the asymmetric motor bias.

**Saved data:** high-rate odometry buffer (~200 Hz) with phase/trial annotations, per-trial latency estimates (wall and ROS timestamps of command and first response).

**Offline analysis:** Average T_delay across trials. Compute multiplicative and additive wheel noise components from the trajectory curvature during the drift run.

---

## General Notes

- All modes reset the odometry via `/reset_pose` at startup. Make sure the TurtleBot4 is connected and the service is available before running.
- Pressing **Ctrl+C** at any point saves the partial data collected so far.
- Output files are saved in the `turtlebot/` folder next to the script.
