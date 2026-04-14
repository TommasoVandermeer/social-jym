# DEPLOYMENT ON TURTLEBOT4

WARNING: Always make sure to be connected to the same WiFi of the turtlebot before running the scripts. 

The action inference will be run on the remote device used to run the scripts and sent to the Turtlebot4 through a ROS2 interface.

## JESSI (JAX-based E2E Safe Social Interpretable navigation)

Instructions to deploy JESSI on the Turtlebot:

- Save your trained policy in the ```turtlebot``` folder.
- Turn on the Turtlebot4 and wait for it to become fully operative (all the LEDs over the display should be green).
- Launch the following script ```turtlebot_jessi_controller.py``` on your remote device using the following command on terminal. 
```
python3 turtlebot_jessi_controller.py -x REPLACE_WITH_GOAL_X -y REPLACE_WITH_GOAL_Y -n REPLACE_WITH_NETWORK_NAME --patrol --interp -s REPLACE_WITH_EXPERIMENT_NAME
```
The ```x``` and ```y``` flags indicate the position of the goal in the robot frame (<b>positive x axis is on the front of the robot, positive y axis is on the left of the robot</b>). The trained network used for control will be the one indicated after the ```n``` flag (include .pkl at the end). The ```patrol``` flag is a boolean indicating whether the robot should go back and forth from its initial position to its goal (True, keep the flag), or if it should just reach its goal once and then stop (False, remove the flag). The ```interp``` flag is a boolean indicating whether the robot pose should be interpolated to match exactly the LiDAR timestamp (True, keep the flag), or if the latest available pose at inference should be used (False, remove the flag). The trajectory data will be saved in the ```turtlebot``` folder under the name indicated after the ```s``` flag (include .pkl at the end).

Note that, to sync the timestamps of each topic (for debugging purposes), it is necessary to run ```sudo chronyc makestep``` on the turtlebot raspberrypi (connect with ssh).

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
python3 turtlebot_sysid.py velocity -v 0.3 --hold 4.0 --rest 2.0 -n 5 -s sysid_velocity.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--v_target` | `0.3` m/s | Step target velocity |
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
python3 turtlebot_sysid.py latency -v 0.3 -d 5.0 -n 10 -s sysid_latency.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--v_cruise` | `0.3` m/s | Velocity used for both sub-tests |
| `-d` / `--dist` | `5.0` m | Straight-line distance for the drift test |
| `-n` / `--n_trials` | `10` | Number of latency trials |
| `-s` / `--save_file` | `sysid_latency.pkl` | Output pickle filename |

The script runs two sub-tests **automatically in sequence**:

1. **Dead-time trials** (`n_trials` repetitions): issues a step command and detects the first odom sample where |vx| exceeds 0.02 m/s. T_delay is reported in milliseconds for each trial.
2. **Drift run**: resets odometry, then commands v = v_cruise, ω = 0 for `dist` metres. Records the full trajectory; lateral deviation at the end quantifies the asymmetric motor bias.

**Saved data:** high-rate odometry buffer (~200 Hz) with phase/trial annotations, per-trial latency estimates (wall and ROS timestamps of command and first response).

**Offline analysis:** Average T_delay across trials. Compute multiplicative and additive wheel noise components from the trajectory curvature during the drift run.

---

## Mode 4 — Staircase Velocity Tracking (`staircase`)

**Purpose:** Evaluate the low-level controller tracking accuracy across multiple linear and/or angular velocity levels, in both acceleration and deceleration directions.

**Setup:** Same as Mode 2 — flat surface. For angular segments, clear space around the robot.

Each `--seg` argument defines one ramp segment executed in order. Two formats are supported:

| Segment format | Effect |
| --- | --- |
| `lin:v_start:v_end:v_step:hold_s` | Linear velocity ramp, ω = 0 |
| `ang:w_start:w_end:w_step:hold_s` | Angular velocity ramp, v = 0 |
| `v_start:v_end:v_step:hold_s` | Legacy shorthand for `lin:...` |

`step` is always positive; direction is inferred from `start` vs `end`. Duplicate `(v, ω)` pairs at segment boundaries are automatically merged.

```bash
# Linear staircase up then down:
python3 turtlebot_sysid.py staircase \
    --seg lin:0:0.3:0.05:0.5 \
    --seg lin:0.3:0:0.1:0.5 \
    -s sysid_staircase.pkl

# Angular staircase up then down:
python3 turtlebot_sysid.py staircase \
    --seg ang:0:0.5:0.1:0.5 \
    --seg ang:0.5:0:0.1:0.5 \
    -s sysid_staircase_ang.pkl

# Linear followed by angular in one run:
python3 turtlebot_sysid.py staircase \
    --seg lin:0:0.3:0.05:0.5 --seg lin:0.3:0:0.1:0.5 \
    --seg ang:0:0.5:0.1:0.5  --seg ang:0.5:0:0.1:0.5 \
    -s sysid_staircase_full.pkl
```

| Flag | Description |
| --- | --- |
| `--seg [lin\|ang:]start:end:step:hold_s` | One ramp segment (repeatable, executed in order) |
| `-s` / `--save_file` | Output pickle filename (default: `sysid_staircase.pkl`) |

**Saved data:** full odometry buffer with `v_cmd`, `w_cmd`, and `step_idx` annotations per sample, plus the full commanded `sequence` list.

**Offline analysis:** Compare `v_cmd` / `ω_cmd` to measured `vx` / `ωz` to evaluate steady-state error and transient response at each level.

---

## Data Visualisation — `turtlebot_sysid_plot.py`

The companion script `turtlebot_sysid_plot.py` loads any pickle file produced by `turtlebot_sysid.py` and generates mode-specific figures and animations. It auto-detects the acquisition mode from the `mode` field inside the pickle.

```bash
python3 turtlebot_sysid_plot.py <pickle_file> [--save]
```

Pass `--save` to write all outputs to files instead of opening interactive windows. Files are saved in the current working directory.

---

### LiDAR pickle (`sysid_lidar.pkl`)

```bash
python3 turtlebot_sysid_plot.py sysid_lidar.pkl [--save]
```

| Output | Description |
| --- | --- |
| `lidar_stats.png` | 4-panel per-ray statistics: mean ± std, standard deviation, NaN/Inf dropout rate, Markov dropout probabilities p_start and p_continue per ray |
| `lidar_range_image.png` | Range-time heatmap (scan index × beam angle) |
| `lidar_animation.gif` | Animated polar scan with the static per-ray mean in the background (subsampled to ≤ 100 frames) |

---

### Velocity pickle (`sysid_velocity.pkl`)

```bash
python3 turtlebot_sysid_plot.py sysid_velocity.pkl [--save]
```

| Output | Description |
| --- | --- |
| `velocity_timeline.png` | Full-session timeline: reference command vs measured vx, lateral velocity and yaw rate, odometry position |
| `velocity_overlay.png` | All step responses overlaid and aligned at command time |
| `velocity_fit.png` | Per-step first-order fit `vx(t) = v_peak·(1 − e^(−t/τ))` on the rising portion, with estimated τ annotated |

---

### Latency pickle (`sysid_latency.pkl`)

```bash
python3 turtlebot_sysid_plot.py sysid_latency.pkl [--save]
```

| Output | Description |
| --- | --- |
| `latency_trials.png` | Per-trial velocity traces around each step command with the detected delay marker |
| `latency_summary.png` | Bar chart and boxplot of detected delays across all trials (mean and std annotated) |
| `latency_drift.png` | XY trajectory vs ideal straight line, and lateral deviation as a function of distance travelled |

---

### Staircase pickle (`sysid_staircase.pkl`)

```bash
python3 turtlebot_sysid_plot.py sysid_staircase.pkl [--save]
```

| Output | Description |
| --- | --- |
| `staircase_tracking.png` | One panel per active channel (v and/or ω): step-function reference overlaid with measured signal and shaded tracking-error band; per-level steady-state mean, std and error printed to stdout |

---

## General Notes

- All modes reset the odometry via `/reset_pose` at startup. Make sure the TurtleBot4 is connected and the service is available before running.
- Pressing **Ctrl+C** at any point saves the partial data collected so far.
- Output files are saved in the `turtlebot/` folder next to the script.
