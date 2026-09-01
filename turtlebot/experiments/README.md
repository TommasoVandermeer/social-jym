# TurtleBot4 Corridor Experiment: JESSI vs DWA

This directory contains the reproducible workflow for comparing JESSI and DWA
over 10 real-world trials per policy. The primary metrics are time to goal,
average translational jerk, path length, and pedestrian-space compliance.

All commands below are run from `/opt/social-jym` inside the project Docker
container unless stated otherwise.

## 1. Safety and experimental controls

- Assign one person exclusively as safety operator. That person must have
  immediate access to the TurtleBot emergency stop and must not participate as
  a pedestrian.
- Use the hardware emergency stop first if contact is imminent. Then press
  `Ctrl+C` in the experiment terminal and label the run as a collision or
  safety stop when prompted.
- Mark a repeatable robot start pose, pedestrian starting region, walking
  direction, and goal on the floor. Do not change them between policies.
- Use the same nominal five-pedestrian opposite-flow instruction in every run.
  Record the actual count and any deviations in the post-run prompts.
- Disable the optional engineering filters for this comparison. Their current
  JESSI path includes policy-specific geometric blending and would not be a
  controlled comparison with DWA.
- Do not use `--patrol`, diagnostics, automatic corridor alignment, or pure
  pursuit unless those options are deliberately enabled in the campaign JSON
  before the first trial and then held constant for all 20 trials.

Operator-confirmed physical collisions are authoritative. The computed
clearance-based collision value is only a proxy and never overrides the
operator label.

## 2. One-time robot and clock preparation

Start the container from the host:

```bash
./docker/run.sh --ip ROBOT_IP
```

Follow the time-synchronization procedure in `turtlebot/README.md`. In
particular, run `sudo chronyc makestep` on the TurtleBot Raspberry Pi before the
campaign. Then verify connectivity and topic activity:

```bash
ros2 topic hz /turtlebot1/scan
ros2 topic hz /turtlebot1/odom
ros2 topic echo /turtlebot1/cmd_vel_stamped --once
ros2 service list | grep /turtlebot1/reset_pose
```

Do not begin if scan or odometry timestamps move backward, topic delivery is
intermittent, or the reset-pose service is unavailable.

The raw ROS bag retains original device headers. Offline conversion maps each
topic into the bag recorder's host clock using a robust median offset and
produces `timestamp_alignment.json`. A p95 alignment residual above 50 ms is a
warning; above 250 ms makes the run invalid until the timing problem is
resolved.

## 3. Configure and initialize the campaign

Copy the template once and edit it:

```bash
cp turtlebot/experiments/experiment_config.example.json \
   turtlebot/experiments/corridor_campaign.json
```

The following fields must be correct before initialization:

- `campaign_name`: directory-safe name for this campaign.
- `goal`: `[x, y]` in metres in the reset robot odometry frame; positive X is
  forward and positive Y is left.
- `jessi_network`: path relative to the JSON file or an absolute path.

The template fixes the agreed protocol defaults: 4 Hz, 300 LiDAR rays, 0.8 m
goal threshold, 120 s timeout, 10 trials per policy, 0.50 m personal-space
clearance, 0.30 m robot and human radii, 15-frame tracking checkpoints, and a
21-frame looping labeling preview. Set `label_preview_window_frames` to another
positive odd number to show more or less context around every labeling frame.

Initialize and inspect the reproducible balanced-randomized schedule:

```bash
python3 turtlebot/experiments/run_experiment.py init \
  --config turtlebot/experiments/corridor_campaign.json

python3 turtlebot/experiments/run_experiment.py status \
  --config turtlebot/experiments/corridor_campaign.json
```

Initialization records the Git commit, effective configuration, JESSI network
SHA-256, and random schedule. It fails rather than overwrite an existing
campaign.

Generated data live under:

```text
turtlebot/experiments/data/CAMPAIGN_NAME/
  campaign_config.json
  campaign_manifest.json
  schedule.json
  run_001_jessi_trial_01/
    manifest.json
    controller.log
    controller.pkl
    raw_sensor_messages.pkl
    rosbag/
    rosbag.log
```

The exact first policy depends on the saved schedule; never choose the next
policy manually.

## 4. Execute each physical trial

Before every run:

1. Clear and inspect the corridor.
2. Put the robot exactly on the marked start pose and heading.
3. Release the emergency stop and confirm all TurtleBot indicators are healthy.
4. Put pedestrians in the agreed starting region and repeat the same walking
   instruction.
5. Check `run_experiment.py status` and announce the scheduled policy.

Start the next scheduled run:

```bash
python3 turtlebot/experiments/run_experiment.py run-next \
  --config turtlebot/experiments/corridor_campaign.json
```

The runner starts the ROS bag before the controller. The controller resets
odometry, publishes at the configured rate, and exits automatically on the
final goal or at 120 seconds. Afterward, enter the actual pedestrian count and
notes.

If intervention is required:

1. Press the physical emergency stop if safety requires it.
2. Press `Ctrl+C` in the runner terminal.
3. Select collision, safety stop, or unrelated operator abort.
4. Add a concise note identifying what happened.

If the interruption was an unrelated operator abort—for example, an incorrect
start pose or unavailable ROS topics—the attempt can be repeated without
consuming its scheduled trial. First archive and reset the latest attempt:

```bash
python3 turtlebot/experiments/run_experiment.py retry-last \
  --config turtlebot/experiments/corridor_campaign.json
```

Then correct the setup problem and start it again with the normal command:

```bash
python3 turtlebot/experiments/run_experiment.py run-next \
  --config turtlebot/experiments/corridor_campaign.json
```

The retry command never deletes or overwrites data. It moves the original run
into `CAMPAIGN_NAME/aborted_attempts/RUN_DIRECTORY_attempt_01`, records that
path in `schedule.json`, and resets the same policy/trial entry to pending.
Additional retries receive increasing attempt numbers, and the replacement
run's manifest records its `attempt_number`. The `status` command also reports
how many attempts were archived. Only `operator_abort`
and `controller_error` outcomes are retryable, and only when they are the most
recent attempted schedule entry. Collisions, safety stops, timeouts, and
successful trials remain part of the experiment and cannot be retried through
this command.

After every trial, and before repositioning for the next one:

```bash
ros2 bag info turtlebot/experiments/data/CAMPAIGN_NAME/RUN_DIRECTORY/rosbag
```

Confirm that scan, odometry, and stamped command topics all contain messages,
that the duration is plausible, and that `manifest.json` has a final outcome.
If logging is incomplete, label the attempt as an operator abort or let the
runner record a controller error, then use `retry-last`. Never manually delete,
rename, or overwrite a run directory.

## 5. Extract and verify pedestrian tracks

Process each run after collection, preferably before leaving the experiment
site so raw data quality can be checked:

```bash
python3 turtlebot/experiments/process_run.py \
  turtlebot/experiments/data/CAMPAIGN_NAME/RUN_DIRECTORY \
  --save-animation
```

Immediately after collecting a trial, the latest recorded run can be selected
from the campaign schedule automatically:

```bash
python3 turtlebot/experiments/process_run.py \
  --latest \
  --config turtlebot/experiments/corridor_campaign.json \
  --save-animation
```

`--latest` selects the highest-ordinal run directory that exists and contains
a manifest; pending schedule entries are ignored. A positional run directory
and `--latest` cannot be used together.

This command performs three stages:

1. Rebuilds timestamp-aligned `sensor_messages.pkl` from the original ROS bag.
2. Opens the human tracker and writes `human_tracks.npz` plus metadata.
3. Computes `metrics.json` and the control-time series
   `control_metrics.csv`.

The fullscreen tracker window has the selectable labeling scan on the left and
a continuously looping scan preview on the right. The preview is centered on
the labeling frame (except where clipped at the beginning or end of a run),
and its title clearly marks the exact `LABEL FRAME`. It is aligned from
odometry relative to the labeling scan so motion is easier to recognize. At
periodic checkpoints, valid track positions and identity numbers are drawn on
past frames and on the current prediction. Future frames remain scan-only and
are marked `FUTURE (scan only)` because their tracks have not been computed
yet. The context panel intentionally does not draw velocity arrows.

At the first tracker window, left-click every visible pedestrian in the left
panel and press Enter. At each periodic checkpoint:

- Click to correct the selected identity.
- Press a number or Tab to select another identity.
- Press `N`, then click, to add a pedestrian entering later.
- Press `D` to deactivate a pedestrian who has left the scene.
- Press `R`, then click, to reactivate the selected identity.
- Press Enter without clicking when all predictions are correct.

Every track includes active, valid, observed, and manual-correction masks.
Three consecutive missed LiDAR updates are tolerated; longer active gaps are
treated as a pedestrian no longer being tracked. Such a track is ignored until
it becomes valid again; it does not invalidate measurements from other visible
pedestrians.

To recompute aligned messages and metrics while keeping an already verified
`human_tracks.npz`:

```bash
python3 turtlebot/experiments/process_run.py RUN_DIRECTORY --skip-tracking
```

The same shortcut works when only metrics need rebuilding:

```bash
python3 turtlebot/experiments/process_run.py --latest \
  --config turtlebot/experiments/corridor_campaign.json --skip-tracking
```

All derived files can be deleted and regenerated from `rosbag/`. A controller
or analysis failure therefore does not require repeating a physical trial.

## 6. Metric definitions

All robot and pedestrian states are evaluated at the actual
`cmd_vel_stamped` publication timestamps, not at assumed 4 Hz timestamps.
Interpolation never extrapolates and rejects source gaps over 0.5 s.

- **Time to goal:** first published control command to the explicit
  goal-reached event. It is only defined for successful runs.
- **Path length:** sum of odometry position increments at control timestamps.
  The primary comparison is success-only; distance until termination is also
  retained for every run.
- **Average jerk:** odometry body speed is transformed into global velocity.
  A timestamp-aware quadratic fit over seven control samples provides
  acceleration and jerk. The reported value is the time-weighted mean jerk
  magnitude in m/s³.
- **Space compliance:** for each currently valid pedestrian track, clearance is
  `center distance - 0.30 m robot radius - 0.30 m human radius`. A sample is
  compliant above 0.50 m clearance, equivalent to a 1.10 m center distance.
  Invalid or departed tracks are ignored. Intervals with no valid pedestrian
  tracks are compliant. A sample is unknown only when timestamp alignment or a
  tracking-data gap prevents evaluating the scene. Runs below 90% analyzable
  temporal coverage are excluded from the policy-level compliance comparison.
- **Collision:** the manifest's operator label is primary. A clearance at or
  below zero is additionally reported as an automatic collision proxy.

The per-run output also contains minimum human clearance, average speed,
control timing mean/standard deviation/p95, tracking coverage, timeout, and
timestamp synchronization validity.

## 7. Aggregate the 20 trials

After all usable runs have `metrics.json`:

```bash
python3 turtlebot/experiments/aggregate_results.py \
  turtlebot/experiments/data/CAMPAIGN_NAME
```

Outputs are:

- `campaign_metrics.csv`: one row per run.
- `policy_summary.csv`: count, mean, standard deviation, median, IQR, and 95%
  bootstrap confidence interval by policy and cohort.
- `policy_comparison.json`: seeded JESSI-minus-DWA mean differences and 95%
  bootstrap intervals.
- `policy_comparison.png`: time, path, jerk, and compliance plots.

Time to goal and primary path length use successful runs only. Jerk and space
compliance are summarized over both all analyzable runs and successful runs.
Success, operator collision, and timeout rates include all completed trials.

## 8. Recovery checklist

- **Controller crashed but bag exists:** keep the run; fix the code and rerun
  `process_run.py`. Metrics rely on bag data plus any surviving controller
  samples. If control samples are absent, mark the trial failed rather than
  inventing them.
- **Tracker was wrong:** rerun `process_run.py` without `--skip-tracking`; the
  original bag is unchanged.
- **Metric implementation changed:** rerun with `--skip-tracking`, then rerun
  aggregation. No robot trial is needed.
- **Timing report invalid:** inspect device synchronization and bag timestamps.
  Do not include that run merely because its metric values look plausible.
- **Wrong policy, goal, or start pose:** record the deviation in the manifest
  notes and exclude the run; do not rename it or overwrite it.
