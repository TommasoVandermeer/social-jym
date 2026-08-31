# Interfacing with TurtleBot4 via Docker

You can interface with the TurtleBot4 directly using the provided Docker container. To do so, ensure the following requirements are met:
1. The TurtleBot4 must be configured in **Discovery Server** mode.
2. Your PC and the TurtleBot4 must be connected to the **same Wi-Fi network**.

You can start the Docker container and connect to the robot by using the `run.sh` script. By default, the script is configured to connect to the Discovery Server at `192.168.8.4` with the namespace `turtlebot1`.

If you need to specify a different IP address, you can pass them as arguments when running the script:

```bash
./docker/run.sh --ip <ROBOT_IP>
```

# Synchronize devices time (Laptop, Raspberry PI, Create3)
Ssh into the turtlebot and add the following script in the home directory (```sync_create_time.py```):

```
import requests
import datetime
import time

def brute_force_create3_time(create3_ip="192.168.186.2"):
    base_url = f"http://{create3_ip}"
   
    print("🔬 --- CREATE 3 HARDWARE TIME DEBUGGER ---")
   
    # 1. PRE-SYNC CHECK (Reading the native HTTP header)
    try:
        resp = requests.get(f"{base_url}/home", timeout=3.0)
        print(f"🕵️ Internal time (Pre-Sync) measured via HTTP:  {resp.headers.get('Date', 'Unknown')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Unable to connect to Create 3: {e}")
        return

    # 2. ATTACKING THE NTP DAEMON
    print("🔨 Sending shutdown/restart signal to the internal NTP daemon...")
    try:
        requests.post(f"{base_url}/api/restart-ntpd", timeout=3.0)
        # Tactical pause: wait for the process to spin down and release the kernel lock
        time.sleep(0.5)
    except Exception as e:
        print(f"⚠️ Error restarting NTP: {e}")

    # 3. FORCING THE NEW TIME
    # Explicitly adding the trailing 'Z' to force Zulu encoding (UTC)
    now = datetime.datetime.utcnow().replace(microsecond=0)
    formatted_time = now.isoformat() + "Z"
   
    print(f"📡 Injecting absolute time: {formatted_time}")
    try:
        response = requests.post(
            f"{base_url}/api/set-datetime",
            data={'newdatetime': formatted_time},
            timeout=5.0
        )
        print(f"Server Response: {response.status_code}")
    except Exception as e:
        print(f"⚠️ POST Error: {e}")

    # 4. POST-SYNC CHECK
    time.sleep(1.0) # Give the kernel time to process the tick
    try:
        resp_after = requests.get(f"{base_url}/home", timeout=3.0)
        print(f"🕵️ Internal time (Post-Sync) measured via HTTP: {resp_after.headers.get('Date', 'Unknown')}")
    except:
        pass
       
    print("🔬 --- END OF DIAGNOSTICS ---")

if __name__ == "__main__":
    brute_force_create3_time()
```

Open ```turtlebot/sync_devices.sh``` and edit ```PI_IP``` to match the IP address given to your turtlebot's Raspberry on the WiFi network.

Finally, on the turtlebot folder, run ```./sync_devices_online.sh``` to synchronize all the devices times (make sure to have chrony on your laptop ```sudo apt update && sudo apt install chrony```).

# DEPLOYMENT ON TURTLEBOT4

WARNING: Always make sure to be connected to the same WiFi of the turtlebot before running the scripts. 

The action inference will be run on the remote device used to run the scripts and sent to the Turtlebot4 through a ROS2 interface.

## JESSI (JAX-based E2E Safe Social Interpretable navigation)

Instructions to deploy JESSI on the Turtlebot:

- Save your trained policy in the ```turtlebot``` folder.
- Turn on the Turtlebot4 and wait for it to become fully operative (all the LEDs over the display should be green).
- Launch the following script ```turtlebot_controller.py``` on your remote device using the following command on terminal. 
```
python3 turtlebot_controller.py --planner JESSI -g REPLACE_WITH_GOALS_X_AND_Y -n REPLACE_WITH_NETWORK_NAME -s REPLACE_WITH_EXPERIMENT_NAME
```
The ```g``` flag indicate the position of the goals in the robot frame (<b>positive x axis is on the front of the robot, positive y axis is on the left of the robot</b>). Goals Xs and Ys must be indicated consecutively. The trained network used for control will be the one indicated after the ```n``` flag (include .pkl at the end). The trajectory data will be saved in the ```turtlebot``` folder under the name indicated after the ```s``` flag (include .pkl at the end). For other interesting options, checkout the ```--help```.

Note that, to sync the timestamps of each topic (for debugging purposes), it is necessary to run ```sudo chronyc makestep``` on the turtlebot raspberrypi (connect with ssh).

To animate the recorded trajectory run:
```
python3 turtlebot_jessi_animate_recorded_trajectory.py -s REPLACE_WITH_EXPERIMENT_NAME
```

# SYSTEM IDENTIFICATION ON TURTLEBOT4

# REAL-WORLD JESSI VS DWA EXPERIMENTS

The reproducible 20-run corridor protocol, ROS bag capture workflow, pedestrian
tracking instructions, and metric tools are documented in
[`experiments/README.md`](experiments/README.md).

After collecting a trial, process the latest recorded run without looking up
or typing its run-directory name:

```bash
python3 turtlebot/experiments/process_run.py \
  --latest \
  --config turtlebot/experiments/corridor_campaign.json \
  --save-animation
```

`--latest` reads the campaign schedule and selects the highest-ordinal run that
has an existing directory and `manifest.json`; pending trials are ignored. It
cannot be combined with a positional run directory. The same shortcut supports
`--skip-tracking` when only aligned data and metrics need to be regenerated:

```bash
python3 turtlebot/experiments/process_run.py \
  --latest \
  --config turtlebot/experiments/corridor_campaign.json \
  --skip-tracking
```
