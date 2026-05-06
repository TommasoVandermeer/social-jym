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
- Launch the following script ```turtlebot_jessi_controller.py``` on your remote device using the following command on terminal. 
```
python3 turtlebot_jessi_controller.py -x REPLACE_WITH_GOAL_X -y REPLACE_WITH_GOAL_Y -n REPLACE_WITH_NETWORK_NAME --patrol --interp --collect -s REPLACE_WITH_EXPERIMENT_NAME
```
The ```x``` and ```y``` flags indicate the position of the goal in the robot frame (<b>positive x axis is on the front of the robot, positive y axis is on the left of the robot</b>). The trained network used for control will be the one indicated after the ```n``` flag (include .pkl at the end). The ```patrol``` flag is a boolean indicating whether the robot should go back and forth from its initial position to its goal (True, keep the flag), or if it should just reach its goal once and then stop (False, remove the flag). The ```interp``` flag is a boolean indicating whether the robot pose should be interpolated to match exactly the LiDAR timestamp (True, keep the flag), or if the latest available pose at inference should be used (False, remove the flag). The ```collect``` flag is a boolean indicating whether the entire messages published on /odom, /scan and /cmd_vel should be saved (True, keep the flag), or not (False, remove the flag). The trajectory data will be saved in the ```turtlebot``` folder under the name indicated after the ```s``` flag (include .pkl at the end).

Note that, to sync the timestamps of each topic (for debugging purposes), it is necessary to run ```sudo chronyc makestep``` on the turtlebot raspberrypi (connect with ssh).

To animate the recorded trajectory run:
```
python3 turtlebot_jessi_animate_recorded_trajectory.py -s REPLACE_WITH_EXPERIMENT_NAME
```

# SYSTEM IDENTIFICATION ON TURTLEBOT4

