#!/bin/bash

# --- CONFIGURATION ---
PI_USER="ubuntu"
PI_IP="192.168.0.111"  # Modify with your Turtlebot's PI IP address
PI_PASS="turtlebot4"
SCRIPT_PATH="~/sync_create_time.py"
# ----------------------

echo "0/2 Synching Laptop (Local)..."
sudo chronyc makestep

echo "⏱️  Synching Turtlebot4 clocks..."
sshpass -p "${PI_PASS}" ssh -t ${PI_USER}@${PI_IP} "
    echo '1/2 Synching Raspberry Pi...' &&
    echo '${PI_PASS}' | sudo -S chronyc makestep &&
    echo '2/2 Synching Create 3...' &&
    python3 ${SCRIPT_PATH}
"
echo "✅ Devices are temporally allined! (Laptop, Raspberry Pi, Create 3)"