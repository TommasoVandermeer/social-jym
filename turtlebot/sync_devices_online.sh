#!/bin/bash

# --- CONFIGURATION ---
PI_USER="ubuntu"
PI_IP="192.168.8.4"
PI_PASS="turtlebot4"
SCRIPT_PATH="~/sync_create_time.py"
# ----------------------

echo "🌐 Forcing Turtlebot4 to sync with global NTP servers (Online Mode)..."

sshpass -p "${PI_PASS}" ssh -o StrictHostKeyChecking=no -t ${PI_USER}@${PI_IP} "
    echo '1/2 Forcing Raspberry Pi clock to match NTP servers...' &&
    echo '${PI_PASS}' | sudo -S systemctl restart chronyd &&
    echo '${PI_PASS}' | sudo -S chronyc makestep &&
    date &&
    python3 ${SCRIPT_PATH}
"

echo "✅ Online Time Sync Complete! Devices are temporally aligned with NTP."