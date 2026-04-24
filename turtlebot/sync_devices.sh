#!/bin/bash

# --- CONFIGURATION ---
PI_USER="ubuntu"
PI_IP="192.168.0.111"
PI_PASS="turtlebot4"
SCRIPT_PATH="~/sync_create_time.py"
# ----------------------

# Prendiamo l'ora in Epoch (Secondi assoluti dal 1970, indipendenti dal fuso orario)
LAPTOP_EPOCH=$(date +%s)
echo "0/2 Laptop Time grabbed: Unix Epoch $LAPTOP_EPOCH"

echo "⏱️  Forcing Turtlebot4 clocks to match Laptop (Offline Mode)..."

# Il flag StrictHostKeyChecking=no impedisce a SSH di bloccarsi se l'IP della saponetta cambia
sshpass -p "${PI_PASS}" ssh -o StrictHostKeyChecking=no -t ${PI_USER}@${PI_IP} "
    echo '1/2 Forcing Raspberry Pi clock to match Laptop...' &&
    echo '${PI_PASS}' | sudo -S date -s '@${LAPTOP_EPOCH}' &&
    echo '2/2 Synching Create 3...' &&
    python3 ${SCRIPT_PATH}
"

echo "✅ Offline Time Sync Complete! Devices are temporally aligned."