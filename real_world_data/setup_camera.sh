#!/bin/bash

device_info=$(v4l2-ctl --list-devices)
camera_path=$(echo "$device_info" | awk 'NR == 2 { sub(/^[ \t]+/, "", $0); print }')
echo "Camera path is now at: $camera_path"

v4l2-ctl -d $camera_path --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d $camera_path --set-ctrl=focus_absolute=30

# v4l2-ctl -d $camera_path --set-ctrl=exposure_auto=1
# v4l2-ctl -d $camera_path --set-ctrl=exposure_absolute=70

echo "Camera setup complete"
