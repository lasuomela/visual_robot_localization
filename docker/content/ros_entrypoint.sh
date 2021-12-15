#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/opt/visual_robot_localization/install/setup.bash"
exec "$@"

