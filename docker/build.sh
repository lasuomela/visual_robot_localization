#!/bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

ROS_DISTRO="foxy"

#cd ${SCRIPT_DIR}/../third_party/ros-bridge/docker
# https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669
# Also, the ROS ppa has to be removed while curl is being installed
#sed -i '12 a RUN /bin/bash -c "mv /etc/apt/sources.list.d/ros2-latest.list /etc/ros2-latest.list"' Dockerfile
#sed -i '13 a RUN /bin/bash -c "apt-get update && apt-get install -y curl"' Dockerfile
#sed -i '14 a RUN /bin/bash -c "mv /etc/ros2-latest.list /etc/apt/sources.list.d/ros2-latest.list"' Dockerfile
#sed -i '15 a RUN /bin/bash -c "curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -; curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"' Dockerfile
#./build.sh
#sed -i '13,16d' Dockerfile
#cd ${SCRIPT_DIR}

docker build \
    -t visual_robot_localization:0.0.1 \
    --build-arg ROS_DISTRO=$ROS_DISTRO \
    -f Dockerfile ${SCRIPT_DIR}/..
