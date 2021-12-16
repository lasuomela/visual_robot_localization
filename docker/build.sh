#!/bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

ROS_DISTRO="foxy"
BASE_IMAGE="ros:$ROS_DISTRO-ros-base"

docker build \
    -t visual_robot_localization:0.0.1 \
    --build-arg ROS_DISTRO=$ROS_DISTRO \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -f Dockerfile ${SCRIPT_DIR}/..
