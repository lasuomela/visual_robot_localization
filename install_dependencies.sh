#!/bin/bash

ROS_DISTRO='foxy'

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

apt-get update && apt-get install -y \
        python3-pip \
        python3-opencv \
        ros-$ROS_DISTRO-cv-bridge \
	colmap=3.6+dev2+git20191105-1build1 \ # Install regular colmap to enable visualization
	libceres-dev \
	wget

# A specific version of pytorch for compatibility with nvidia 3090	
pip3 install -r $SCRIPT_DIR/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
