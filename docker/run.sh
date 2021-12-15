#!/bin/bash

usage() { echo "Usage: $0 [-t <tag>] [-i <image>]" 1>&2; exit 1; }

# Defaults
DOCKER_IMAGE_NAME="visual_robot_localization"
TAG="0.0.1"

while getopts ":ht:i:" opt; do
  case $opt in
    h)
      usage
      exit
      ;;
    t)
      TAG=$OPTARG
      ;;
    i)
      DOCKER_IMAGE_NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

echo "Using $DOCKER_IMAGE_NAME:$TAG"

export SCRIPT=$(readlink -f "$0")
export CWD=$(dirname "$SCRIPT")

xhost + local:
docker run \
    -it --rm \
    --net=host \
    -e SDL_VIDEODRIVER='x11' \
    -e DISPLAY=$DISPLAY\
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/localtime:/etc/localtime \
    --shm-size=8gb \
    --mount "type=bind,src=$CWD/../visual_robot_localization/,dst=/opt/visual_robot_localization/src/visual_robot_localization" \
    --gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
    "$DOCKER_IMAGE_NAME:$TAG" "$@" 
xhost - local: 
