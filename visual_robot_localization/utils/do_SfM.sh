#!/bin/bash
SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")

# Defaults
IMAGE_DB_ROOT=$CWD/../test/example_dir/
GLOBAL_EXTRACTOR='netvlad'
LOCAL_EXTRACTOR='superpoint_aachen'
LOCAL_MATCHER='superglue'

N_MATCHES=4 # try with 25
IM_SIZE_X=800
IM_SIZE_Y=600
IM_FOV=90

# Parse the CLI args if given
while [ $# -gt 0 ]; do
  case "$1" in
    --image_db_root)
      IMAGE_DB_ROOT="$2"
      ;;
    --global_extractor)
      GLOBAL_EXTRACTOR="$2"
      ;;
    --local_extractor)
      LOCAL_EXTRACTOR="$2"
      ;;
    --local_matcher)
      LOCAL_MATCHER="$2"
      ;;
    --n_matches)
      N_MATCHES="$2"
      ;;
    --im_size_x)
      IM_SIZE_X="$2"
      ;;
    --im_size_y)
      IM_SIZE_Y="$2"
      ;;
    --im_fov)
      IM_FOV="$2"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done


# Triangulate feature 3D locations from known camera poses
python3 ${CWD}/../scripts/SfM_from_camera_poses.py \
	--image_dir ${IMAGE_DB_ROOT} \
	--global_extractor_name ${GLOBAL_EXTRACTOR} \
	--local_extractor_name ${LOCAL_EXTRACTOR} \
	--local_matcher_name ${LOCAL_MATCHER} \
	--n_matches ${N_MATCHES} \
	--im_size_x ${IM_SIZE_X} \
	--im_size_y ${IM_SIZE_Y} \
	--im_fov ${IM_FOV}
