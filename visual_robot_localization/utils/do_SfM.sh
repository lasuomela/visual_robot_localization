#!/bin/bash
SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")

IMAGE_DB_ROOT=$CWD/../test/example_dir/
GLOBAL_EXTRACTOR='netvlad'
LOCAL_EXTRACTOR='superpoint_aachen'
LOCAL_MATCHER='superglue'
N_MATCHES=4 # try with 25

IM_SIZE_X=800
IM_SIZE_Y=600
IM_FOV=90

# Triangulate feature 3D locations based on known cameras
python3 ../scripts/SfM_from_camera_poses.py \
	--image_dir ${IMAGE_DB_ROOT} \
	--global_extractor_name ${GLOBAL_EXTRACTOR} \
	--local_extractor_name ${LOCAL_EXTRACTOR} \
	--local_matcher_name ${LOCAL_MATCHER} \
	--n_matches ${N_MATCHES} \
	--im_size_x ${IM_SIZE_X} \
	--im_size_y ${IM_SIZE_Y} \
	--im_fov ${IM_FOV}
