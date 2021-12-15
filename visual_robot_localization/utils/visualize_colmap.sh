#!/bin/bash
SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")

# Point this to the directory which contains your gallery images
image_folder=$CWD/../test/example_dir/

# Check the name of the directory that contains the output of do_SfM.sh
localization_method_name=netvlad+superpoint_aachen+superglue

model_path=$image_folder/outputs/$localization_method_name/
db_path=$model_path/database.db
sfm_path=$model_path/sfm_$localization_method_name/

colmap gui \
    --database_path $db_path \
    --image_path $image_folder \
    --import_path $sfm_path
