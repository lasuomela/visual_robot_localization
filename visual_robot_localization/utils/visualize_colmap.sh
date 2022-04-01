#!/bin/bash
SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")

# Default value for directory which contains your gallery images
image_folder=$CWD/../test/example_dir/

# Default value for the localization method
localization_combination_name=netvlad+superpoint_aachen+superglue

# Parse the image folder and localization method CLI args if given
while [ $# -gt 0 ]; do
  case "$1" in
    --image_folder)
      image_folder="$2"
      ;;
    --localization_combination_name)
      localization_combination_name="$2"
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


model_path=$image_folder/outputs/$localization_combination_name/
db_path=$model_path/database.db
sfm_path=$model_path/sfm_$localization_combination_name/

colmap gui \
    --database_path $db_path \
    --image_path $image_folder \
    --import_path $sfm_path
