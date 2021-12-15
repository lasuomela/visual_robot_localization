#!/bin/bash
image_folder=2021-11-15_13:37:47 #2021-10-06_16:43:21
model_folder=dir+superpoint_aachen+superglue #netvlad+superpoint_aachen+superglue #sfm_superpoint+superglue

colmap gui \
    --database_path /image-gallery/$image_folder/outputs/$model_folder/sfm_$model_folder/database.db \
    --image_path /image-gallery/$image_folder/ \
    --import_path /image-gallery/$image_folder/outputs/$model_folder/sfm_$model_folder/ # only available in colmap 3.6
