from visual_robot_localization.visual_6dof_localize import VisualPoseEstimator

import transforms3d as t3d
import re
import json
import cv2
import os
import numpy as np
import time

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))

    query_img_path = script_dir+'/example_dir/im_0001.png'

    global_extractor_name = 'netvlad'
    local_extractor_name = 'superpoint_aachen'
    local_matcher_name = 'superglue'

    image_gallery_path = script_dir+'/example_dir'
    gallery_global_descriptor_path = script_dir+'/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
    gallery_local_descriptor_path = script_dir+'/example_dir/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5'
    reference_sfm_path = script_dir+'/example_dir/outputs/netvlad+superpoint_aachen+superglue/sfm_netvlad+superpoint_aachen+superglue'

    # gallery_global_descriptor_path = script_dir+'/example_dir/outputs/dir+superpoint_aachen+superglue/global-feats-dir.h5'
    # gallery_local_descriptor_path = script_dir+'/example_dir/outputs/dir+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5'
    # reference_sfm_path = script_dir+'/example_dir/outputs/dir+superpoint_aachen+superglue/sfm_dir+superpoint_aachen+superglue'

    topk = 5

    image_types = ['.png', '.jpg', '.jpeg']
    odometry_suffix = '_odometry_camera.json'

    query_img = cv2.imread(query_img_path, cv2.IMREAD_COLOR)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    estimator = VisualPoseEstimator(global_extractor_name,
                                    local_extractor_name,
                                    local_matcher_name,
                                    image_gallery_path,
                                    gallery_global_descriptor_path,
                                    gallery_local_descriptor_path,
                                    reference_sfm_path,
                                    12)

    # Exclude the best match which is the query image itself
    for i in range(5):
        start = time.time()
        ret = estimator.estimate_pose(query_img, topk, exclude_best_match=True)
        print('{:.3f}s'.format(time.time()-start))

    # Choose the pose estimate based on the number of ransac inliers
    best_inliers = 0
    best_estimate = None
    for estimate in ret['pnp_estimates']:
        if 'num_inliers' in estimate:
            if estimate['num_inliers'] > best_inliers:
                best_estimate = estimate

    # If no estimates are succesfull, use top place recognition pose 
    if best_estimate:
        tvec, qvec = best_estimate['tvec'], best_estimate['qvec']
    else:
        tvec, qvec = ret['place_recognition'][0]['tvec'], ret['place_recognition'][0]['qvec']

    # Load ground truth poses
    re_string = '('
    for i, im_type in enumerate(image_types):
        if i != 0:
            re_string = re_string + '|'
        re_string = re_string + im_type
    re_string = re_string + ')'
    pattern = re.compile(re_string)
    odom_path = pattern.sub(odometry_suffix, query_img_path)

    with open(odom_path, 'r') as odom_file:
        odom = json.load(odom_file)

    true_pos = odom['pose']['pose']['position']
    true_pos = np.array([true_pos['x'], true_pos['y'], true_pos['z']])

    true_orient = odom['pose']['pose']['orientation']
    true_orient = np.array([true_orient['w'], true_orient['x'], true_orient['y'], true_orient['z']])

    print('T: x: {}, y: {}, z: {}'.format( tvec[0], tvec[1], tvec[2] ))
    print('Q: w: {}, x: {}, y: {}, z: {}'.format( qvec[0], qvec[1], qvec[2], qvec[3] ))

    # Compute error between estimate and real pose
    diff_t = np.linalg.norm(tvec-true_pos)
    qd = t3d.quaternions.qmult(qvec, t3d.quaternions.qinverse(true_orient))
    qd = qd/t3d.quaternions.qnorm(qd)
    angle, dist = t3d.quaternions.quat2axangle(qd)
    # return the angular distance to the direction with smaller angle
    diff_deg = min(dist/(2*np.pi), 1-dist/(2*np.pi))*360

    if best_estimate:
        print('Colmap estimation successful')
    else:
        print('Colmap estimation failed, returning top1 place recognition query coordinates')

    print('Diff T: {} m'.format( diff_t ))
    print('Diff R: {} deg'.format( diff_deg ))

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='CLI test for 6DoF visual localization')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main()
