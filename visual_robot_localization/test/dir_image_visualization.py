from visual_robot_localization.visual_6dof_localize import VisualPoseEstimator
from visual_robot_localization.coordinate_transforms import colmap2ros_coord_transform
from hloc.localize_sfm import do_covisibility_clustering


import transforms3d as t3d
import re
import json
import cv2
import os
import numpy as np
import time
from pathlib import Path

import math
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import defaultdict
import pycolmap


#function to find transform matrix colmap coordinates to real floorplan pixel coordinates
def map_to_real():
    # parameters for the TAU indoors 3rd floor map (tf3_s1_images)
    pts_src = np.array(
        [[0, 0], [58.523251, 4.388469], [13.228824, 27.472746], [42.884182, 29.959358], [40.872982, 8.744554]])
    # corresponding points from real map
    pts_dst = np.array([[1257, 2011], [2493, 1859], [1501, 1413], [2134, 1330], [2110, 1782]])

    # calculate matrix H
    h, status = cv2.findHomography(pts_src, pts_dst)

    return h

class DebugVisualPoseEstimator(VisualPoseEstimator):
    '''
    Visual localization visualizer. Display query and retrieved gallery images + extracted local features in the images.
    The estimated poses are displayed on top of a 2d map. The map is provided for the TAU-Indoors dataset. The 'test' folder
    contains a small subset of TAU-Indoors. You can request the full dataset from the authors. Performamce with
    the subset only is likely to be bad because of the small number of images.
    
    Move through the images in query folder using 'a' and 'd' keys.

    '''

    def __init__(self,
                global_extractor_name,
                local_extractor_name,
                local_matcher_name,
                image_gallery_path,
                gallery_global_descriptor_path,
                gallery_local_descriptor_path,
                reference_sfm_path,
                ransac_thresh = 12):

        super().__init__(global_extractor_name,
                        local_extractor_name,
                        local_matcher_name,
                        image_gallery_path,
                        gallery_global_descriptor_path,
                        gallery_local_descriptor_path,
                        reference_sfm_path,
                        ransac_thresh)
                        
        self.image_gallery_path = image_gallery_path

    def estimate_pose(self, query_img_list, topk, covisibility_clustering = True, exclude_best_match = False):
        '''
        Override the original implementation to add visualizations
        '''

	# Map image here
        map_img = cv2.imread(self.image_gallery_path + '/resource/3_grey.png', cv2.IMREAD_GRAYSCALE)
        map_img = cv2.cvtColor(map_img,cv2.COLOR_GRAY2RGB)
        h = map_to_real() #load transform matrix

        img_number = 0
        while img_number < len(query_img_list):
            query_img_path = query_img_list[img_number]
            query_img = cv2.imread(str(query_img_path), cv2.IMREAD_COLOR)
            query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

            # Retrieve topk similar gallery images
            ksmallest_filepaths, ksmallest_distances, ksmallest_odometries = self.pr_querier.match(query_img_rgb, topk)
            ksmallest_filenames = [filename.split('/')[-1] for filename in ksmallest_filepaths]

            # For testing with images from same sequence remove the first match since its the image itself
            if exclude_best_match:
                ksmallest_filenames = ksmallest_filenames[1:]
                ksmallest_odometries = ksmallest_odometries[1:]


            # Get the sfm database id's of the images
            db_ids = []
            for n in ksmallest_filenames:
                if n not in self.db_name_to_id:
                    print('Image {n} was retrieved but not in database')
                    continue
                db_ids.append(self.db_name_to_id[n])

            # Get the local descriptors of the gallery images
            topk_gallery_images_local_descriptors = [self.gallery_local_descriptor_file[filename] for \
                                                    filename in ksmallest_filenames]

            topk_gallery_images_local_descriptors = [{key: torch.tensor(value[:]).to(self.device) for \
                                                        key, value in descriptor.items()} for \
                                                        descriptor in topk_gallery_images_local_descriptors]

            topk_gallery_poses = [ literal_eval(odom)['pose']['pose'] for odom in ksmallest_odometries ]

            # Extract the local descriptors for the query image
            query_local_descriptors = self.local_extractor(query_img)

            gallery_matches = None
            # At least 2 feature descriptors needed to attempt feature matching
            if query_local_descriptors['descriptors'].size(dim=-1) > 1: 
                # The order of the descriptors matters! Query first.
                data = self.local_matcher.prepare_data(query_local_descriptors, topk_gallery_images_local_descriptors, query_img.shape[0:2])
                gallery_matches, _ = self.local_matcher(data)

            if covisibility_clustering:
                # Logic to partition the top k gallery images into distinct clusters by feature covisibility
                clusters = do_covisibility_clustering(db_ids, self.reconstruction)

            else:
                # A single cluster with all of the images retrieved by place recognition
                clusters = [db_ids]

            estimates = {} 
            estimates['pnp_estimates'] = []
            estimates['place_recognition'] = [{'tvec':np.array([pose['position']['x'],
                                                                pose['position']['y'],
                                                                pose['position']['z'] ]),

                                                'qvec':np.array([pose['orientation']['w'],
                                                                pose['orientation']['x'],
                                                                pose['orientation']['y'],
                                                                pose['orientation']['z'] ])
                } for pose in topk_gallery_poses]

            for cluster_ids in clusters:
                cluster_idxs = [ db_ids.index(id) for id in cluster_ids ]
                if gallery_matches is not None:
                    cluster_matches = [gallery_matches[idx,:] for idx in cluster_idxs]
                    # PnP to estimate the 6DoF location of the query image
                    ret, qkps, gallery_kps, mp3d_ids, p3d_idx_to_db_kp = self._pose_from_cluster_online(cluster_ids, cluster_matches, query_local_descriptors)
                else:
                    # If local feature matching failed
                    ret = {'success': False}
                    ret['gallery_kps'] = []
                    ret['p3d_idx_to_db_kp'] = []
                    ret['mp3d_ids'] = []
                    ret['place_recognition_idx'] = cluster_idxs

                ret['place_recognition_idx'] = cluster_idxs
                ret['gallery_kps'] = gallery_kps

                ret['p3d_idx_to_db_kp'] = p3d_idx_to_db_kp
                ret['mp3d_ids'] = mp3d_ids

                if ret['success']:
                    ret['tvec'], ret['qvec'] = colmap2ros_coord_transform(ret['tvec'], ret['qvec'])

                    ret['qkps'] = qkps

                estimates['pnp_estimates'].append(ret)

            # Choose the pose estimate based on the number of ransac inliers
            best_inliers = 0
            best_estimate = None
            for estimate in estimates['pnp_estimates']:
                if 'num_inliers' in estimate:
                    if estimate['num_inliers'] > best_inliers:
                        best_estimate = estimate

            map_display = map_img.copy()

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

            gallery_imgs = []

            for pr_estimate in estimates['place_recognition']:

                new_loc = cv2.perspectiveTransform( np.expand_dims( pr_estimate['tvec'][:2], axis=[0,1]), h)
                _, _, yaw = t3d.euler.quat2euler(pr_estimate['qvec'])
                yaw = yaw + np.pi/2

                arrow_len = 50
                start_x = round(new_loc[0][0][0])
                start_y = round(new_loc[0][0][1])

                end_x = round(arrow_len * np.sin( yaw  ) + new_loc[0][0][0])
                end_y = round(arrow_len * np.cos( yaw  ) + new_loc[0][0][1])
                cv2.arrowedLine(map_display, (start_x, start_y), (end_x, end_y), (255,0,0), thickness=2, line_type = 4)


            for color_idx, estimate in enumerate(estimates['pnp_estimates']):

                cluster_prs = [estimates['place_recognition'][ idx ] for idx in estimate['place_recognition_idx']]
                cluster_pr_distances = [ksmallest_distances[idx] for idx in estimate['place_recognition_idx']]
                print('Smallest pr distance: {:.3f}'.format(min(cluster_pr_distances)))


                for pr_estimate in cluster_prs:

                    new_loc = cv2.perspectiveTransform( np.expand_dims( pr_estimate['tvec'][:2], axis=[0,1]), h)
                    _, _, yaw = t3d.euler.quat2euler(pr_estimate['qvec'])
                    yaw = yaw + np.pi/2

                    arrow_len = 50
                    start_x = round(new_loc[0][0][0])
                    start_y = round(new_loc[0][0][1])

                    end_x = round(arrow_len * np.sin( yaw  ) + new_loc[0][0][0])
                    end_y = round(arrow_len * np.cos( yaw  ) + new_loc[0][0][1])
                    cv2.arrowedLine(map_display, (start_x, start_y), (end_x, end_y), colors[color_idx], thickness=2, line_type = 4)

                if estimate['success']:

                    # Draw the matched keypoint locations on the query image
                    inlier_qkps = estimate['qkps'][estimate['inliers']]
                    for kp in estimate['qkps']:
                        cv2.circle(query_img, (round(kp[0]), round(kp[1])), 10, (0,0,255))

                    for kp in inlier_qkps:
                        cv2.drawMarker(query_img, (round(kp[0]), round(kp[1])), (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=10)

                    # Draw the 6DoF pose estimates on the map
                    new_loc = cv2.perspectiveTransform( np.expand_dims( estimate['tvec'][:2], axis=[0,1]), h)

                    _, _, yaw = t3d.euler.quat2euler(estimate['qvec'])
                    yaw = yaw + np.pi/2

                    arrow_len = 50
                    fov_line_len = 100
                    start_x = round(new_loc[0][0][0])
                    start_y = round(new_loc[0][0][1])

                    fov_offset = 55/360*2*np.pi

                    end_x = round(arrow_len * np.sin( yaw  ) + new_loc[0][0][0])
                    end_y = round(arrow_len * np.cos( yaw  ) + new_loc[0][0][1])

                    end_x_left = round(fov_line_len * np.sin( yaw + fov_offset ) + new_loc[0][0][0])
                    end_y_left = round(fov_line_len * np.cos( yaw + fov_offset ) + new_loc[0][0][1])

                    end_x_right = round(fov_line_len * np.sin( yaw - fov_offset ) + new_loc[0][0][0])
                    end_y_right = round(fov_line_len * np.cos( yaw - fov_offset ) + new_loc[0][0][1])

                    cv2.arrowedLine(map_display, (start_x, start_y), (end_x, end_y), colors[color_idx], thickness=2, line_type = 4)
                    cv2.line(map_display, (start_x, start_y), (end_x_left, end_y_left), colors[color_idx], thickness=2)
                    cv2.line(map_display, (start_x, start_y), (end_x_right, end_y_right), colors[color_idx], thickness=2)

                    print(estimate['num_inliers'])

                # Draw feature matches on gallery images
                cluster_gallery_filenames = [ksmallest_filepaths[idx] for idx in estimate['place_recognition_idx']]
                scale_factor = 0.4
                for idx, n in enumerate(cluster_gallery_filenames):
                    img = cv2.imread(n, cv2.IMREAD_COLOR)

                    # Matches
                    for kp in estimate['gallery_kps'][idx]:
                        cv2.circle(img, (int(kp[0].round(0)), int(kp[1].round(0))), 10, (0,0,255), thickness=2)

                    # Inlier matches
                    if 'inliers' in ret:
                        inlier_gallery_kps = []
                        valid_p3ds = np.array(mp3d_ids)[np.array(ret['inliers'])]
                        for p3d_id in valid_p3ds:
                            if p3d_id in p3d_idx_to_db_kp[idx]:
                                inlier_gallery_kps.append( p3d_idx_to_db_kp[idx][p3d_id] )
                        for kp in inlier_gallery_kps:
                            cv2.drawMarker(img, (round(kp[0]), round(kp[1])), (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=5)

                    img = cv2.resize(img, np.round(np.array([img.shape[1]*scale_factor, img.shape[0]*scale_factor])).astype(int))
                    img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,0])
                    gallery_imgs.append( img )


            gallery_panel = cv2.hconcat(gallery_imgs)

            # Crop the map image
            map_display = map_display[ round(1000*50/47.1403):, :round(1000*160/47.1403) ]
            map_display = cv2.resize(map_display, np.round(np.array([map_display.shape[1]*scale_factor, map_display.shape[0]*scale_factor])).astype(int))

            cv2.imshow('map', map_display)
            cv2.imshow('gallery', gallery_panel)
            cv2.imshow('query', query_img)

            key = cv2.waitKey(0) % 256
            if key == ord('a'):
                img_number -= 1
            elif key == ord('d'):
                img_number += 1
            elif key == ord('q'):
                break
    
    def _pose_from_cluster_online(self, db_ids, query_matches, query_local_descriptors):

        # Magic to get the 3D points from the sfm database
        # that correspond to the matched 2D image features
        #
        # Adapted from hloc pose_from_cluster

        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        kp_idx_to_3D = defaultdict(list)

        p3d_idx_to_db_kp = []
        gallery_kps = []

        for i, db_id in enumerate(db_ids):
            matches = query_matches[i]

            image = self.reconstruction.images[db_id]
            if image.num_points3D() == 0:
                print('No 3D points found for image {image.name}')
                gallery_kps.append([])
                p3d_idx_to_db_kp.append({})
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])
            
            valid = np.where(matches > -1)[0]
            valid = valid[points3D_ids[matches[valid]] != -1]

            points2D = [p.xy for p in image.points2D]
            points2D_valid = np.array(points2D)[matches[valid]]
            gallery_kps.append(points2D_valid)

            points3D_ids_valid = np.array(points3D_ids)[matches[valid]]

            p3d_idx_to_db_kp.append( {p3d_id: kp for p3d_id, kp in zip(points3D_ids_valid, points2D_valid)})

            for idx in valid:
                id_3D = points3D_ids[matches[idx]]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
        
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mkpq = query_local_descriptors['keypoints'][mkp_idxs]
        mkpq += 0.5  # COLMAP coordinates

        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        mp3d = [self.reconstruction.points3D[j].xyz for j in mp3d_ids]
        mp3d = np.array(mp3d).reshape(-1, 3)

        # Pycolmap pose estimation
        ret = pycolmap.absolute_pose_estimation(mkpq.cpu().numpy(), mp3d, self.query_camera, max_error_px = self.ransac_thresh)


        # for k in db_ids:
        #     intersection = list(set(kp_3D_to_2D[k].keys()) & set(mp3d_ids))
        #     kp_3D_to_2D[k] = [kp_3D_to_2D[k][point_3d_id] for point_3d_id in intersection]
        # print(kp_3D_to_2D)

        
        return ret, mkpq.cpu().numpy(), gallery_kps, mp3d_ids, p3d_idx_to_db_kp


def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # global_extractor_name = 'netvlad'
    # local_extractor_name = 'superpoint_aachen'
    # local_matcher_name = 'superglue'

    # global_extractor_name = 'netvlad'
    # local_extractor_name = 'sift'
    # local_matcher_name = 'NN-ratio'

    global_extractor_name = 'netvlad'
    local_extractor_name = 'superpoint_aachen'
    local_matcher_name = 'NN-ratio'


    image_gallery_path = script_dir + '/tf3_images/tf3_s1_images' #'/image-gallery/tf3_s1_images'
    gallery_global_descriptor_path = image_gallery_path+'/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
    gallery_local_descriptor_path = image_gallery_path+'/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5'
    reference_sfm_path = image_gallery_path+'/outputs/netvlad+superpoint_aachen+superglue/sfm_netvlad+superpoint_aachen+superglue'

    # gallery_global_descriptor_path = image_gallery_path+'/outputs/netvlad+sift+NN-ratio/global-feats-netvlad.h5'
    # gallery_local_descriptor_path = image_gallery_path+'/outputs/netvlad+sift+NN-ratio/feats-sift.h5'
    # reference_sfm_path = image_gallery_path+'/outputs/netvlad+sift+NN-ratio/sfm_netvlad+sift+NN-ratio'

    #gallery_global_descriptor_path = image_gallery_path+'/outputs/netvlad+r2d2+NN-ratio/global-feats-netvlad.h5'
    #gallery_local_descriptor_path = image_gallery_path+'/outputs/netvlad+r2d2+NN-ratio/feats-r2d2-n5000-r1024.h5'
    #reference_sfm_path = image_gallery_path+'/outputs/netvlad+r2d2+NN-ratio/sfm_netvlad+r2d2+NN-ratio'

    #query_dir_path = '/image-gallery/tf3_debug/'
    
    query_dir_path = script_dir + '/tf3_images/tf3_debug/'
    query_img_path = query_dir_path+'frame02140.jpg'


    topk = 5

    image_types = ['.png', '.jpg', '.jpeg']
    odometry_suffix = '_odometry_camera.json'



    estimator = DebugVisualPoseEstimator(global_extractor_name,
                                    local_extractor_name,
                                    local_matcher_name,
                                    image_gallery_path,
                                    gallery_global_descriptor_path,
                                    gallery_local_descriptor_path,
                                    reference_sfm_path,
                                    12)

    start = time.time()

    images =list(Path(query_dir_path).glob('*.jpg'))
    images.sort()
    # images = images[300:]

    #images = images[4700:]

    ret = estimator.estimate_pose(images, topk, covisibility_clustering = False, exclude_best_match=False)
    print('{:.3f}s'.format(time.time()-start))



def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize hloc results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main()
