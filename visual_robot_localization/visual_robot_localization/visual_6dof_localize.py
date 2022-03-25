from hloc.localize_sfm import do_covisibility_clustering
from visual_robot_localization.place_recognition_querier import PlaceRecognitionQuerier
from visual_robot_localization.coordinate_transforms import colmap2ros_coord_transform
from visual_robot_localization.hloc_models import FeatureExtractor, FeatureMatcher
import hloc.extract_features
import hloc.match_features


import h5py
import torch
import numpy as np
import pycolmap
from collections import defaultdict
from pathlib import Path
from ast import literal_eval


class VisualPoseEstimator:

    def __init__(self,
                global_extractor_name,
                local_extractor_name,
                local_matcher_name,
                image_gallery_path,
                gallery_global_descriptor_path,
                gallery_local_descriptor_path,
                reference_sfm_path,
                ransac_thresh = 12):

        assert Path(image_gallery_path).exists, image_gallery_path
        assert Path(gallery_global_descriptor_path).exists, gallery_global_descriptor_path
        assert Path(gallery_local_descriptor_path).exists, gallery_local_descriptor_path
        assert Path(reference_sfm_path).exists, reference_sfm_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        extractor_confs = hloc.extract_features.confs.copy()
        matcher_confs = hloc.match_features.confs.copy()

        global_extractor = FeatureExtractor(extractor_confs[global_extractor_name])
        self.pr_querier = PlaceRecognitionQuerier(global_extractor, gallery_global_descriptor_path, image_gallery_path)

        # Visual 6DoF pose estimation init
        self.local_extractor = FeatureExtractor(extractor_confs[local_extractor_name])
        self.gallery_local_descriptor_file = h5py.File(gallery_local_descriptor_path, 'r')
        self.local_matcher = FeatureMatcher(matcher_confs[local_matcher_name])

        # Load the prebuilt colmap 3D pointcloud
        self.reconstruction = pycolmap.Reconstruction(reference_sfm_path)
        self.query_camera = self.reconstruction.cameras[1]

        self.ransac_thresh = ransac_thresh
        self.db_name_to_id = {image.name: i for i, image in self.reconstruction.images.items()}

    def estimate_pose(self, query_img, topk, covisibility_clustering = True, exclude_best_match = False):

        # Retrieve topk similar gallery images
        ksmallest_filenames, _, ksmallest_odometries = self.pr_querier.match(query_img, topk)
        ksmallest_filenames = [filename.split('/')[-1] for filename in ksmallest_filenames]

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
                ret = self._pose_from_cluster_online(cluster_ids, cluster_matches, query_local_descriptors)
            else:
                # If local feature matching failed
                ret = {'success': False}

            ret['place_recognition_idx'] = cluster_idxs

            if ret['success']:
                ret['tvec'], ret['qvec'] = colmap2ros_coord_transform(ret['tvec'], ret['qvec'])

            estimates['pnp_estimates'].append(ret)

        return estimates

    def _pose_from_cluster_online(self, db_ids, query_matches, query_local_descriptors):

        # Magic to get the 3D points from the sfm database
        # that correspond to the matched 2D image features
        #
        # Adapted from hloc pose_from_cluster

        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        kp_idx_to_3D = defaultdict(list)

        for i, db_id in enumerate(db_ids):
            matches = query_matches[i]

            image = self.reconstruction.images[db_id]
            if image.num_points3D() == 0:
                print('No 3D points found for image {image.name}')
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])
            
            valid = np.where(matches > -1)[0]
            valid = valid[points3D_ids[matches[valid]] != -1]

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
        return ret