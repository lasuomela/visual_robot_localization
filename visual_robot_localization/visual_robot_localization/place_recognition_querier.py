import numpy as np

from visual_robot_localization.gallery_db import HlocPlaceRecognitionDBHandler

class PlaceRecognitionQuerier:

    def __init__(self, extractor, gallery_db_path, odometry_dir_path):
        self.extractor = extractor
        self.db_handler = HlocPlaceRecognitionDBHandler(gallery_db_path, odometry_dir_path)

    def match(self, img, k):
        if k > self.db_handler.gallery_len:
            raise ValueError('Specified value of k ({}) exceeds size of gallery ({})'.format(k, self.db_handler.gallery_len))

        query_ret = self.extractor(img)

        diff = self.db_handler.get_descriptors()-query_ret['global_descriptor'].numpy()
        dist = np.linalg.norm(diff, axis=1)

        if k < len(dist):
            idx_part = np.argpartition(dist, k)[:(k)]
        else:
            idx_part = np.arange(k)

        idx_partsort = np.argsort(dist[idx_part])
        idx_ksmallest = idx_part[idx_partsort]

        ksmallest_dist = dist[idx_ksmallest]
        ksmallest_filenames = []
        ksmallest_odometries = []

        for idx in idx_ksmallest:
            filename, _, odometry = self.db_handler.get_by_idx(idx)
            ksmallest_filenames.append(filename)
            ksmallest_odometries.append(odometry)

        return ksmallest_filenames, ksmallest_dist, ksmallest_odometries

    def match_dot(self, img, k):
        if k > self.db_handler.gallery_len:
            raise ValueError('Specified value of k ({}) exceeds size of gallery ({})'.format(k, self.db_handler.gallery_len))

        query_ret = self.extractor(img)

        dist = 1/np.dot(self.db_handler.get_descriptors(), query_ret['global_descriptor'].numpy().squeeze())

        if k < len(dist):
            idx_part = np.argpartition(dist, k)[:(k)]
        else:
            idx_part = np.arange(k)

        idx_partsort = np.argsort(dist[idx_part])
        idx_ksmallest = idx_part[idx_partsort]

        ksmallest_dist = dist[idx_ksmallest]
        ksmallest_filenames = []
        ksmallest_odometries = []

        for idx in idx_ksmallest:
            filename, _, odometry = self.db_handler.get_by_idx(idx)
            ksmallest_filenames.append(filename)
            ksmallest_odometries.append(odometry)

        return ksmallest_filenames, ksmallest_dist, ksmallest_odometries

