import os
import numpy as np
import argparse
import time

import matplotlib.pyplot as plt

from visual_robot_localization.gallery_db import HlocPlaceRecognitionDBHandler



class PlaceRecognitionExponentialFilter:
    '''
    '''
    def __init__(
        self, extractor, gallery_db_path, odometry_dir_path, delta=5, window_lower=-2, window_upper=10,
    ):
        # store map data
        self.extractor = extractor
        self.db_handler = HlocPlaceRecognitionDBHandler(gallery_db_path, odometry_dir_path)

        # Sort the descriptors according to filename. Assumes filenames of the format 
        # n.suffix where n is integer and suffix is image format. The path is represented by images with indices from 0 to len(path).
        descriptors = self.db_handler.get_descriptors()
        filenames = self.db_handler.get_filepaths()
        idxs = [int(os.path.splitext(os.path.basename(idx))[0]) for idx in filenames]
        self.descriptors = np.array([x for _, x in sorted(zip(idxs, descriptors), key=lambda pair: pair[0])])
        self.node_distances = None
        self.filter_place_recognition = True

    def match(self, img):
        query_desc = self.extractor(img)['global_descriptor'].numpy().squeeze()

        diff = self.descriptors - query_desc
        dists = np.linalg.norm(diff, axis=1)


        if self.node_distances is None:
            self.node_distances = dists
        else:
            self.node_distances = 0.8 * self.node_distances + 0.2 * dists
                    
        if not self.filter_place_recognition:
            sg_idx = np.argmin(dists)
        else:
            sg_idx = np.argmin(self.node_distances)

        return sg_idx
