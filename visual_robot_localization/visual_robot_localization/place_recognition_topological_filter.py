import os
import numpy as np
import argparse
import time

import matplotlib.pyplot as plt

from visual_robot_localization.gallery_db import HlocPlaceRecognitionDBHandler


class PlaceRecognitionTopologicalFilter:
    '''
    Adapted from https://github.com/mingu6/ProbFiltersVPR/blob/master/src/models/TopologicalFilter.py
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

        # initialize hidden states and obs lhood parameters
        self.delta = delta
        self.lambda1 = 0.0
        self.belief = None
        # parameters for prior transition matrix
        self.window_lower = window_lower
        self.window_upper = window_upper
        self.window_size = int((window_upper - window_lower) / 2)
        self.transition = np.ones(window_upper - window_lower)

    def initialize_model(self, img):
        query_desc = self.extractor(img)['global_descriptor'].numpy().squeeze()
        dists = np.sqrt(2 - 2 * np.dot(self.descriptors, query_desc))
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (
            descriptor_quantiles[1] - descriptor_quantiles[0]
        )
        self.belief = np.exp(-self.lambda1 * dists)
        self.belief /= self.belief.sum()

    def obs_lhood(self, descriptor):
        vsim = np.exp(
            -self.lambda1 * np.sqrt(2 - 2 * np.dot(self.descriptors, descriptor))
        )
        
        return vsim

    def match(self, img):
        query_desc = self.extractor(img)['global_descriptor'].numpy().squeeze()

        w_l = self.window_lower
        if w_l < 0:
            conv_ind_l, conv_ind_h = np.abs(w_l), len(self.belief) + np.abs(w_l)
            bel_ind_l, bel_ind_h = 0, len(self.belief)
        else:
            conv_ind_l, conv_ind_h = 0, len(self.belief) - w_l
            bel_ind_l, bel_ind_h = w_l, len(self.belief)

        # apply prior transition matrix
        self.belief[bel_ind_l:bel_ind_h] = np.convolve(self.belief, self.transition)[
            conv_ind_l:conv_ind_h
        ]
        if w_l > 0:
            self.belief[:w_l] = 0.0

        # observation likelihood update
        self.belief *= self.obs_lhood(query_desc)
        self.belief /= self.belief.sum()

        # Argmax of the belief
        max_bel = np.argmax(self.belief)

        nhood_inds = np.arange(
            max(max_bel - 2 * self.window_size, 0),
            min(max_bel + 2 * self.window_size, len(self.belief) - 1),
        )
        score = np.sum(self.belief[nhood_inds])
        proposal = int(np.rint(np.average(nhood_inds, weights=self.belief[nhood_inds])))
        
        return proposal, score