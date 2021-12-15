import h5py
import os
import numpy as np
from pathlib import Path

class HlocPlaceRecognitionDBHandler:

    def __init__(self, db_path, odometry_dir_path):
        
        odometry_suffix = '_odometry_camera.json'

        self.gallery_db = h5py.File(db_path, 'r')
        self.gallery_len = len(self.gallery_db)

        img_paths = []
        descriptors = []
        odometries = []

        odometry_dir_path = Path(odometry_dir_path)
        img_names = self.gallery_db.keys()
        for img_name in img_names:
            img_attrs = self.gallery_db[img_name]
            img_descriptor = img_attrs['global_descriptor']

            odometry_path = str(odometry_dir_path / Path(img_name).stem) + odometry_suffix
            with open(odometry_path, 'r') as f:
                odometry = f.read()

            img_paths.append(str(odometry_dir_path / Path(img_name)))
            descriptors.append(img_descriptor)
            odometries.append(odometry)

        self.img_paths = np.array(img_paths)
        self.descriptors = np.array(descriptors)
        self.odometries = np.array(odometries)

    def get_descriptors(self):
        return self.descriptors

    def get_filepaths(self):
        return self.img_paths

    def get_by_filename(self, filename):
        idx = np.argwhere(self.img_paths == filename).flatten()[0]
        return self.descriptors[idx], self.odometries[idx]

    def get_by_idx(self, idx):
        return self.img_paths[idx], self.descriptors[idx], self.odometries[idx]


