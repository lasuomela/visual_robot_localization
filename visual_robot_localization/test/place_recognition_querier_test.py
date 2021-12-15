from visual_robot_localization.place_recognition_querier import PlaceRecognitionQuerier
from visual_robot_localization.hloc_models import FeatureExtractor
import hloc.extract_features

import os
import matplotlib.pyplot as plt
import cv2
import re
import glob
from pathlib import Path
import json

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))

    extractor_name = 'netvlad'
    gallery_global_descriptor_path = script_dir+'/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
    image_gallery_path = script_dir+'/example_dir/'
    top_k = 4

    image_types = ['.png', '.jpg', '.jpeg']

    image_gallery_path = Path(image_gallery_path)

    extractor_confs = hloc.extract_features.confs.copy()
    extractor = FeatureExtractor(extractor_confs[extractor_name])

    querier = PlaceRecognitionQuerier(extractor, gallery_global_descriptor_path, str(image_gallery_path))

    test_img_paths = []
    for extension in image_types:
        test_img_paths = test_img_paths + glob.glob(str(image_gallery_path / '*{}'.format(extension)))

    for test_img_path in test_img_paths:
        test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        ksmallest_filenames, ksmallest_dist, ksmallest_odometries = querier.match(test_img, top_k)

    visualize_matches(test_img, test_img_path, ksmallest_filenames, ksmallest_dist, ksmallest_odometries )

def visualize_matches(test_img, test_img_path, ksmallest_filenames, ksmallest_dist, ksmallest_odometries ):

    test_odom_path = re.sub(r'.png', '_odometry_camera.json', test_img_path)
    with open(test_odom_path) as json_file:
        test_odometry_json = json.load(json_file)

    x = test_odometry_json['pose']['pose']['position']['x']
    y = test_odometry_json['pose']['pose']['position']['y']

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=[18,10])
    plt.subplots_adjust(top=0.966,
                        bottom=0.014,
                        left=0.008,
                        right=0.992,
                        hspace=0.115,
                        wspace=0.0)
    axes = axes.ravel()
    axes[0].imshow(test_img)
    axes[0].set_title('Query: x: {:.2f}, y: {:.2f}'.format(x, y))
    axes[0].axis('off')

    for i, odometry in enumerate(ksmallest_odometries):
        odometry_json = json.loads(odometry)
        x = odometry_json['pose']['pose']['position']['x']
        y = odometry_json['pose']['pose']['position']['y']

        img = cv2.imread(ksmallest_filenames[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i+1].imshow(img)
        axes[i+1].set_title('Match {}: x: {:.2f}, y: {:.2f}'.format(i, x, y))
    for ax in axes:
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

