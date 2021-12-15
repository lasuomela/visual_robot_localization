import os
import glob
import h5py
import cv2
import matplotlib.cm as cm
from pathlib import Path
import numpy as np

import hloc.extract_features
import hloc.match_features

from visual_robot_localization.hloc_models import FeatureExtractor, FeatureMatcher
from third_party.SuperGluePretrainedNetwork.models.utils import make_matching_plot

def main(test_img1_name, test_img2_name, test_img_dir, use_superglue, create_visualizations):

    script_dir = os.path.dirname(os.path.realpath(__file__))

    test_img_dir = Path(script_dir + test_img_dir)
    output_dir = test_img_dir / 'outputs/test'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_types = ['.png', '.jpg', '.jpeg']
    extractor_confs = hloc.extract_features.confs.copy()

    # List all the images in the main folder
    img_list = []
    for img_type in img_types:
        img_list = img_list + glob.glob(str(test_img_dir / '*{}'.format(img_type)))
    img_list = list(map(os.path.basename, img_list))
    img_list.sort()

    match_dict = {}
    diff_dict = {}
    for key, extractor_conf in extractor_confs.items():

        # Do batch extraction using hloc
        batch_descriptor_path = output_dir / '{}.h5'.format(extractor_conf['output'])
        try:
            os.remove(batch_descriptor_path)
        except OSError:
            pass

        hloc.extract_features.main( extractor_conf, test_img_dir, output_dir, as_half=False, image_list = img_list, feature_path = batch_descriptor_path)
        batch_descriptors = h5py.File(batch_descriptor_path, 'r')

        # Do single image extraction
        extractor = FeatureExtractor(extractor_conf)

        # Check similarity between individual and hloc batch extraction
        match_dict[key] = True
        differences = [] 
        for img_name in img_list:
            if ('grayscale' in extractor_conf['preprocessing']) & (extractor_conf['preprocessing']['grayscale']):
                # Read images in grayscale to see if features match exactly
                img = cv2.imread(str(test_img_dir / img_name), cv2.IMREAD_GRAYSCALE)
                single_descriptor = extractor(img, disable_grayscale = True)
            else:
                img = cv2.imread(str(test_img_dir / img_name), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                    
                single_descriptor = extractor(img)

            if 'descriptors' in single_descriptor:
                single_descriptor = single_descriptor['descriptors'].cpu().numpy()
            elif 'global_descriptor' in single_descriptor:
                single_descriptor = single_descriptor['global_descriptor'].numpy()
            single_descriptor = single_descriptor.squeeze()

            batch_img = batch_descriptors[img_name]
            if 'descriptors' in batch_img:
                batch_descriptor = batch_img['descriptors']
            elif 'global_descriptor' in batch_img:
                batch_descriptor = batch_img['global_descriptor']
            
            if single_descriptor.shape == batch_descriptor.shape:
                diff = (single_descriptor - batch_descriptor)
                same = np.allclose(single_descriptor, batch_descriptor)
                if not same:
                    avg_diff_prct = np.mean(np.linalg.norm(diff)/max(np.linalg.norm(single_descriptor), np.linalg.norm(batch_descriptor)))*100
                    reason = 'Value mismatch, avg difference {}%'.format(avg_diff_prct)
                    differences.append(avg_diff_prct)
            else:
                same = False
                reason = 'Shape mismatch'

            if not same:
                print('Discrepancy between batch and individually extracted descriptors for {} on image {}'.format(key, img_name))
                print('Reason: {}'.format(reason))
                match_dict[key] = False

        if match_dict[key]:
            print('Features match')
            print('\n')
        else:
            if differences:
                diff_dict[key] = np.mean(differences)
            print('\n')

    print('\nBatch and individually extracted descriptors match:')
    for key,v in match_dict.items():
        if key in diff_dict:
            print('{}: {}, avg diff {}%'.format(key,v,diff_dict[key]))
        else:
            print('{}: {}'.format(key,v))
    print("It's likely that discrepancy below 0.02% is insignificant since\nthat is the magnitude of error caused by comparing fp32 values with same values rounded to fp16\n")

    # Visualize keypoint matches for local extractors
    if create_visualizations:
        print('Creating visuals')
        for key, extractor_conf in extractor_confs.items():

            extractor = FeatureExtractor(extractor_conf)

            if ('grayscale' in extractor_conf['preprocessing']) & (extractor_conf['preprocessing']['grayscale']):
                # Read images in grayscale to see if features match exactly
                img1 = cv2.imread(str(test_img_dir / test_img1_name), cv2.IMREAD_GRAYSCALE)
                res1 = extractor.extract(img1, disable_grayscale = True)

                img2 = cv2.imread(str(test_img_dir / test_img2_name), cv2.IMREAD_GRAYSCALE)
                res2 = extractor.extract(img2, disable_grayscale = True)
            else:
                img1 = cv2.imread(str(test_img_dir / test_img1_name), cv2.IMREAD_COLOR)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)                    
                res1 = extractor.extract(img1)

                img2 = cv2.imread(str(test_img_dir / test_img2_name), cv2.IMREAD_COLOR)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)                    
                res2 = extractor.extract(img2)

            if 'keypoints' in res1:
                visualize_keypoint_matches(key, extractor_conf, res1, res2, img1, img2, use_superglue, output_dir)

def visualize_keypoint_matches(key, extractor_conf, desc1, desc2, img1, img2, use_superglue, output_dir):

    matcher_confs = hloc.match_features.confs
    matcher_common = FeatureMatcher(matcher_confs['NN-mutual'])
    if use_superglue:
        matcher_sp = FeatureMatcher(matcher_confs['superglue'])
    else:
        matcher_sp = FeatureMatcher(matcher_confs['NN-superpoint'])

    # Visualize matches
    if extractor_conf['model']['name'] == 'superpoint':
        matcher = matcher_sp
    else:
        matcher = matcher_common

    data = matcher.prepare_data(desc1, desc2, img1.shape)
    matches, scores = matcher(data)
    
    kpts0 = data['keypoints0'].cpu().numpy()
    kpts1 = data['keypoints1'].cpu().numpy()
    conf=scores

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    color = cm.jet(mconf)
    text = [
        extractor_conf['model']['name'],
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    save_path = output_dir / 'test_{}.png'.format(key)
    
    make_matching_plot(img1, img2, kpts0, kpts1, mkpts0, mkpts1,
                    color, text, str(save_path), show_keypoints=True,
                    fast_viz=False, opencv_display=False,
                    opencv_title='matches', small_text=[])


def parse_arguments():
    import argparse
    from distutils.util import strtobool
    parser = argparse.ArgumentParser(description='CLI test for hloc feature extraction and matching')
    parser.add_argument('--test_img1_name', type=str, help='Image name 1 for visualization', default = 'im_0001.png')
    parser.add_argument('--test_img2_name', type=str, help='Image name 2 for visualization', default = 'im_0002.png')
    parser.add_argument('--test_img_dir', type=str, help='Directory containing the test images', default = '/example_dir/')
    parser.add_argument('--use_superglue', dest='use_superglue', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--create_visualizations', dest='create_visualizations', type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args.test_img1_name, args.test_img2_name, args.test_img_dir, use_superglue=args.use_superglue, create_visualizations=args.create_visualizations)
