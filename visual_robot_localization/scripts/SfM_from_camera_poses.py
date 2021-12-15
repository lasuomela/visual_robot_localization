from pathlib import Path
import glob
import os
import shutil
import torch

from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

from visual_robot_localization.gallery_db import HlocPlaceRecognitionDBHandler
from colmap_poses_from_odometry import create_empty_colmap

def create_place_recognition_pairs(odometry_dir_path, descriptor_file_path, n_matches, output_path):

    db_handler = HlocPlaceRecognitionDBHandler(descriptor_file_path, odometry_dir_path)

    image_names = db_handler.get_filepaths()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_descriptors = torch.from_numpy(db_handler.get_descriptors()).to(device)

    similarity = torch.einsum('id,jd->ij', image_descriptors, image_descriptors)
    topk = torch.topk(similarity, n_matches, dim=1).indices.cpu().numpy()

    pairs = []
    for img_name, idxs in zip(image_names, topk):
        for i in idxs:
            img_name_parsed = img_name.split('/')[-1]
            img2_name_parsed = image_names[i].split('/')[-1]

            pair = (img_name_parsed, img2_name_parsed)
            pairs.append(pair)

    with open(output_path, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

def main(img_dir,
        global_extractor_name,
        local_extractor_name,
        local_matcher_name,
        n_matches,
        im_size_x,
        im_size_y,
        im_fov):

    img_dir =  Path(img_dir)
    assert img_dir.exists(), img_dir

    img_types = ['.png', '.jpg', '.jpeg']
    odometry_suffix = '_odometry_camera.json'
    coord_transform = True

    output_dir = img_dir / 'outputs' / '{}+{}+{}'.format(global_extractor_name,
                                                        local_extractor_name,
                                                        local_matcher_name)
    place_reg_pair_path = output_dir / 'pairs.txt'
    empty_sfm = output_dir / 'sfm_empty'

    extractor_confs = extract_features.confs.copy()
    matcher_confs = match_features.confs.copy()

    # Manually add r2d2 extractor configuration since the hloc pr hasn't been merged yet
    # https://github.com/cvg/Hierarchical-Localization/pull/85/commits
    extractor_confs['r2d2'] = {
        'output': 'feats-r2d2-n5000-r1024',
        'model':{
            'name': 'r2d2',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    }

    assert (global_extractor_name in extractor_confs), global_extractor_name
    assert (local_extractor_name in extractor_confs), local_extractor_name
    assert (local_matcher_name in matcher_confs), local_matcher_name

    # sfm output path
    output_sfm = output_dir / 'sfm_{}+{}+{}'.format(global_extractor_name,
                                                    local_extractor_name,
                                                    local_matcher_name)
    if output_sfm.exists() and output_sfm.is_dir():
        shutil.rmtree(output_sfm)

    # List the images to process to avoid including images in subdirectories
    img_list = []
    for img_type in img_types:
        img_list = img_list + glob.glob(str(img_dir / '*{}'.format(img_type)))
    img_list = list(map(os.path.basename, img_list))
    img_list.sort()

    print('Extracting global features...')
    global_feature_path = extract_features.main(extractor_confs[global_extractor_name], img_dir, output_dir, as_half=False, image_list = img_list)

    create_place_recognition_pairs(str(img_dir), str(global_feature_path), n_matches, str(place_reg_pair_path))

    create_empty_colmap(str(img_dir), str(empty_sfm), im_size_x, im_size_y, im_fov, odometry_suffix, coord_transform)

    print('Extracting local features...')
    feature_path = extract_features.main(extractor_confs[local_extractor_name], img_dir, output_dir, as_half=False, image_list = img_list)

    print('Matching local features...')
    sfm_match_path = match_features.main(matcher_confs[local_matcher_name], place_reg_pair_path, extractor_confs[local_extractor_name]['output'], export_dir=output_dir)

    print('Triangulating point matches...')
    triangulation.main(
        output_sfm,
        empty_sfm,
        img_dir,
        place_reg_pair_path,
        feature_path,
        sfm_match_path,
        colmap_path='colmap')  # change if COLMAP is not in your PATH


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Build a sparse Colmap model based on ROS2 odometry poses')
    parser.add_argument('--image_dir', type=str, help='The directory to load images from', default='/image-gallery/example_dir')
    parser.add_argument('--global_extractor_name', type=str, help='Name of the global extractor, as specified in hloc.extract_features.extractor_confs')
    parser.add_argument('--local_extractor_name', type=str, help='Name of the local extractor, as specified in hloc.extract_features.confs')
    parser.add_argument('--local_matcher_name', type=str, help='Name of the local matcher, as specified in hloc.match_features.confs')
    parser.add_argument('--n_matches', type=int, help='Number of best matching images (as identified by place recognition) from which to look for local feature matches')
    parser.add_argument('--im_size_x', type=int, help='Horizontal image size')
    parser.add_argument('--im_size_y', type=int, help='Vertical image size')
    parser.add_argument('--im_fov', type=int, help='Camera field of view in degrees')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args.image_dir,
        args.global_extractor_name,
        args.local_extractor_name,
        args.local_matcher_name,
        args.n_matches,
        args.im_size_x,
        args.im_size_y,
        args.im_fov)
