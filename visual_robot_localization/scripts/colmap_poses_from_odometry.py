import glob
import os
import json
import numpy as np
from distutils.util import strtobool
from pathlib import Path
import re

import sys
sys.path.append('/opt/third_party/colmap/colmap/scripts/python')
import read_write_model as rw

from visual_robot_localization.coordinate_transforms import ros2colmap_coord_transform

def write_intrinsics(cam_spec_path, cam_id, im_size_x, im_size_y, fov):
    # Build the K projection matrix:
    # K = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]

    focal = im_size_x / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = im_size_x / 2.0
    K[1, 2] = im_size_y / 2.0

    cam_spec = '{} SIMPLE_PINHOLE {} {} {} {} {}'.format(cam_id, im_size_x, im_size_y, (K[0,0]), (K[0,2]), (K[1,2]))

    with open(cam_spec_path, 'w+') as cam_spec_file:
        cam_spec_file.write(cam_spec)


def create_empty_colmap(img_dir, output_dir, im_size_x, im_size_y, fov, odometry_suffix, coord_transform):
    cam_id=1
    im_types = ['.png', '.jpg', '.jpeg']

    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    
    if not os.path.exists(output_dir):
    	os.makedirs(str(output_dir))

    with open(output_dir / 'points3D.txt', 'w+') as f:
        pass

    write_intrinsics(output_dir / 'cameras.txt', cam_id, im_size_x, im_size_y, fov)

    imgs = []
    for im_type in im_types:
        imgs = imgs + glob.glob(str(img_dir / '*{}'.format(im_type)))
    imgs = sorted(imgs)

    re_string = '('
    for i, im_type in enumerate(im_types):
        if i != 0:
            re_string = re_string + '|'
        re_string = re_string + im_type
    re_string = re_string + ')'
    pattern = re.compile(re_string)

    with open(output_dir / 'images.txt', 'w+') as im_list_file:

        for i, img_path in enumerate(imgs):
            odom_path = pattern.sub(odometry_suffix, img_path)

            with open(odom_path, 'r') as odom_file:
                odom = json.load(odom_file)

            pos = odom['pose']['pose']['position']
            orient = odom['pose']['pose']['orientation']

            if coord_transform:
                tx_c, ty_c, tz_c, qw_c, qx_c, qy_c, qz_c = ros2colmap_coord_transform(pos['x'],
                                                                                    pos['y'],
                                                                                    pos['z'],
                                                                                    orient['w'],
                                                                                    orient['x'],
                                                                                    orient['y'],
                                                                                    orient['z'])
            else:
                tx_c, ty_c, tz_c, qw_c, qx_c, qy_c, qz_c =  pos['x'], \
                                                            pos['y'], \
                                                            pos['z'], \
                                                            orient['w'], \
                                                            orient['x'], \
                                                            orient['y'], \
                                                            orient['z']

            im_list_file.write('{} {} {} {} {} {} {} {} {} {}\n'.format(i+1,
                                                                        qw_c,
                                                                        qx_c,
                                                                        qy_c,
                                                                        qz_c,
                                                                        tx_c,
                                                                        ty_c,
                                                                        tz_c,
                                                                        cam_id,
                                                                        Path(img_path).name))
            im_list_file.write('\n')

    points = rw.read_points3D_text(output_dir / 'points3D.txt')
    rw.write_points3D_binary(points, output_dir / 'points3D.bin')

    cams = rw.read_cameras_text(output_dir / 'cameras.txt')
    rw.write_cameras_binary(cams, output_dir / 'cameras.bin')

    imgs = rw.read_images_text(output_dir / 'images.txt')
    rw.write_images_binary(imgs, output_dir / 'images.bin')


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Build an empty Colmap model based on ROS2 odometry poses')
    parser.add_argument('--image_dir', type=str, help='The directory to load images from')
    parser.add_argument('--output_dir', type=str, help='The directory to load images from')

    parser.add_argument('--im_size_x', type=int, default = 800 )
    parser.add_argument('--im_size_y', type=int, default = 600 )
    parser.add_argument('--fov', type=int, default = 90 )

    parser.add_argument('--odometry_suffix', type=str, default='_odometry_camera.json')
    parser.add_argument('--coord_transform', dest='coord_transform', type=lambda x: bool(strtobool(x)), default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    create_empty_colmap(args.image_dir,
         args.im_size_x,
         args.im_size_y,
         args.fov,
         args.odometry_suffix,
         args.coord_transform)
