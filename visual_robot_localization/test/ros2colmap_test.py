from visual_robot_localization.coordinate_transforms import ros2colmap_coord_transform, colmap2ros_coord_transform
import json
import os
import numpy as np

def main():

    # Check ros2colmap transforms
    script_dir = os.path.dirname(os.path.realpath(__file__))

    path= script_dir + '/example_dir/im_0001_odometry_camera.json'
    with open(path, 'r') as file:
        odometry_msg = json.load(file)
    pos_msg = odometry_msg['pose']['pose']['position']
    or_msg = odometry_msg['pose']['pose']['orientation']

    tx_r, ty_r, tz_r = pos_msg['x'], pos_msg['y'], pos_msg['z']
    qw_r, qx_r, qy_r, qz_r = or_msg['w'], or_msg['x'], or_msg['y'], or_msg['z']

    tvec_ros_o = np.array([ tx_r, ty_r, tz_r ])
    qvec_ros_o = np.array([ qw_r, qx_r, qy_r, qz_r ])

    tx_c, ty_c, tz_c, qw_c, qx_c, qy_c, qz_c = ros2colmap_coord_transform(tx_r, ty_r, tz_r, qw_r, qx_r, qy_r, qz_r)

    tvec_col = np.array([ tx_c, ty_c, tz_c ])
    qvec_col = np.array([ qw_c, qx_c, qy_c, qz_c ])

    tvec_ros, qvec_ros = colmap2ros_coord_transform(tvec_col, qvec_col)

    print('T original: {}'.format(tvec_ros_o.round(4)))
    print('T transformed: {}'.format(tvec_ros.round(4)))
    print('Q original: {}'.format(qvec_ros_o.round(4)))
    print('Q transformed: {}'.format(qvec_ros.round(4)))

if __name__ == '__main__':
    main()
