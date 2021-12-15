from visual_robot_localization.gallery_db import HlocPlaceRecognitionDBHandler
import os
def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))

    db_path = script_dir+'/example_dir/outputs/test/global-feats-netvlad.h5'
    odometry_dir_path = script_dir+'/example_dir'

    a = HlocPlaceRecognitionDBHandler(db_path, odometry_dir_path)
    descriptor, odometry = a.get_by_filename(odometry_dir_path+'/im_0001.png')
    print(descriptor)
    print(odometry)

if __name__ == '__main__':
    main()
