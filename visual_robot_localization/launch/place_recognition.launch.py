import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
	name='camera_topic',
    	default_value='/carla/ego_vehicle/rgb_front/image'
        ),
        launch.actions.DeclareLaunchArgument(
	name='pose_publish_topic',
    	default_value='/carla/ego_vehicle/place_reg_loc'
        ),
        launch.actions.DeclareLaunchArgument(
	name='place_reg_markers_topic',
    	default_value='/carla/ego_vehicle/place_reg_markers'
        ),
        launch.actions.DeclareLaunchArgument(
	name='global_extractor_name',
    	default_value='netvlad'
        ),
        launch.actions.DeclareLaunchArgument(
	name='gallery_global_descriptor_path',
    	default_value='/image-gallery/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
        ),
        launch.actions.DeclareLaunchArgument(
	name='image_gallery_path',
    	default_value='/image-gallery/example_dir/'
        ),
        launch.actions.DeclareLaunchArgument(
	name='localization_frequence',
    	default_value='2.0'
        ),
        launch.actions.DeclareLaunchArgument(
	name='top_k_matches',
    	default_value='4'
        ),
        launch_ros.actions.Node(
            package='visual_robot_localization',
            executable='place_recognition_node',
            output='screen',
            emulate_tty=True,
            parameters=[
            	{
                    'camera_topic': launch.substitutions.LaunchConfiguration('camera_topic')
                },
                {
                    'pose_publish_topic': launch.substitutions.LaunchConfiguration('pose_publish_topic')
                },
                {
                    'extractor_conf': launch.substitutions.LaunchConfiguration('global_extractor_name')
                },
                {
                    'gallery_db_path': launch.substitutions.LaunchConfiguration('gallery_global_descriptor_path')
                },
                {
                    'image_gallery_path': launch.substitutions.LaunchConfiguration('image_gallery_path')
                },
                {
                    'localization_frequence': launch.substitutions.LaunchConfiguration('localization_frequence')
                },
                {
                    'top_k_matches': launch.substitutions.LaunchConfiguration('top_k_matches')
                }
        	]
        )
        ])
    return ld
	
if __name__ == '__main__':
    generate_launch_description()
	
     
