import launch
import launch_ros.actions

import pytest
import unittest
import launch_testing
import rclpy
import sys
import os

@pytest.mark.rostest
def generate_test_description():
    path_to_test = os.path.dirname(__file__)
    print(path_to_test)
    test_node = launch_ros.actions.Node(
            executable=sys.executable,
            arguments=[ path_to_test + '/visual_localizer_node_test_node.py'],
            output='screen',
            emulate_tty=True,
            parameters=[
            {
            	'image_publish_topic': launch.substitutions.LaunchConfiguration('camera_topic')
            },
            {
            	'visual_pose_estimate_subscription_topic': launch.substitutions.LaunchConfiguration('pose_publish_topic')
            },
            {
            	'test_image_path': launch.substitutions.LaunchConfiguration('test_image_path')
    	    }
    	    ]
            )
            
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
	name='test_image_path',
    	default_value='/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/im_0001.png'
        ),
        launch.actions.DeclareLaunchArgument(
	name='pose_publish_topic',
    	default_value='/carla/ego_vehicle/visual_pose_estimate'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='camera_topic',
    	default_value='/carla/ego_vehicle/rgb_front/image'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='base_frame',
    	default_value='ego_vehicle'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='sensor_frame',
    	default_value='ego_vehicle/rgb_front'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='global_extractor_name',
    	default_value='netvlad'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='local_extractor_name',
    	default_value='superpoint_aachen'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='local_matcher_name',
    	default_value='superglue'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_global_descriptor_path',
    	default_value='/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_local_descriptor_path',
    	default_value='/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='image_gallery_path',
    	default_value='/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_sfm_path',
    	default_value='/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/outputs/netvlad+superpoint_aachen+superglue/sfm_netvlad+superpoint_aachen+superglue'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='compensate_sensor_offset',
    	default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='localization_frequence',
    	default_value='2.0'
        ),
        
        launch.actions.DeclareLaunchArgument(
    	name='use_sim_time',
    	default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='top_k_matches',
    	default_value='3'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='ransac_thresh',
    	default_value='12'
        ),
        launch_ros.actions.Node(
            package='visual_robot_localization',
            executable='visual_localizer_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {    # If the localization frequence is in ROS or wall time
    		     'use_sim_time': False
                },
                {
                    'pose_publish_topic': '/carla/ego_vehicle/visual_pose_estimate'
                },
                {
                    'camera_topic': '/carla/ego_vehicle/rgb_front/image'
                },
                {
                    'global_extractor_name': 'netvlad'
                },
                {
                    'local_extractor_name': 'superpoint_aachen'
                },
                {
                    'local_matcher_name': 'superglue'
                },
                {
                    'gallery_global_descriptor_path': path_to_test + '/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5'
                },
                {
                    'gallery_local_descriptor_path': path_to_test + '/example_dir/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5'
                },
                {
                    'image_gallery_path':  path_to_test + '/example_dir/'
                },
                {
                    'gallery_sfm_path': path_to_test + '/example_dir/outputs/netvlad+superpoint_aachen+superglue/sfm_netvlad+superpoint_aachen+superglue'
                },
                {
                    'compensate_sensor_offset': False
                },
                {
                    'localization_frequence': 2.0
                },
                {
                    'top_k_matches': 4
                },
                {
                    'ransac_thresh': 12
                }
            ]
        ),
        test_node,
        launch.actions.RegisterEventHandler(
            launch.event_handlers.OnProcessExit(
                target_action=test_node,
                on_exit=[
                    launch.actions.LogInfo(msg='Test completed'),
                    launch.actions.EmitEvent(event=launch.events.Shutdown(
                        reason='Test completed\n'))
                ]
            )
        ),
        launch_testing.actions.ReadyToTest(),
    ])
    return ld

class TestShit(unittest.TestCase):
        
    def test_node_start(self, proc_output):
        rclpy.init()
        msg = 'Got Result!'
        proc_output.assertWaitFor(msg, process=None, stream='stdout', timeout=20)
        rclpy.shutdown()

