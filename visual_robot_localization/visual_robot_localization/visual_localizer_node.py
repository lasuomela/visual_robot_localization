import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from cv_bridge import CvBridge
from rosidl_runtime_py.set_message import set_message_fields
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry

from visual_localization_interfaces.msg import VisualPoseEstimate

import cv2
import numpy as np
import transforms3d as t3d
import threading
from copy import deepcopy
import threading

from visual_robot_localization.visual_6dof_localize import VisualPoseEstimator
from visual_robot_localization.coordinate_transforms import colmap2ros_coord_transform, SensorOffsetCompensator


class VisualLocalizer(Node):
    def __init__(self):
        super().__init__('visual_localization_node')

        self.declare_parameter("pose_publish_topic", "/carla/ego_vehicle/visual_pose_estimate")
        pose_publish_topic = self.get_parameter('pose_publish_topic').get_parameter_value().string_value

        self.declare_parameter("camera_topic", "/carla/ego_vehicle/rgb_front/image")
        camera_topic_name = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.declare_parameter("global_extractor_name", "netvlad")
        global_extractor_name = self.get_parameter('global_extractor_name').get_parameter_value().string_value

        self.declare_parameter("local_extractor_name", "superpoint_aachen")
        local_extractor_name = self.get_parameter('local_extractor_name').get_parameter_value().string_value

        self.declare_parameter("local_matcher_name", "superglue")
        local_matcher_name = self.get_parameter('local_matcher_name').get_parameter_value().string_value

        self.declare_parameter("gallery_global_descriptor_path", "/image-gallery/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5")
        gallery_global_descriptor_path = self.get_parameter('gallery_global_descriptor_path').get_parameter_value().string_value

        self.declare_parameter("gallery_local_descriptor_path", "/image-gallery/example_dir/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5")
        gallery_local_descriptor_path = self.get_parameter('gallery_local_descriptor_path').get_parameter_value().string_value

        self.declare_parameter("image_gallery_path", "/image-gallery/example_dir/")
        image_gallery_path = self.get_parameter('image_gallery_path').get_parameter_value().string_value

        self.declare_parameter("gallery_sfm_path", "/image-gallery/example_dir/outputs/sfm_netvlad+superpoint_aachen+superglue/")
        gallery_sfm_path = self.get_parameter('gallery_sfm_path').get_parameter_value().string_value

        self.declare_parameter("localization_frequence", 2.0)
        self.pr_freq = self.get_parameter('localization_frequence').get_parameter_value().double_value

        self.declare_parameter("top_k_matches", 4)
        self.top_k = self.get_parameter('top_k_matches').get_parameter_value().integer_value

        self.declare_parameter("compensate_sensor_offset", True)
        self.compensate_sensor_offset = self.get_parameter('compensate_sensor_offset').get_parameter_value().bool_value

        self.declare_parameter("base_frame", "ego_vehicle")
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.declare_parameter("sensor_frame", "ego_vehicle/rgb_front")
        sensor_frame = self.get_parameter('sensor_frame').get_parameter_value().string_value

        self.declare_parameter("ransac_thresh", 12)
        ransac_thresh = self.get_parameter('ransac_thresh').get_parameter_value().integer_value

        self.publisher = self.create_publisher(VisualPoseEstimate, pose_publish_topic, 10)

        self.subscription = self.create_subscription(
            Image,
            camera_topic_name,
            self.camera_subscriber_callback,
            10)

        self.timer = self.create_timer( 1/self.pr_freq , self.computation_callback)

        self.lock = threading.Lock()
        self.latest_image = None
        self.cv_bridge = CvBridge()
        
        self.pose_estimator = VisualPoseEstimator(global_extractor_name,
                                                    local_extractor_name,
                                                    local_matcher_name,
                                                    image_gallery_path,
                                                    gallery_global_descriptor_path,
                                                    gallery_local_descriptor_path,
                                                    gallery_sfm_path,
                                                    ransac_thresh)

        if self.compensate_sensor_offset:
            self.get_logger().info('Constructing sensor offset compensator...')
            self.sensor_offset_compensator = SensorOffsetCompensator(base_frame, sensor_frame, True)

    def camera_subscriber_callback(self, image_msg):
        '''
        Use the camera subscriber callback only for updating the image data
        '''
        with self.lock:
            self.latest_image = image_msg

    def computation_callback(self):
        '''
        Perform the heavy computation inside the timer callback
        '''
        with self.lock:
            image_msg = deepcopy(self.latest_image)

        if image_msg is not None:
    
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            ret = self.pose_estimator.estimate_pose(cv2_img, self.top_k)

            timestamp = image_msg.header.stamp
            pose_estimate_msg = self._construct_visual_pose_msg(ret, timestamp)

            if self.compensate_sensor_offset:
                pr_poses = []
                for pose_msg in pose_estimate_msg.place_recognition_poses:
                    pr_poses.append( self.sensor_offset_compensator.remove_offset(pose_msg) )
                pose_estimate_msg.place_recognition_poses = pr_poses

                for pose_msg in pose_estimate_msg.pnp_estimates:
                    pose_msg.pose = self.sensor_offset_compensator.remove_offset(pose_msg.pose)

            self.publisher.publish(pose_estimate_msg)

    def _construct_visual_pose_msg(self, ret, timestamp):

        pr_poses = [ {'position': np2point_msg(pose['tvec']),
                      'orientation': np2quat_msg(pose['qvec'])} 
                      for pose in ret['place_recognition']]

        pnp_estimates = []
        for estimate in ret['pnp_estimates']:
            estimate_msg_dict = {}
            estimate_msg_dict['success'] = { 'data': estimate['success'] }
            estimate_msg_dict['place_recognition_idx'] = [ {'data': idx} for idx in estimate['place_recognition_idx']]

            if estimate['success']:
                estimate_msg_dict['num_inliers'] = {'data': estimate['num_inliers']}
                estimate_msg_dict['pose'] = {'position': np2point_msg(estimate['tvec']),
                                             'orientation': np2quat_msg(estimate['qvec'])}
            pnp_estimates.append(estimate_msg_dict)

        pose_dict = {'header': { 'stamp': timestamp, 'frame_id': 'map'},
                        'place_recognition_poses': pr_poses,
                        'pnp_estimates': pnp_estimates
                    }

        vpe_msg = VisualPoseEstimate()
        set_message_fields(vpe_msg, pose_dict)
        return vpe_msg

    @staticmethod
    def choose_best_estimate(visual_pose_estimate_msg):
        # Choose the pose estimate based on the number of ransac inliers
        best_inliers = 0
        best_estimate = None
        best_idx = None
        for i, estimate in enumerate(visual_pose_estimate_msg.pnp_estimates):
            if estimate.num_inliers.data > best_inliers:
                best_estimate = estimate.pose
                best_idx = i

        if best_estimate is None:
            best_estimate = visual_pose_estimate_msg.place_recognition_poses[0]

        return best_estimate, best_idx

def np2point_msg(np_point):
    msg_point = Point(x=np_point[0], 
                    y=np_point[1], 
                    z=np_point[2])
    return msg_point

def np2quat_msg(np_quat):
    msg_quat = Quaternion(w=np_quat[0],
                        x=np_quat[1],
                        y=np_quat[2],
                        z=np_quat[3])
    return msg_quat

def main(args=None):

    rclpy.init(args=args)

    try:
        localizer = VisualLocalizer()
        executor = SingleThreadedExecutor()
        executor.add_node(localizer)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        localizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
