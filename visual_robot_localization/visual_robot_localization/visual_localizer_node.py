import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from cv_bridge import CvBridge

from std_msgs.msg import ColorRGBA, Header, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseWithCovariance, Vector3
from visualization_msgs.msg import Marker

from visual_localization_interfaces.msg import VisualPoseEstimate

import cv2
from copy import deepcopy
import threading
import numpy as np
from matplotlib import cm

from visual_robot_localization.visual_6dof_localize import VisualPoseEstimator
from visual_robot_localization.coordinate_transforms import SensorOffsetCompensator


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

        self.declare_parameter("align_camera_frame", True)
        align_camera_frame = self.get_parameter('align_camera_frame').get_parameter_value().bool_value

        self.declare_parameter("ransac_thresh", 12)
        ransac_thresh = self.get_parameter('ransac_thresh').get_parameter_value().integer_value

        self.declare_parameter("visualize_estimates", False)
        self.visualize_estimates = self.get_parameter('visualize_estimates').get_parameter_value().bool_value

        self.vloc_publisher = self.create_publisher(
            VisualPoseEstimate,
            pose_publish_topic,
            10)

        # Visualization publishers
        if self.visualize_estimates:
            self.place_recognition_publisher = self.create_publisher(Marker, '/place_recognition_visualization', 10)
            self.pnp_estimate_publisher = self.create_publisher(PoseArray, '/visual_pose_estimate_visualization', 10)
            self.colormap = cm.get_cmap('Accent')

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
            self.sensor_offset_compensator = SensorOffsetCompensator(base_frame, sensor_frame, align_camera_frame)

        loc_var = 0.1
        loc_cov = 0.0
        or_var = 0.1
        or_cov = 0.0
        self.vloc_estimate_covariance =    [loc_var,    loc_cov,    loc_cov,    0.0,    0.0,    0.0,
                                            loc_cov,    loc_var,    loc_cov,    0.0,    0.0,    0.0,
                                            loc_cov,    loc_cov,    loc_var,    0.0,    0.0,    0.0,
                                            0.0,    0.0,    0.0,    or_var,    or_cov,    or_cov,
                                            0.0,    0.0,    0.0,    or_cov,    or_var,    or_cov,
                                            0.0,    0.0,    0.0,    or_cov,    or_cov,    or_var]
        self.vloc_estimate_covariance = np.array(self.vloc_estimate_covariance)

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

            computation_start_time = self.get_clock().now()
    
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            ret = self.pose_estimator.estimate_pose(cv2_img, self.top_k)

            if self.compensate_sensor_offset:
                # Compute the pose of the vehicle the camera is attached to
                for pose in ret['place_recognition']:
                    pose['tvec'], pose['qvec'] = self.sensor_offset_compensator.remove_offset_from_array(pose['tvec'], pose['qvec'])

                for pose in ret['pnp_estimates']:
                    if pose['success']:
                        pose['tvec'], pose['qvec'] = self.sensor_offset_compensator.remove_offset_from_array(pose['tvec'], pose['qvec'])
            
            best_estimate, best_cluster_idx = self.choose_best_estimate(ret)

            vloc_computation_delay = (self.get_clock().now()-computation_start_time)
            visual_pose_estimate_msg = self._construct_visual_pose_msg(best_estimate, image_msg.header.stamp, vloc_computation_delay)
            
            self.vloc_publisher.publish(visual_pose_estimate_msg)

            if self.visualize_estimates:
                self._estimate_visualizer(ret, image_msg.header.stamp, best_cluster_idx)


    def _construct_visual_pose_msg(self, best_estimate, timestamp, vloc_computation_delay):

        if best_estimate is not None:
            best_pose_msg = PoseWithCovariance(pose=Pose(position=np2point_msg(best_estimate['tvec']),
                                                        orientation=np2quat_msg(best_estimate['qvec'])),
                                                covariance = self.vloc_estimate_covariance)
            pnp_success = Bool(data=True)
        else:
            best_pose_msg = PoseWithCovariance()
            pnp_success = Bool(data=False)

        visual_pose_estimate_msg = VisualPoseEstimate(header=Header(frame_id='map', stamp=timestamp),
                                                    pnp_success = pnp_success,
                                                    computation_delay=vloc_computation_delay.to_msg(),
                                                    pose=best_pose_msg)
        return visual_pose_estimate_msg


    def choose_best_estimate(self, visual_pose_estimates):
        # Choose the pose estimate based on the number of ransac inliers
        best_inliers = 0
        best_estimate = None
        best_idx = None
        for i, estimate in enumerate(visual_pose_estimates['pnp_estimates']):
            if estimate['success']:
                if estimate['num_inliers'] > best_inliers:
                    best_inliers = estimate['num_inliers']
                    best_estimate = estimate
                    best_idx = i

        return best_estimate, best_idx


    def _estimate_visualizer(self, ret, timestamp, best_pose_idx):
        # Place recognition & PnP localization visualizations
        header = Header(frame_id='map', stamp=timestamp)
        marker = Marker(header=header, scale=Vector3(x=1.0,y=1.0,z=1.0), type=8, action=0, color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
        poses = PoseArray(header=header, poses=[])
        for i, estimate in enumerate(ret['pnp_estimates']):
            if i == best_pose_idx:
                color = self.colormap(0)
            else:
                color = self.colormap(i+1)
            color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

            place_recognition_idxs = estimate['place_recognition_idx']
            for idx in place_recognition_idxs:
                marker.colors.append(color)

                place_reg_position = np2point_msg(ret['place_recognition'][idx]['tvec'])
                marker.points.append(place_reg_position)

            if estimate['success']:
                pose_msg = Pose(position=np2point_msg(estimate['tvec']), orientation=np2quat_msg(estimate['qvec']))
                poses.poses.append(pose_msg)

        self.place_recognition_publisher.publish(marker)
        self.pnp_estimate_publisher.publish(poses)


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
