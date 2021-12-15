import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import cv2
import json

from cv_bridge import CvBridge
from rosidl_runtime_py.set_message import set_message_fields

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String, Header

from visual_localization_interfaces.msg import KBestPRMatches

from visual_robot_localization.hloc_models import FeatureExtractor 
from visual_robot_localization.place_recognition_querier import PlaceRecognitionQuerier
import hloc.extract_features

class PlaceRecognizer(Node):
    def __init__(self):
        super().__init__('place_recognition')

        self.declare_parameter("camera_topic", "/carla/ego_vehicle/rgb_front/image")
        camera_topic_name = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.declare_parameter("pose_publish_topic", "/carla/ego_vehicle/place_reg_loc")
        pose_publish_topic = self.get_parameter('pose_publish_topic').get_parameter_value().string_value

        self.declare_parameter("extractor_conf", "dir")
        extractor_conf = self.get_parameter('extractor_conf').get_parameter_value().string_value

        self.declare_parameter("gallery_db_path", "/image-gallery/example_dir/outputs/test/global-feats-dir.h5")
        gallery_db_path = self.get_parameter('gallery_db_path').get_parameter_value().string_value

        self.declare_parameter("image_gallery_path", "/image-gallery/example_dir/")
        image_gallery_path = self.get_parameter('image_gallery_path').get_parameter_value().string_value

        self.declare_parameter("localization_frequence", 1.0)
        self.pr_freq = self.get_parameter('localization_frequence').get_parameter_value().double_value

        self.declare_parameter("top_k_matches", 4)
        self.top_k = self.get_parameter('top_k_matches').get_parameter_value().integer_value

        self.subscription = self.create_subscription(
            Image,
            camera_topic_name,
            self.callback,
            10)

        self.cv_bridge = CvBridge()
        self.latest_saved_time = None
        
        self.publisher = self.create_publisher(KBestPRMatches, pose_publish_topic, 10)

        extractor_confs = hloc.extract_features.confs.copy()
        extractor = FeatureExtractor(extractor_confs[extractor_conf])

        self.pr_querier = PlaceRecognitionQuerier(extractor, gallery_db_path, image_gallery_path)      
  
    def callback(self, image_msg):

        timestamp = image_msg.header.stamp
        time = timestamp.sec + timestamp.nanosec*(1e-9)
        if self.latest_saved_time is not None:
            time_diff = time - self.latest_saved_time
        else:
            time_diff = None

        if (time_diff is None) or (time_diff >= self.pr_freq):
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            ksmallest_filenames, ksmallest_dist, ksmallest_odometries = self.pr_querier.match(cv2_img, self.top_k)

            poses = []
            for odometry in ksmallest_odometries:
                odometry_json = json.loads(odometry)
                pose = Pose()
                set_message_fields(pose, odometry_json['pose']['pose'])

                poses.append(pose)

            header = Header(stamp=timestamp, frame_id=odometry_json['header']['frame_id'])

            kbm_msg = KBestPRMatches(header=header,
                        filenames=[String(data=filename) for filename in ksmallest_filenames],
                        locations=PoseArray(header=header, poses=poses))

            self.publisher.set_data(kbm_msg)
            self.latest_saved_time = time

def main(args=None):
    rclpy.init(args=args)

    executor = SingleThreadedExecutor()
    place_recognizer = PlaceRecognizer()
    executor.add_node(place_recognizer)
    executor.spin()

    place_recognizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
