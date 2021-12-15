import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor 
import launch.actions
import launch.events

from cv_bridge import CvBridge, CvBridgeError
from rosidl_runtime_py.convert import message_to_ordereddict

from sensor_msgs.msg import Image

import cv2
import json

from visual_localization_interfaces.msg import VisualPoseEstimate

class VisualLocalizerTestNode(Node):
    def __init__(self):
        super().__init__('visual_localizer_test_node')

        self.declare_parameter("image_publish_topic", '/carla/ego_vehicle/rgb_front/image')
        image_publish_topic = self.get_parameter('image_publish_topic').get_parameter_value().string_value

        self.declare_parameter("visual_pose_estimate_subscription_topic", '/carla/ego_vehicle/visual_pose_estimate')
        visual_pose_estimate_subscription_topic = self.get_parameter('visual_pose_estimate_subscription_topic').get_parameter_value().string_value

        self.declare_parameter("test_image_path", '/opt/visual_robot_localization/src/visual_robot_localization/test/example_dir/im_0001.png')
        test_image_path = self.get_parameter('test_image_path').get_parameter_value().string_value

        self.cv_bridge = CvBridge()
        
        self.publisher = self.create_publisher(Image, image_publish_topic, 10)

        self.subscription = self.create_subscription(
            VisualPoseEstimate,
            visual_pose_estimate_subscription_topic,
            self.result_callback,
            10)

        self.timer = self.create_timer( 3, self.timer_callback)
        
        test_img = cv2.imread(test_image_path)

        self.test_image_msg = self.cv_bridge.cv2_to_imgmsg(test_img, encoding="bgr8")
        print('Initialized visual localizer test node')

    def timer_callback(self):
        self.publisher.publish( self.test_image_msg )

    def result_callback(self, visual_pose_estimate_msg):
        print('Got Result!')
        pose = message_to_ordereddict(visual_pose_estimate_msg)
        print(json.dumps(pose, indent=3))



def main(args=None):
    rclpy.init(args=args)

    try:
        executor = SingleThreadedExecutor()
        node = VisualLocalizerTestNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
