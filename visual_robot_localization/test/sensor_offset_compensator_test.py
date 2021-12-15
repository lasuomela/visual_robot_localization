from visual_robot_localization.coordinate_transforms import SensorOffsetCompensator
from geometry_msgs.msg import PoseStamped

def main():
    # Check sensor offset compensation
    # Requires that the transform between the frames is published by tf2
    # -> ros-bridge is running
    a = SensorOffsetCompensator( 'ego_vehicle', 'ego_vehicle/rgb_front', True)
    m = PoseStamped()
    m.pose.position.x = 152.0
    m.pose.position.y = -133.0
    m.pose.position.z = 2.0

    m.pose.orientation.w = 0.700
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = -0.707

    mod = a.add_offset(m)
    print(mod)

    m.pose.orientation.w = 0.0
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = -1.0
    mod = a.remove_offset(m)
    print(mod)

if __name__ == '__main__':
    main()
