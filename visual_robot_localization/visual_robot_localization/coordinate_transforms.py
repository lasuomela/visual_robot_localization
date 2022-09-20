import numpy as np
import transforms3d as t3d
import threading

import rclpy
from rclpy.node import Node
import tf2_ros

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

import psutil

def ros2colmap_coord_transform(tx_r, ty_r, tz_r, qw_r, qx_r, qy_r, qz_r):

    '''
    Transform coordinate and orientation from ROS world frame to Colmap projection parameters t, R
     See:  https://colmap.github.io/format.html#output-format#images-txt
     and:  https://www.ros.org/reps/rep-0103.html 
    '''

    # Switch the coordinate axes (ROS world coordinates -> Colmap world coordinates)
    tx_r_al = -ty_r
    ty_r_al = -tz_r
    tz_r_al = tx_r
    vec_r_al = np.array([tx_r_al, ty_r_al, tz_r_al])

    qw_r_al = qw_r
    qx_r_al = -qy_r
    qy_r_al = -qz_r
    qz_r_al = qx_r

    # Projection from Colmap world coordinates -> Colmap camera centered coordinate frame
    quat_r_al = t3d.quaternions.qinverse( np.array([ qw_r_al, qx_r_al, qy_r_al, qz_r_al ]))
    tx_c, ty_c, tz_c = t3d.quaternions.rotate_vector( -vec_r_al,  quat_r_al )

    qw_c =  quat_r_al[0]
    qx_c = quat_r_al[1]
    qy_c = quat_r_al[2]
    qz_c = quat_r_al[3]

    return tx_c, ty_c, tz_c, qw_c, qx_c, qy_c, qz_c

def colmap2ros_coord_transform(tvec_col, qvec_col):
    qinv = t3d.quaternions.qinverse(qvec_col)
    tvec_rot = t3d.quaternions.rotate_vector(-tvec_col, qinv)
    tvec_ros = np.array([ tvec_rot[2], -tvec_rot[0], -tvec_rot[1]])
    qvec_ros = np.array( [qinv[0], qinv[3], -qinv[1], -qinv[2] ])
    return tvec_ros, qvec_ros


class SensorOffsetCompensator:
    '''
    Get the pose of a sensor relative to vehicle base (or other frame)
    and use the information to compensate the sensor offset 
    in map coordinate frame. Basically, the conversion sensor pose -> vehicle base pose.
    '''

    def __init__(self, base_frame_name, sensor_frame_name, align_camera_frame=False):

        '''
        frame_id (str): name of the transform source frame
        child_frame_id (str): name of the transform target frame
        align_camera_frame (bool):  The camera coordinate frame convention sometimes differs from
                                    the convention for the robot base and map. This parameter 
                                    defines if a correction is used.
        '''

        tf_subscription_freq = 10
        tf_wait_timeout = 5
        self.camera_frame_alignment_qvec = np.array([ 0.5, 0.5, -0.5, 0.5])
        self.tvec, self.qvec = self._get_transform(base_frame_name, sensor_frame_name, tf_subscription_freq, tf_wait_timeout, align_camera_frame)

    def _get_transform(self, frame_id, child_frame_id, tf_subscription_freq, tf_wait_timeout, align_camera_frame):

        try:
            # If called independently
            rclpy.init()
            owns_rclpy = True
        except RuntimeError:
            # If called from inside an already initiated rclpy context
            owns_rclpy = False
            pass

        rate_node = Node('sensor_offset_remover_utility')

        tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tfBuffer, rate_node, spin_thread=True)
        rate = rate_node.create_rate(tf_subscription_freq)

        transform = None
        wait=0
        print('Waiting to acquire transform {} -> {} from tf2...'.format(frame_id, child_frame_id))
        while rclpy.ok() & (transform is None):
            if wait > tf_wait_timeout:
                print(f'Transform {frame_id} -> {child_frame_id} not received!')
                return None, None
                #raise Exception('Transform {} -> {} not received!'.format(frame_id, child_frame_id))
            else:
                try:
                    transform = tfBuffer.lookup_transform( target_frame=frame_id,
                                                    source_frame=child_frame_id,
                                                    time=rclpy.duration.Duration(seconds=0))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    #rate.sleep()
                    print("Didn't receive transform, retrying...")
                wait += 1/tf_subscription_freq


        rate_node.destroy_node()
        tf_listener.unregister()
        tf_listener.__del__()
        if owns_rclpy:
            rclpy.shutdown()

        tvec = transform.transform.translation
        tvec = np.array([tvec.x, tvec.y, tvec.z])

        qvec = transform.transform.rotation
        qvec = np.array([qvec.w, qvec.x, qvec.y, qvec.z])

        if align_camera_frame:
            # Atleast in Carla, the camera coordinate frame convention is 
            # z front, x right, y down as opposed to map frame
            # x front, y left, z up
            qvec = t3d.quaternions.qmult(qvec, self.camera_frame_alignment_qvec)

        print('Initialized sensor offset compensator {} -> {} with parameters'.format(frame_id, child_frame_id))
        print('T: {}'.format(tvec))
        print('Q: {}'.format(qvec.round(3)))
        return tvec, qvec

    def add_offset(self, sensor_pose_msg):
        '''
        sensor_pose_msg: geometry_msgs.msg.PoseStamped
        '''
        if type(sensor_pose_msg) == PoseStamped:
            position = sensor_pose_msg.pose.position
            orientation = sensor_pose_msg.pose.orientation
        elif type(sensor_pose_msg) == Pose:
            position = sensor_pose_msg.position
            orientation = sensor_pose_msg.orientation

        sensor_tvec = np.array([position.x, position.y, position.z])
        sensor_qvec = np.array([orientation.w, orientation.x, orientation.y, orientation.z])

        base_world_rot = t3d.quaternions.qmult(sensor_qvec, self.qvec)
        base_world_rot = base_world_rot/t3d.quaternions.qnorm(base_world_rot)
        base_world_pos = sensor_tvec + t3d.quaternions.rotate_vector(self.tvec, sensor_qvec)

        base_pose_msg = Pose(position=Point(x=base_world_pos[0], y=base_world_pos[1], z=base_world_pos[2]),
                            orientation=Quaternion(w=base_world_rot[0], x=base_world_rot[1], y=base_world_rot[2], z=base_world_rot[3]))

        if type(sensor_pose_msg) == PoseStamped:
            base_pose_msg = PoseStamped(header=sensor_pose_msg.header, pose=base_pose_msg)
        return base_pose_msg

    def remove_offset(self, sensor_pose_msg):
        '''
        sensor_pose_msg: geometry_msgs.msg.PoseStamped | geometry_msgs.msg.Pose
        '''
        if type(sensor_pose_msg) == PoseStamped:
            position = sensor_pose_msg.pose.position
            orientation = sensor_pose_msg.pose.orientation
        elif type(sensor_pose_msg) == Pose:
            position = sensor_pose_msg.position
            orientation = sensor_pose_msg.orientation
        else:
            raise TypeError('Message type {} not supported'.format(type(sensor_pose_msg)))

        sensor_tvec = np.array([position.x, position.y, position.z])
        sensor_qvec = np.array([orientation.w, orientation.x, orientation.y, orientation.z])

        base_world_pos, base_world_rot = self.remove_offset_from_array(sensor_tvec, sensor_qvec)
        base_pose_msg = Pose(position=Point(x=base_world_pos[0], y=base_world_pos[1], z=base_world_pos[2]),
                            orientation=Quaternion(w=base_world_rot[0], x=base_world_rot[1], y=base_world_rot[2], z=base_world_rot[3]))

        if type(sensor_pose_msg) == PoseStamped:
            base_pose_msg = PoseStamped(header=sensor_pose_msg.header, pose=base_pose_msg)
 
        return base_pose_msg

    def remove_offset_from_array(self, sensor_tvec, sensor_qvec):
        '''
        Prams:
            sensor_tvec: Position of the sensor
            type: np.array

            sensor_qvec: Orientation of the sensor
            type: np.array

        Return:
            base_world_pos: Position of the vehicle the sensor is attached to
            base_world_rot: Orientation of the vehicle the sensor is attached to
        '''

        base_world_rot = t3d.quaternions.qmult(sensor_qvec, t3d.quaternions.qinverse( self.qvec))
        base_world_rot = base_world_rot/t3d.quaternions.qnorm(base_world_rot)
        base_world_pos = sensor_tvec - t3d.quaternions.rotate_vector(self.tvec, base_world_rot)

        return base_world_pos, base_world_rot
        

def compensate_sensor_offset(odometry_msg, sensor_pos, sensor_rot):
    '''
    Legacy, prefer the SensorOffsetCompensator class if possible
    '''
    frame_world_pos = point_msg2np( odometry_msg.pose.pose.position )
    frame_world_rot = quat_msg2np( odometry_msg.pose.pose.orientation )

    sensor_world_pos = frame_world_pos + t3d.quaternions.rotate_vector(sensor_pos, frame_world_rot)
    sensor_world_rot = t3d.quaternions.qmult(frame_world_rot, sensor_rot)
    sensor_world_rot = sensor_world_rot/t3d.quaternions.qnorm(sensor_world_rot)

    sensor_odometry_msg = deepcopy(odometry_msg)
    sensor_odometry_msg.pose.pose.position = np2point_msg(sensor_world_pos)
    sensor_odometry_msg.pose.pose.orientation = np2quat_msg(sensor_world_rot)
    return sensor_odometry_msg