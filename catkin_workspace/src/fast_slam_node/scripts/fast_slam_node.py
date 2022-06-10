#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import sys
sys.path.insert(1, 'auxiliar_scripts')
from particle_filter import *
from landmark_extractor import  *
from landmark_matching import  *
import numpy as np

def euler_angle_to_quaternion(X, Y, Z):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param X: The X (rotation around x-axis) angle in radians.
    :param Y: The Y (rotation around y-axis) angle in radians.
    :param Z: The Z (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion {"x","y","z","w"} format
    """
    qx = np.sin(X/2) * np.cos(Y/2) * np.cos(Z/2) - np.cos(X/2) * np.sin(Y/2) * np.sin(Z/2)
    qy = np.cos(X/2) * np.sin(Y/2) * np.cos(Z/2) + np.sin(X/2) * np.cos(Y/2) * np.sin(Z/2)
    qz = np.cos(X/2) * np.cos(Y/2) * np.sin(Z/2) - np.sin(X/2) * np.sin(Y/2) * np.cos(Z/2)
    qw = np.cos(X/2) * np.cos(Y/2) * np.cos(Z/2) + np.sin(X/2) * np.sin(Y/2) * np.sin(Z/2)
    
    return {"x": qx, "y": qy, "z": qz, "w": qw}

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return {"x": X, "y": Y, "z": Z}

def H(xr, yr, tr):
    """ Jacobian Matrix """
    
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

def odom_callback(data):
    global last_pose_estimate
    odom = {
        "header": {
            "seq": data.header.seq,
            "stamp": data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
            "frame_id": data.header.frame_id
        },
        "child_frame_id": data.child_frame_id,
        "pose" : {
            "pose": {
                "position": {
                    "x": data.pose.pose.position.x,
                    "y": data.pose.pose.position.y,
                    "z": data.pose.pose.position.z
                },
                "orientation": {
                    "x": data.pose.pose.orientation.x,
                    "y": data.pose.pose.orientation.y,
                    "z": data.pose.pose.orientation.z,
                    "w": data.pose.pose.orientation.w
                }
            },
            "covariance": data.pose.covariance
        },
        "twist": {
            "twist": {
                "linear": {
                    "x": data.twist.twist.linear.x,
                    "y": data.twist.twist.linear.y,
                    "z": data.twist.twist.linear.z
                },
                "angular": {
                    "x": data.twist.twist.angular.x,
                    "y": data.twist.twist.angular.y,
                    "z": data.twist.twist.angular.z
                }
            },
            "covariance": data.pose.covariance
        }
    }
    
    pos = odom["pose"]["pose"]["position"]
    rot = odom["pose"]["pose"]["orientation"]

    rot = quaternion_to_euler_angle(rot["w"], rot["x"], rot["y"], rot["z"])

    # Get odometry by subtracting the current pose from the last pose estimation
    pose = np.array([pos["x"], pos["y"], rot["z"]]) - last_pose_estimate
    last_pose_estimate = pose

    # Update particles position if particle moved (needs to be changed to a more realistic threshold)
    if pos["x"] > 0 or pos["y"] > 0 or rot["z"] > 0:
        pf.sample_pose(pose, odom_covariance)

def scan_callback(data):
    laser = {
        "header": {
            "seq": data.header.seq,
            "stamp": data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
            "frame_id": data.header.frame_id
        },
        "angle_min": data.angle_min,
        "angle_max": data.angle_max,
        "angle_increment": data.angle_increment,
        "time_increment": data.time_increment,
        "scan_time": data.scan_time,
        "range_min": data.range_min,
        "range_max": data.range_max,
        "ranges": data.ranges,
        "intensities": data.intensities
    }

    landmarks = extract_landmarks(laser)

    
    if len(landmarks) != 0:
        pf.observe_landmarks(landmarks, H)

    pf.resample(pf.N)

    update_map()

def update_map():

    global map

    # Needs to expand if the map grows larger
    map["map_metadata"].height = 384 
    map["map_metadata"].width = 384 

    # Is it constant?
    map["map_metadata"].resolution = 0.05 

    # Copied from rosbags so the map appear on the middle of the RViz referencial
    map["map_metadata"].origin.orientation.w = 1.0
    map["map_metadata"].origin.position.x = -10.0
    map["map_metadata"].origin.position.y = -10.0


    # Simulate unknown map (all -1)
    map["grid"].data = (-1 *np.ones(map["map_metadata"].height*map["map_metadata"].width, np.int_)).tolist()

    # Define map header and info
    map["grid"].header.stamp = rospy.Time.now()
    map["grid"].header.frame_id = "map"
    map["grid"].info = map["map_metadata"]

    # Define pose header
    map["pose"].header.stamp = rospy.Time.now()
    map["pose"].header.frame_id = "map"
    
    # Example on how to change position and orientation 
    map["pose"].pose.position.x += 0.01
    map["pose"].pose.position.y += 0.01
    rot = euler_angle_to_quaternion(0, 0, np.pi / 4)
    map["pose"].pose.orientation.x = rot["x"]
    map["pose"].pose.orientation.y = rot["y"]
    map["pose"].pose.orientation.z = rot["z"]
    map["pose"].pose.orientation.w = rot["w"]
    
    # Publish new map and pose
    publishers["map_metadata"].publish(map["map_metadata"])
    publishers["grid"].publish(map["grid"])
    publishers["pose"].publish(map["pose"])

def main():
    global odom_covariance
    odom_covariance = np.array([0.1, 0.1, 0.1])

    global Qt
    Qt = np.array([[0.01, 0], [0, 0.0003]])

    global last_pose_estimate
    last_pose_estimate = np.array([0, 0, 0])

    global pf 
    pf = ParticleFilter(10, Qt)
   
    # Init node
    rospy.init_node('fast_slam_node', anonymous=True)
    rospy.Subscriber("odom", Odometry, odom_callback)
    rospy.Subscriber("scan", LaserScan, scan_callback)
    global publishers
    publishers = {
        "grid": rospy.Publisher('map', OccupancyGrid, queue_size=10),
        "map_metadata": rospy.Publisher('map_metadata', MapMetaData, queue_size=10),
        "pose": rospy.Publisher('pose', PoseStamped, queue_size=10)
    }
    global map
    map = {
        "grid": OccupancyGrid(),
        "map_metadata": MapMetaData(),
        "pose": PoseStamped()
    }

    print("Fast Slam Node initialized, now listening for scans and odometry to update the current estimated map and pose")
    rospy.spin()

if __name__ == '__main__':
    main()