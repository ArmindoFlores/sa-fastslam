#!/usr/bin/env python
import math
import os
import sys
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

sys.path.insert(1, 'auxiliar_scripts')
from landmark_extractor import *
from landmark_matching import *
from particle_filter import *


particle_lock = threading.Lock()


def to_cartesian(theta, r):
    if r > 0.001:
        return r * np.array((math.cos(theta), math.sin(theta)))
    
    return None

def get_current_pose_estimate():
    """ Calculates center of mass of the particles """

    global N_particles
    global map
    
    #pose = np.array([.0, .0, .0])
    max_weight = pf.particles[0].weight
    pose = np.array([pf.particles[0].pose[0], pf.particles[0].pose[1], pf.particles[0].pose[2]])
    

    for i in range(N_particles):
        # For pose array
        map["pose_array"].poses[i].position.x = pf.particles[i].pose[0]
        map["pose_array"].poses[i].position.y = pf.particles[i].pose[1]
        rot = euler_angle_to_quaternion(0, 0, pf.particles[i].pose[2])
        map["pose_array"].poses[i].orientation.x = rot["x"]
        map["pose_array"].poses[i].orientation.y = rot["y"]
        map["pose_array"].poses[i].orientation.z = rot["z"]
        map["pose_array"].poses[i].orientation.w = rot["w"]
        if i != 0:
            # get best particle
            if pf.particles[i].weight > max_weight:
                pose = np.array([pf.particles[i].pose[0], pf.particles[i].pose[1], pf.particles[i].pose[2]])
                max_weight = pf.particles[i].weight

        # For average pose
        #pose += pf.particles[i].pose
    

    return pose #pose / N_particles

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
    return {
        "x": np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        "y": np.arcsin(2 * (w * y - z * x)), 
        "z": np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    }

def H(xr, yr, tr):
    """ Jacobian Matrix """
    
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

def odom_callback(data):
    global last_pose_estimate, N_particles, bag_initial_time

    if bag_initial_time is None:
        bag_initial_time = rospy.Time.now().to_sec() - data.header.stamp.to_sec()

    time = rospy.Time.now().to_sec() - bag_initial_time

    odom = {
        "header": {
            "seq": data.header.seq,
            "stamp": data.header.stamp.to_sec(),
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

    # If the scan was a long time ago
    if time - odom['header']['stamp'] > 0.2:
        return

    pos = odom["pose"]["pose"]["position"]
    rot = odom["pose"]["pose"]["orientation"]

    rot = quaternion_to_euler_angle(rot["w"], rot["x"], rot["y"], rot["z"])
    pose = np.array([pos["x"], pos["y"], rot["z"]])

    pose_estimate = pose - last_pose_estimate
    last_pose_estimate = pose

    # Update particles position if particle moved (needs to be changed to a more realistic threshold)
    if abs(pose_estimate[0]) > 0.0001 or abs(pose_estimate[1]) > 0.0001 or abs(pose_estimate[2]) > 0.00005:
        with particle_lock:
            pf.sample_pose(pose_estimate, odom_covariance)

def scan_callback(data):
    global bag_initial_time, total_missed, total

    total += 1
    if bag_initial_time is None:
        bag_initial_time = rospy.Time.now().to_sec() - data.header.stamp.to_sec()
    
    time = rospy.Time.now().to_sec() - bag_initial_time
    
    laser = {
        "header": {
            "seq": data.header.seq,
            "stamp": data.header.stamp.to_sec(),
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

    # If the scan was a long time ago
    if time - laser['header']['stamp'] > 0.2:
        total_missed += 1
        return

    #rospy.loginfo(f"Time difference: {time - laser['header']['stamp']} s")
    rospy.loginfo(f"Miss %: {round(total_missed/total * 100)}")

    landmarks = extract_landmarks(laser)
    
    if len(landmarks) != 0:
        total_matches, max_matches = pf.observe_landmarks(landmarks, H)
        rospy.loginfo(f"Seen: {len(landmarks)} Max Matches: {max_matches} Total Matches: {total_matches} ({round(total_matches / pf.N, 2)} per particle)")

    update_map(laser["ranges"], laser["angle_increment"], laser["angle_min"])
    with particle_lock:
        pf.resample(pf.N)

    #update_map(laser["ranges"], laser["angle_increment"], laser["angle_min"])
    publish_map()

def update_map(ranges, angle_increment, min_angle):
    """ Update grid map and pose estimate """
    global map

    # Offset from the matrix to the frame
    offset = np.array([map["map_metadata"].width // 2, map["map_metadata"].height // 2 ])

    # Find the current pose estimate
    pose = get_current_pose_estimate()
    rot = euler_angle_to_quaternion(0, 0, pose[2])

    # Update pose
    map["pose"].pose.position.x = pose[0] 
    map["pose"].pose.position.y = pose[1]
    map["pose"].pose.orientation.x = rot["x"]
    map["pose"].pose.orientation.y = rot["y"]
    map["pose"].pose.orientation.z = rot["z"]
    map["pose"].pose.orientation.w = rot["w"]

    angle = min_angle + pose[2]
    for r in ranges:
        point = to_cartesian(angle, r)
        
        angle += angle_increment
        if point is None:
            continue
        
        # Transform from the robot frame to the world frame
        point += np.array_split(pose, 2)[0]

        # Transform from the world frame to the matrix
        point /= map["map_metadata"].resolution
        point = point.astype(int)
        column = offset[1] + point[0]
        row = offset[0] + point[1]
        

        index = row*map["map_metadata"].width + column
        if index < len(map["grid"].data):
            map["grid"].data[index] += 10

            if  map["grid"].data[index] > 100:
                map["grid"].data[index] = 100

        

def publish_map():
    global map

    # Define map header and info
    map["grid"].header.stamp = rospy.Time.now()
    map["grid"].header.frame_id = "map"
    map["grid"].info = map["map_metadata"]

    # Define pose header
    map["pose"].header.stamp = rospy.Time.now()
    map["pose"].header.frame_id = "map"

    map["pose_array"].header.stamp = rospy.Time.now()
    map["pose_array"].header.frame_id = "map"
    
    # Publish new map and pose
    publishers["map_metadata"].publish(map["map_metadata"])
    publishers["grid"].publish(map["grid"])
    publishers["pose"].publish(map["pose"])
    publishers["pose_array"].publish(map["pose_array"])

def main():
    global total
    global total_missed
    total = 0
    total_missed = 0

    global odom_covariance
    # odom_covariance = np.array([0.005, 0.005, 0.001])
    odom_covariance = np.array([0.00005, 0.00005, 0.00005])

    global Qt
    Qt = np.array([[0.01, 0], [0, 0.0003]])

    global last_pose_estimate
    last_pose_estimate = np.array([0, 0, 0])

    global N_particles
    N_particles = 150

    global pf 
    pf = ParticleFilter(N_particles, Qt)

    global bag_initial_time 
    bag_initial_time = None
   
    # Init node
    rospy.init_node('fast_slam_node', anonymous=True)
    rospy.Subscriber("odom", Odometry, odom_callback)
    rospy.Subscriber("scan", LaserScan, scan_callback)
    global publishers
    publishers = {
        "grid": rospy.Publisher('map', OccupancyGrid, queue_size=10),
        "map_metadata": rospy.Publisher('map_metadata', MapMetaData, queue_size=10),
        "pose": rospy.Publisher('pose', PoseStamped, queue_size=10),
        "pose_array": rospy.Publisher('pose_array', PoseArray, queue_size=10)
    }
    global map
    map = {
        "grid": OccupancyGrid(),
        "map_metadata": MapMetaData(),
        "pose": PoseStamped(),
        "pose_array": PoseArray()
    }

    # Initialize pose array
    for i in range(N_particles):
        map["pose_array"].poses.append(Pose())

    # Needs to expand if the map grows larger
    map["map_metadata"].height = 832 
    map["map_metadata"].width = 832 

    
    map["map_metadata"].resolution = 0.05 

    # So the map appear on the middle of the RViz frame
    map["map_metadata"].origin.position.x = -(map["map_metadata"].width // 2) * map["map_metadata"].resolution
    map["map_metadata"].origin.position.y = -(map["map_metadata"].height // 2)  * map["map_metadata"].resolution

    # Initialize map
    map["grid"].data = (0 *np.ones(map["map_metadata"].height*map["map_metadata"].width, np.int_)).tolist()


    rospy.loginfo("Fast Slam Node initialized, now listening for scans and odometry to update the current estimated map and pose")
    rospy.spin()

if __name__ == '__main__':
    main()
