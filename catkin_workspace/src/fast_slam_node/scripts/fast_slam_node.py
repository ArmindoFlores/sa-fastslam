#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import nav_msgs.msg
import sensor_msgs.msg
import sys
sys.path.insert(1, 'auxiliar_scripts')
from particle_filter import *
from landmark_extractor import  *
from landmark_matching import  *
import numpy as np


def H(xr, yr, tr):
    """ Jacobian Matrix """
    
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

def odom_callback(data):
    global last_pose
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
    

    pose = np.array([pos["x"], pos["y"], rot["z"]]) - last_pose
    last_pose = pose

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
        pf.observe_landmarks(landmarks, H, Qt)
    
    
    for p in pf.particles:
        print(p)

    pf.resample(pf.N)

    
def main():

    global pf 
    global odom_covariance
    global laser_covariance
    global Qt
    global last_pose

    pf = ParticleFilter(10)
    odom_covariance = np.array([0.1, 0.1, 0.1])
    laser_covariance = 0.01 * np.identity(2)
    Qt = 0.01
    last_pose = np.array([0, 0, 0])


    rospy.init_node('fast_slam_node', anonymous=True)

    rospy.Subscriber("odom", nav_msgs.msg.Odometry, odom_callback)
    rospy.Subscriber("scan", sensor_msgs.msg.LaserScan, scan_callback)
    
    rospy.spin()

if __name__ == '__main__':
    main()