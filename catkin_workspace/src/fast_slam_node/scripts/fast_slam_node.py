#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import nav_msgs.msg
import sensor_msgs.msg
from auxiliar_scripts.particle_filter import *
from auxiliar_scripts.landmark_extractor import  *
from auxiliar_scripts.landmark_matching import  *
import numpy as np


def H(xr, yr, tr):
    """ Jacobian Matrix """
    
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

def odom_callback(data):
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
    
    twist = odom["twist"]["twist"]
    pos = twist["linear"]
    rot = twist["angular"]
    # Update particles position if particle moved (needs to be changed to a more realistic threshold)
    if pos["x"] > 0 or pos["y"] > 0 or rot["z"] > 0:
        pf.sample_pose(np.array([pos["x"], pos["y"], rot["z"]]), odom_covariance)

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

    landmarks = extract_landmarks(laser["ranges"])

    if len(landmarks) != 0:
        pf.observe_landmarks(landmarks, H, laser_covariance, Qt)
    

    pf.resample(pf.N)

    
def main():

    global pf 
    global odom_covariance
    global laser_covariance
    global Qt

    pf = ParticleFilter(10)
    odom_covariance = np.array([0.0005, 0.0005, 0.0001])
    laser_covariance = 0.01 * np.identity(2)
    Qt = 0.01


    rospy.init_node('fast_slam_node', anonymous=True)

    rospy.Subscriber("odom", nav_msgs.msg.Odometry, odom_callback)
    rospy.Subscriber("scan", sensor_msgs.msg.LaserScan, scan_callback)
    
    rospy.spin()

if __name__ == '__main__':
    main()