#!/usr/bin/env python
import pickle

import rospy
import nav_msgs.msg


def odom_msg_cb(data):
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
    with open(f"odom{odom['header']['stamp']}.pkl", "wb") as f:
        pickle.dump(odom, f)

def main():
    global MAP_METADATA, MAP_DATA

    rospy.init_node("odometry_reader", anonymous=True)
    rospy.Subscriber("odom", nav_msgs.msg.Odometry, odom_msg_cb)
    rospy.loginfo("Listening for odometry data")
    rospy.spin()
    

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
