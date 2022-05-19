#!/usr/bin/env python
import pickle

import rospy
import sensor_msgs.msg


def ls_msg_cb(data):
    ls = {
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
    with open(f"ls{ls['header']['stamp']}.pkl", "wb") as f:
        pickle.dump(ls, f)

def main():
    global MAP_METADATA, MAP_DATA

    rospy.init_node("laserscan_reader", anonymous=True)
    rospy.Subscriber("scan", sensor_msgs.msg.LaserScan, ls_msg_cb)
    rospy.loginfo("Listening for laser scan data")
    rospy.spin()
    

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
