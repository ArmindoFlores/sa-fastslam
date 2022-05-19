#!/usr/bin/env python
import pickle
import threading

import matplotlib.pyplot as plt
import nav_msgs.msg
import numpy as np
import rospy

LOCK = threading.Lock()
MAP_DATA = None
MAP_METADATA = None

def map_msg_cb(data):
    global MAP_DATA
    rospy.loginfo("Received map data")
    with LOCK:
        MAP_DATA = data.data

def map_metadata_msg_cb(metadata):
    global MAP_METADATA
    rospy.loginfo("Received map metadata")
    with LOCK:
        MAP_METADATA = metadata

def main():
    global MAP_METADATA, MAP_DATA

    rospy.init_node("map_reader", anonymous=True)
    rospy.Subscriber("map", nav_msgs.msg.OccupancyGrid, map_msg_cb)
    rospy.Subscriber("map_metadata", nav_msgs.msg.MapMetaData, map_metadata_msg_cb)
    rospy.loginfo("Listening for map data")
    
    n = 0
    while not rospy.is_shutdown():
        map2d = None
        with LOCK:
            if MAP_METADATA is not None and MAP_DATA is not None:
                width, height = MAP_METADATA.width, MAP_METADATA.height
                if width * height != len(MAP_DATA):
                    rospy.signal_shutdown("Invalid width or height")
                map2d = np.abs(np.array(MAP_DATA).reshape((height, width)))
                map2d = 255 - np.where(map2d == 100, 2, map2d) * 127
                MAP_METADATA = MAP_DATA = None

        if map2d is not None:
            n += 1
            with open(f"map_frame{n}.pkl", "wb") as f:
                pickle.dump(map2d, f)
            plt.imshow(map2d, cmap="gray")
        plt.pause(0.05)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
