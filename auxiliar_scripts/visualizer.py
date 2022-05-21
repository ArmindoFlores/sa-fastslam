import math
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import landmark_extractor
import loader

SAMPLE = "corredor-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)
        

def main(t="ls", save=False):
    if t == "ls": scans = loader.from_dir(SCANS_DIR, "ls")
    else: odoms = loader.from_dir(ODOM_DIR, "odom")
    
    n = 0
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    
    if t == "ls":
        for scan in scans:
            with open(scan, "rb") as f:
                scan_info = pickle.load(f)
            
            img = np.ones((257, 257), dtype=np.uint8) * 255
            scale_factor = 128 / 5
            offset = np.array([128, 128])
            
            for i, r in enumerate(scan_info["ranges"]):
                theta = scan_info["angle_min"] + scan_info["angle_increment"] * i
                x, y = round(r * np.cos(theta) * scale_factor + offset[0]), round(r * np.sin(theta) * scale_factor + offset[1])
                img[int(y)][int(x)] = 0
            
            start_time = time.time()
            landmarks = landmark_extractor.extract_landmarks(scan_info)
            end_time = time.time()
            print(f"Time taken: {round((end_time - start_time) * 1000, 2)}ms")
            
            for _, _, (start, end), landmark_pos in landmarks:
                start, end = np.array(start), np.array(end)
                start = start * scale_factor + offset
                end = end * scale_factor + offset
                plt.plot([start[0], end[0]], [start[1], end[1]], "r")
                plt.plot(*(landmark_pos * scale_factor + offset), "bo")
                    
            plt.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, -3, 3))
            plt.xlim([0, img.shape[0]])
            plt.ylim([0, img.shape[1]])
            plt.plot(0, 0, "ro")
            
            if save:
                plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            else:
                m = n+100
                plt.title(str(m))
                plt.pause(0.05)
            plt.clf()
            n += 1
    else:
        x, y = [], []
        for odom in odoms:
            with open(odom, "rb") as f:
                odom_info = pickle.load(f)
            pos = odom_info["pose"]["pose"]["position"]
            x.append(pos["x"])
            y.append(pos["y"])
        
        plt.scatter(x, y)
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])