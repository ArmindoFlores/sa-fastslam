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
REAL_LANDMARK_THRESHOLD = 6


def main(t="ls", save=False):
    global global_landmarks
    if t == "ls": scans = loader.from_dir(SCANS_DIR, "ls")
    else: odoms = loader.from_dir(ODOM_DIR, "odom")
    
    n = 0

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    plt.tight_layout()
    
    if t == "ls":
        global_landmarks = []
        
        for scan in scans:
            print("Real landmarks:", len(tuple(filter(lambda l: l.count > REAL_LANDMARK_THRESHOLD, global_landmarks))))
            
            with open(scan, "rb") as f:
                scan_info = pickle.load(f)
            
            img = np.ones((257, 257), dtype=np.uint8) * 255
            scale_factor = 128 / 5
            offset = np.array([128, 128])
            
            for i, r in enumerate(scan_info["ranges"]):
                theta = scan_info["angle_min"] + scan_info["angle_increment"] * i
                x, y = round(r * np.cos(theta) * scale_factor + offset[0]), round(r * np.sin(theta) * scale_factor + offset[1])
                img[int(y)][int(x)] = 0
            
            ax1.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, -3, 3))
            ax2.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, -3, 3))
            ax1.set_xlim([0, img.shape[0]])
            ax1.set_ylim([0, img.shape[1]])
            ax2.set_xlim([0, img.shape[0]])
            ax2.set_ylim([0, img.shape[1]])
            
            start_time = time.time()
            landmarks = landmark_extractor.extract_landmarks(scan_info)
            end_time = time.time()
            print(f"Time taken: {round((end_time - start_time) * 1000, 2)}ms")
            
            for real_landmark in filter(lambda l: l.count > REAL_LANDMARK_THRESHOLD, global_landmarks):
                start = real_landmark.start * scale_factor + offset
                end = real_landmark.end * scale_factor + offset
                # landmark_pos = real_landmark.closest_point(0, 0)
                # ax2.plot(*(landmark_pos * scale_factor + offset), "o", color="black")
                ax2.plot([start[0], end[0]], [start[1], end[1]], color="black")
                
            for landmark in landmarks:
                start = landmark.start * scale_factor + offset
                end = landmark.end * scale_factor + offset
                ax1.plot([start[0], end[0]], [start[1], end[1]], "r")
                ax1.plot(*(landmark.closest_point(0, 0) * scale_factor + offset), "bo")
                
                # Match landmarks
                closest = None
                for glandmark in global_landmarks:
                    ld = glandmark.distance(landmark)
                    if ld < 1 and (closest is None or ld < closest["difference"]):
                        closest = {"difference": ld, "landmark": glandmark}
                if closest is not None:
                    # Update global landmark
                    if closest["landmark"].count >= REAL_LANDMARK_THRESHOLD:
                        start = closest["landmark"].start * scale_factor + offset
                        end = closest["landmark"].end * scale_factor + offset
                        # landmark_pos = closest["landmark"].closest_point(0, 0)
                        # ax2.plot(*(landmark_pos * scale_factor + offset), "o", color="green")
                        ax2.plot([start[0], end[0]], [start[1], end[1]], color="green")
                    closest["landmark"].update(*landmark.equation, landmark.start, landmark.end)
                else:
                    # A match was not found
                    global_landmarks.append(landmark)    
                    
            ax1.plot(*offset, "ro")
            ax2.plot(*offset, "ro")
            
            if save:
                plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            else:
                ax1.grid()
                ax2.grid()
                ax1.set_title(str(n))
                ax2.set_title(str(n))
                plt.pause(0.05)
            ax1.clear()
            ax2.clear()
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