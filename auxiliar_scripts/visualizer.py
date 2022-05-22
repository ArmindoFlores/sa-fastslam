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
        global_real_landmarks = []
        
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
            
            global_real_landmarks = list(filter(lambda l: l.count > REAL_LANDMARK_THRESHOLD, global_landmarks))
            # for i, rl1 in enumerate(global_real_landmarks):
            #     for rl2 in global_real_landmarks:
            #         if rl1 == rl2 or rl1 is None or rl2 is None:
            #             continue
            #         # if rl1.intersects(rl2) and rl1.distance(rl2) < 0.2
            #         if np.linalg.norm(rl1.closest_point(0, 0) - rl2.closest_point(0, 0)) < 0.25:
            #             global_real_landmarks[i] = None
            #             global_landmarks.remove(rl1)
            #             rl2.update(rl1)
            #             rl1 = None
            #             print("Would merge")
            # global_real_landmarks[:] = filter(lambda l: l is not None, global_real_landmarks)
               
            for real_landmark in global_real_landmarks:     
                start = real_landmark.start * scale_factor + offset
                end = real_landmark.end * scale_factor + offset
                ax2.plot([start[0], end[0]], [start[1], end[1]], color="black")
            
            matches = 0
            for landmark in landmarks:
                start = landmark.start * scale_factor + offset
                end = landmark.end * scale_factor + offset
                ax1.plot([start[0], end[0]], [start[1], end[1]], "r")
                # ax1.plot(*(landmark.closest_point(0, 0) * scale_factor + offset), "bo")
                
                # Match landmarks
                closest = None
                for glandmark in global_landmarks:
                    # ld = glandmark.distance(landmark)
                    ld = np.linalg.norm(landmark.closest_point(0, 0) - glandmark.closest_point(0, 0))
                    # if glandmark.intersects(landmark) and ld < 1 and (closest is None or ld < closest["difference"]):
                    if ld < 0.25 and (closest is None or ld < closest["difference"]):
                        closest = {"difference": ld, "landmark": glandmark}
                if closest is not None:
                        # Update global landmark
                        if closest["landmark"].count >= REAL_LANDMARK_THRESHOLD:
                            start = closest["landmark"].start * scale_factor + offset
                            end = closest["landmark"].end * scale_factor + offset
                            ax2.plot([start[0], end[0]], [start[1], end[1]], color="green")
                            matches += 1
                        closest["landmark"].update(landmark)
                else:
                    global_landmarks.append(landmark)                     
                    
            ax1.plot(*offset, "ro")
            ax2.plot(*offset, "ro")
            ax1.set_title(f"Landmarks seen: {len(landmarks)}")
            ax2.set_title(f"Matched landmarks: {matches}/{len(global_real_landmarks)}")
            ax1.grid()
            ax2.grid()
            
            if save:
                plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            else:
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