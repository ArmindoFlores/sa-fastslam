import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import landmark_extractor
import loader

SAMPLE = "salalab-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)
REAL_LANDMARK_THRESHOLD = 6


def transform_landmark(landmark, position, rotmat):
    start = position[:2] + np.dot(rotmat, -landmark.start)
    end =  position[:2] + np.dot(rotmat, -landmark.end)
    v = end - start    
    if v[0] != 0:
        a = v[1] / v[0]
        b = -1
        c = start[1] - a * start[0]
    else:
        a = -1
        b = v[0] / v[1]
        c = start[0] - b * start[1]
    return landmark_extractor.Landmark(a, b, c, start, end)

def main(t="ls", save=False):
    global positions
    scans = loader.from_dir(SCANS_DIR, "ls")
    try:
        odoms = loader.from_dir(ODOM_DIR, "odom")
    except Exception:
        odoms = None
    
    n = 0

    if t == "ls":
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        
        if odoms is None:
            positions = None
        else:
            t = np.ones(len(odoms))
            positions = np.zeros((len(odoms), 4))
            for i, odom in enumerate(odoms):
                with open(odom, "rb") as f:
                    odom_info = pickle.load(f)
                pose = odom_info["pose"]["pose"]
                pos = pose["position"]
                rot = pose["orientation"]
                positions[i] = np.array([pos["x"], pos["y"], rot["z"], rot["w"]])
                t[i] = odom_info["header"]["stamp"]
            positions -= positions[0]

        global_landmarks = []
        global_real_landmarks = []
        
        for scan in scans:           
            with open(scan, "rb") as f:
                scan_info = pickle.load(f)
                
            tnow = scan_info["header"]["stamp"]
            pnow = positions[np.argwhere(t >= tnow)[0]][0]
            
            img = np.ones((257, 257), dtype=np.uint8) * 255
            scale_factor = 128 / 5
            offset = np.array([128, 128])

            rnow = np.pi - (pnow[2] - pnow[3])
            rotmat = np.array([[np.cos(rnow), -np.sin(rnow)], [np.sin(rnow), np.cos(rnow)]])
            
            for i, r in enumerate(scan_info["ranges"]):
                theta = scan_info["angle_min"] + scan_info["angle_increment"] * i
                index = np.array((r * np.cos(theta + rnow + np.pi), r * np.sin(theta + rnow + np.pi)))
                index += pnow[:2]
                
                index = np.round(index * scale_factor + offset).astype(np.int32)
                if 0 <= index[1] < img.shape[0] and 0 <= index[0] < img.shape[1]:
                    img[index[1]][index[0]] = 0
            
            ax1.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, -3, 3))
            # ax2.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, -3, 3))
            ax1.set_xlim([0, img.shape[0]])
            ax1.set_ylim([0, img.shape[1]])
            ax2.set_xlim([0, img.shape[0]])
            ax2.set_ylim([0, img.shape[1]])
            
            landmarks = landmark_extractor.extract_landmarks(scan_info)
            
            global_real_landmarks = list(filter(lambda l: l.count > REAL_LANDMARK_THRESHOLD, global_landmarks))
               
            for real_landmark in global_real_landmarks:     
                start = real_landmark.start * scale_factor + offset
                end = real_landmark.end * scale_factor + offset
                ax2.plot([start[0], end[0]], [start[1], end[1]], color="black")
            
            matches = 0
            for landmark in map(lambda l: transform_landmark(l, pnow[:2], rotmat), landmarks):
                # start = (-pnow[:2] + np.dot(rotmat, -landmark.start)) * scale_factor + offset
                # end =  (-pnow[:2] + np.dot(rotmat, -landmark.end))* scale_factor + offset
                start = landmark.start * scale_factor + offset
                end =  landmark.end * scale_factor + offset
                ax1.plot([start[0], end[0]], [start[1], end[1]], "r")
                
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
                 
            ax1.plot(*(pnow[:2] * scale_factor + offset), "ro")
            ax2.plot(*(pnow[:2] * scale_factor + offset), "ro")
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
        x, y, z, w = [], [], [], []
        for odom in odoms:
            with open(odom, "rb") as f:
                odom_info = pickle.load(f)
            pos = odom_info["twist"]["twist"]["angular"]
            x.append(pos["x"])
            y.append(pos["y"])
            z.append(pos["z"])
            # w.append(pos["w"])
        
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        plt.plot(w)
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])