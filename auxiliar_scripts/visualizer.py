import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

import particle_filter
import landmark_extractor
import loader

SAMPLE = "roundtrip-30-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)
REAL_LANDMARK_THRESHOLD = 6
IMG_SIZE = 256


def H(xr, yr, tr):
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

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
    return landmark_extractor.Landmark(a, b, c, start, end, landmark._r2)

def main(t="ls", save=False):
    global positions
    scan_files = loader.from_dir(SCANS_DIR, "ls")
    try:
        odoms = loader.from_dir(ODOM_DIR, "odom")
    except Exception:
        odoms = None
    
    n = 0

    if t == "ls":
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        t = np.ones(len(odoms))
        positions = np.zeros((len(odoms), 3))
        for i, odom in enumerate(odoms):                
            with open(odom, "rb") as f:
                odom_info = pickle.load(f)
            twist = odom_info["twist"]["twist"]
            pos = twist["linear"]
            rot = twist["angular"]
            if i == 0:
                positions[i] = np.array([pos["x"], pos["y"], rot["z"]])
            else:
                positions[i] += np.array([pos["x"], pos["y"], rot["z"]])
            t[i] = odom_info["header"]["stamp"]

        
        scans = []
        for scan in scan_files:           
            with open(scan, "rb") as f:
                scan_info = pickle.load(f)
                scans.append(scan_info)
                
        pf = particle_filter.ParticleFilter(200)
        
        active_scan = None
        new_scan = False
        for i in range(1, len(odoms)):
            pose_estimate = positions[i] - positions[i-1]
            
            new_scan = False
            if active_scan is None or (active_scan < len(scans) - 2 and scans[active_scan+1]["header"]["stamp"] < t[i]):
                if active_scan is None:
                    active_scan = 0
                else:
                    active_scan += 1
                new_scan = True
            
            # Update particle filter with new odometry data
            if np.sum(pose_estimate) > 0.01:
                pf.sample_pose(pose_estimate, np.array([0.0005, 0.0005, 0.0001]))
            
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            scale_factor = IMG_SIZE // 10
            offset = np.array([IMG_SIZE // 2, IMG_SIZE // 2])

            rnow = np.pi - positions[i][2]
            rotmat = np.array([[np.cos(rnow), -np.sin(rnow)], [np.sin(rnow), np.cos(rnow)]])
            
            if active_scan is not None:
                scan_info = scans[active_scan]
                
                if new_scan:
                    landmarks = landmark_extractor.extract_landmarks(
                        scan_info
                    )
                    if len(landmarks) != 0:
                        pf.observe_landmarks(landmarks, H, 0.01 * np.identity(2), 0.01)
                                    
                for j, r in enumerate(scan_info["ranges"]):
                    theta = scan_info["angle_min"] + scan_info["angle_increment"] * j
                    index = np.array((r * np.cos(theta + rnow + np.pi), r * np.sin(theta + rnow + np.pi)))
                    index += positions[i][:2]
                    
                    index = np.round(index * scale_factor + offset).astype(np.int32)
                    if 0 <= index[1] < img.shape[0] and 0 <= index[0] < img.shape[1]:
                        img[index[1]][index[0]] = 0
            
            ax1.imshow(np.zeros_like(img), cmap="Greys", interpolation="nearest")
            # ax1.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, 3, -3))
            ax2.imshow(img, cmap="gray", interpolation="nearest")#, extent=(-3, 3, 3, -3))
            ax1.set_xlim([0, img.shape[0]])
            ax1.set_ylim([0, img.shape[1]])
            ax2.set_xlim([0, img.shape[0]])
            ax2.set_ylim([0, img.shape[1]])
            # ax1.set_xlim([-3, 3])
            # ax1.set_ylim([-3, 3])
            # ax2.set_xlim([-3, 3])
            # ax2.set_ylim([-3, 3])
            

            # ax1.plot(*(positions[i][:2] * scale_factor * img_scale), "ro")
            # ax2.plot(*(positions[i][:2] * scale_factor * img_scale), "ro")
            # best = None
            best = pf.particles[0]
            for particle in pf.particles:
                if best is None or best.weight < particle.weight:
                    best = particle
                position = particle.pose[:2] * scale_factor + offset
                ax1.plot(position[0], position[1], "go", markersize=3, alpha=0.1)
                
            for landmark in best.landmark_matcher.valid_landmarks:
                m = -landmark.equation[0] / landmark.equation[1]
                b = -landmark.equation[2] / landmark.equation[1] * scale_factor
                b = -m * offset[0] + b + offset[1]
                start = (0, b)
                end = (IMG_SIZE, m * IMG_SIZE + b)
                ax1.plot([start[0], end[0]], [start[1], end[1]], "g")
                
            if new_scan:
                pf.resample(pf.N)
            
            # ax1.set_title(f"Landmarks seen: {len(landmarks)}")
            ax1.set_xlabel("x [m]")
            ax1.set_ylabel("y [m]")
            ax2.set_xlabel("x [m]")
            ax2.set_ylabel("y [m]")
            ax1.set_title(f"Frame {i}/{len(odoms)} ({round(100*i/len(odoms), 1)}%)")
            ax2.set_title(f"Frame {active_scan}/{len(scans)} ({round(100*active_scan/len(scans), 1)}%)")
            ax1.grid()
            ax2.grid()
            
            # print(len(matcher.landmarks))
            
            if save:
                plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            else:
                plt.pause(0.01)
            ax1.clear()
            ax2.clear()
            n += 1
    else:
        x, y, z, w = [], [], [], []
        for odom in odoms:
            with open(odom, "rb") as f:
                odom_info = pickle.load(f)
            pose = odom_info["pose"]["pose"]["position"]
            x.append(pose["x"])
            y.append(pose["y"])
            # z.append(pos["z"])
            # w.append(pos["w"])
        plt.scatter(x, y)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
