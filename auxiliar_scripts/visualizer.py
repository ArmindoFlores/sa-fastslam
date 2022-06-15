import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import landmark_extractor
import loader
import particle_filter

SAMPLE = "roundtrip-30-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)
REAL_LANDMARK_THRESHOLD = 6
IMG_SIZE = 256

def quaternion_to_euler(x, y, z, w):
    return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(2 * (w * y - z * x)), np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

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

def main(t="ls", save=True, display=False):
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
        for i, odom in tqdm.tqdm(enumerate(odoms), desc="Loading odometry files"):                
            with open(odom, "rb") as f:
                odom_info = pickle.load(f)
            pose = odom_info["pose"]["pose"]
            pos = pose["position"]
            rot = pose["orientation"]
            rot_euler = quaternion_to_euler(rot["x"], rot["y"], rot["z"], rot["w"])
            # if i == 0:
            #     positions[i] = np.array([pos["x"], pos["y"], rot["z"]])
            # else:
            positions[i] = np.array([pos["x"], pos["y"], rot_euler[2]])
            t[i] = odom_info["header"]["stamp"]
        positions -= positions[0]
        
        scans = []
        for scan in tqdm.tqdm(scan_files, desc="Loading scan files"):           
            with open(scan, "rb") as f:
                scan_info = pickle.load(f)
                scans.append(scan_info)
                
        pf = particle_filter.ParticleFilter(200, np.array([[0.01, 0], [0, 0.003]]), (0, 0, 0))
        
        prev_landmarks = []
        active_scan = None
        new_scan = False
        for i in tqdm.tqdm(range(8000, len(odoms))):
            c, s = np.cos(positions[i-1][2]), np.sin(positions[i-1][2])
            tmat = np.array([
                [c, s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            pose_estimate = tmat.dot(positions[i] - positions[i-1])
            
            new_scan = False
            while active_scan is None or (active_scan < len(scans) - 2 and scans[active_scan+1]["header"]["stamp"] < t[i]):
                if active_scan is None:
                    active_scan = 0
                else:
                    active_scan += 1
                new_scan = True
            
            # Update particle filter with new odometry data
            if np.sum(np.abs(pose_estimate)) > 0.0001:
                pf.sample_pose(pose_estimate, np.array([0.001, 0.001, 0.001]))
            
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            ax1_scale_factor = IMG_SIZE // 30
            ax2_scale_factor = IMG_SIZE // 10
            ax1_offset = np.array([40, IMG_SIZE - 40])
            ax2_offset = np.array([IMG_SIZE // 2, IMG_SIZE // 2])
            
            if active_scan is not None:
                scan_info = scans[active_scan]
                
                if new_scan:
                    landmarks = landmark_extractor.extract_landmarks(
                        scan_info,
                        N=500
                    )
                    if len(landmarks) != 0:
                        pf.observe_landmarks(landmarks, H)
                    prev_landmarks = landmarks
                                    
                for j, r in enumerate(scan_info["ranges"]):
                    theta = scan_info["angle_min"] + scan_info["angle_increment"] * j
                    index = np.array((r * np.cos(theta), r * np.sin(theta)))
                    
                    index = np.round(index * ax2_scale_factor + ax2_offset).astype(np.int32)
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
            
            for landmark in prev_landmarks:
                m = -landmark.equation[0] / landmark.equation[1]
                b = -landmark.equation[2] / landmark.equation[1] * ax2_scale_factor
                b = -m * ax2_offset[0] + b + ax2_offset[1]
                start = (0, b)
                end = (IMG_SIZE, m * IMG_SIZE + b)
                ax2.plot([start[0], end[0]], [start[1], end[1]], "g")
            
            best = pf.particles[0]
            max_n_landmarks = 0
            max_n_valid_landmarks = 0
            for particle in pf.particles:
                if best.weight < particle.weight:
                    best = particle
                    
                l1 = len(particle.landmark_matcher.landmarks)
                l2 = len(particle.landmark_matcher.valid_landmarks)
                if l1 > max_n_landmarks:
                    max_n_landmarks = l1
                if l2 > max_n_valid_landmarks:
                    max_n_valid_landmarks = l2
                position = particle.pose[:2] * ax1_scale_factor + ax1_offset
                ax1.plot(position[0], position[1], "go", markersize=3, alpha=0.1)

            # if new_scan:
                # print(f"\nMax landmarks: {max_n_landmarks}\nMax valid landmarks: {max_n_valid_landmarks}")

            # Display cloud center of mass
            position = np.array([0, 0, 0], dtype=np.float64)
            for particle in pf.particles:
                position += particle.pose
            position /= len(pf.particles)
            
            odom_position = positions[i].copy()
            odom_position[:2] = odom_position[:2] * ax1_scale_factor + ax1_offset
            
            position[:2] = position[:2] * ax1_scale_factor + ax1_offset
            ax1.plot(position[0], position[1], "ro", markersize=3)
            ax1.plot([position[0], position[0]+20*np.cos(position[2])], [position[1], position[1]+20*np.sin(position[2])], "r")
            
            ax1.plot(odom_position[0], odom_position[1], "bo", markersize=3)
            ax1.plot([odom_position[0], odom_position[0]+20*np.cos(odom_position[2])], [odom_position[1], odom_position[1]+20*np.sin(odom_position[2])], "b")
                
            for ekf in best.landmark_matcher.valid_landmarks:
                landmark = ekf.landmark
                m = -landmark.equation[0] / landmark.equation[1]
                b = -landmark.equation[2] / landmark.equation[1] * ax1_scale_factor
                b = -m * ax1_offset[0] + b + ax1_offset[1]
                start = (0, b)
                end = (IMG_SIZE, m * IMG_SIZE + b)
                ax1.plot([start[0], end[0]], [start[1], end[1]], "black")
                
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
                plt.savefig(f"output/ls{str(n+1).zfill(4)}.png")
            if display:
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
