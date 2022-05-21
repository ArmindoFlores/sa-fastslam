import os
import pickle
import random
import sys
import time
import cProfile, pstats, io
from pstats import SortKey

import matplotlib.pyplot as plt
import numpy as np

import loader

SAMPLE = "corredor-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)


def to_cartesian(theta, r):
    if r > 0.001:
        return r * np.cos(theta), r * np.sin(theta)
    else:
        return 500 * np.cos(theta), 500 * np.sin(theta)

def closest_to_line(x, y, m, b):
    return (((x + m * y) - m * b) / (m**2 + 1), (m * (x + m * y) + b) / (m**2 + 1))

def extract_features(ls, N=400, C=20, X=0.02, D=9):
    features = []
    D = int(round(np.radians(D) / ls["angle_increment"])) 
    S = 6
    rsize = len(ls["ranges"])
    cartesian = tuple(to_cartesian(ls["angle_min"] + i * ls["angle_increment"], r) for i, r in enumerate(ls["ranges"]))
    available = [i for i in range(rsize) if ls["ranges"][i] > 0.001]
    
    # Pre-allocate lists
    total = [None] * 360
    total_size = 0
    cartesian_sample_x = [0] * 360
    cartesian_sample_y = [0] * 360
    cartesian_sample_size = 0
    close_enough = [None] * 360
    close_enough_size = 0
    
    n = 0
    while n < N and len(available) and rsize > C:
        n += 1
        
        # Select a random reading
        R = random.choice(available)
        
        # Sample S readings withn D degrees of R
        total_size = 0
        for i in available:
            if (R - i) % rsize <= D:
                total[total_size] = i
                total_size += 1
        
        if total_size < S:
            continue
        sample = random.sample(total[:total_size], S)
        sample.append(R)
        
        # Transform these points to cartesian coordinates and compute a least squares best fit line
        cartesian_sample_size = 0
        for i in sample:
            x, y = cartesian[(i - D) % rsize]
            cartesian_sample_x[cartesian_sample_size] = x
            cartesian_sample_y[cartesian_sample_size] = y
            cartesian_sample_size += 1
            
        A = np.vstack([cartesian_sample_x[:cartesian_sample_size], np.ones(cartesian_sample_size)]).T
        m, b = np.linalg.lstsq(A, cartesian_sample_y[:cartesian_sample_size], rcond=None)[0]
        
        # Find all readings within X meters of the line
        d = 1 / pow(m*m + 1, 0.5)
        close_enough_size = 0
        for i, (x, y) in enumerate(cartesian):
            if abs(m * x - y + b) * d < X:
                close_enough[close_enough_size] = i
                close_enough_size += 1
        
        # If the number of points close to the line is above a threshold C
        if close_enough_size > C:
            # Calculate new least squares best fit line
            A = np.vstack([[cartesian[i][0] for i in close_enough[:close_enough_size]], np.ones(close_enough_size)]).T
            m, b = np.linalg.lstsq(A, np.array([cartesian[i][1] for i in close_enough[:close_enough_size]]), rcond=None)[0]
            
            line_points = sorted((closest_to_line(*cartesian[i], m, b) for i in close_enough[:close_enough_size]), key=lambda i: i[0])
            features.append((m, b, (line_points[0], line_points[-1])))
            
            for point in close_enough[:close_enough_size]:
                try:
                    available.remove(point)
                except ValueError:
                    pass
    return features
        

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
            features = extract_features(scan_info)
            end_time = time.time()
            print(f"Time taken: {round((end_time - start_time) * 1000, 2)}ms")
            landmarks = []
            for m, b, (start, end) in features:
                start, end = np.array(start), np.array(end)
                start = start * scale_factor + offset
                end = end * scale_factor + offset
                landmark_pos = np.array(closest_to_line(0, 0, m, b))
                
                isnew = True
                for landmark in landmarks:
                    if np.linalg.norm(landmark - landmark_pos) < 0.2:
                        isnew = False
                        break
                    
                if isnew:
                    landmarks.append(landmark_pos)
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