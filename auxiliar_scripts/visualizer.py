import os
import pickle
import random
import sys

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

def line_distance(x, y, m, b):
    return abs(m * x - y + b) / np.sqrt(m**2 + 1)

def closest_to_line(x, y, m, b):
    return (((x + m * y) - m * b) / (m**2 + 1), (m * (x + m * y) + b) / (m**2 + 1))

def extract_features(ls, N=500, C=15, X=0.01, D=15):
    features = []
    D = int(round(np.radians(D) / ls["angle_increment"])) 
    S = 6
    # padded = ls["ranges"][-D:] + ls["ranges"] + ls["ranges"][:D]
    cartesian = tuple((to_cartesian(ls["angle_min"] + i * ls["angle_increment"], r) for i, r in enumerate(ls["ranges"])))
    available = [i for i in range(len(ls["ranges"])) if ls["ranges"][i] > 0.001]
    n = 0
    while n < N and len(available) and len(ls["ranges"]) > C:
        # print("n", n, "a", len(available))
        n += 1
        
        # Select a random reading
        R = random.choice(available)
        
        # Sample S readings withn D degrees of R
        total = [(i, ls["ranges"][i]) for i in available if (R - i) % len(ls["ranges"]) <= D]
        if len(total) < S:
            continue
        sample = random.sample(total, S)
        sample += [(R, ls["ranges"][R])]
        
        # Transform these points to cartesian coordinates and compute a least squares best fit line
        cartesian_sample = [cartesian[(i - D) % len(ls["ranges"])] for i, _ in sample]
        A = np.vstack([[x for x, _ in cartesian_sample], np.ones(len(cartesian_sample))]).T
        m, b = np.linalg.lstsq(A, np.array([y for _, y in cartesian_sample]), rcond=None)[0]
        
        # Find all readings within X meters of the line
        close_enough = []
        for i, point in enumerate(cartesian):
            if line_distance(*point, m, b) < X:
                close_enough.append(i)
        
        # If the number of points close to the line is above a threshold C
        if len(close_enough) > C:
            # Calculate new least squares best fit line
            A = np.vstack([[cartesian[i][0] for i in close_enough], np.ones(len(close_enough))]).T
            m, b = np.linalg.lstsq(A, np.array([cartesian[i][1] for i in close_enough]), rcond=None)[0]
            
            line_points = sorted((closest_to_line(*cartesian[i], m, b) for i in close_enough), key=lambda i: i[0])
            features.append((m, b, (line_points[0], line_points[-1])))
            
            for point in close_enough:
                try:
                    available.remove(point)
                except ValueError:
                    pass           
    return features
        

def main(t="ls", save=False):
    if t == "ls": scans = loader.from_dir(SCANS_DIR, "ls")[100:]
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
            
            features = extract_features(scan_info)
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
            # plt.plot(0, 0, "ro")
            
            if save:
                plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            else:
                m = n+100
                plt.title(str(m))
                plt.pause(0.1)
                # plt.show()
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
