import os
import pickle

import numpy as np

import loader

N_SAMPLES = 50
SAMPLE = "salalab-16-maio"
ODOM_DIR = os.path.join("odometry", SAMPLE)


def main():
    odom = loader.from_dir(ODOM_DIR, "odom")[:50]
    
    xy_estimates = np.zeros((50, 2))
    for i, file in enumerate(odom):
        with open(file, "rb") as f:
            odom_reading = pickle.load(f)
            
        pos = odom_reading["pose"]["pose"]["position"]
        xy_estimates[i] = np.array([pos["x"], pos["y"]])

    variance = np.std(xy_estimates, axis=0) ** 2
    print(f"variance = {variance * 1000} mm")
    
    
if __name__ == "__main__":
    main()