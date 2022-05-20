import os
import pickle

import numpy as np

import loader

N_SAMPLES = 50
SAMPLE = "salalab-16-maio"
ODOM_DIR = os.path.join("odometry", SAMPLE)


def main():
    odom = loader.from_dir(ODOM_DIR, "odom")[:50]
    
    position = np.array([0, 0], dtype=np.float64)
    for file in odom:
        with open(file, "rb") as f:
            odom_reading = pickle.load(f)
            
        pos = odom_reading["pose"]["pose"]["position"]
        position += np.abs(np.array([pos["x"], pos["y"]]))
    print(position / N_SAMPLES)
    
    
if __name__ == "__main__":
    main()