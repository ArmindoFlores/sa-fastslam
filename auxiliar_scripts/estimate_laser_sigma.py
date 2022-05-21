import os
import pickle
import sys

import numpy as np

import loader

SAMPLE = "corredor-16-maio"
LASER_DIR = os.path.join("laser-scans", SAMPLE)


def main():
    laser = loader.from_dir(LASER_DIR, "ls")[:143]

    dist_estimates = np.zeros((len(laser), 360))
    for i, file in enumerate(laser):
        with open(file, "rb") as f:
            laser_reading = pickle.load(f)
            dist_estimates[i] = laser_reading["ranges"]

    dist_estimates = dist_estimates.transpose()

    # Evaluate what angles need to be discarded
    flag = np.zeros(360)
    for i in range(360):
        for j in range(len(laser) - 1):
            if abs(dist_estimates[i][j] - dist_estimates[i][j + 1]) > 0.5:
                flag[i] = 1
                break

    # Discard invalid data arrays
    j = 0
    for i in range(360):
        if flag[i] == 1:
            dist_estimates = np.delete(dist_estimates, j, 0)
            j -= 1
        j += 1
    
    # Estimate maximum variance
    std_all = np.zeros(len(dist_estimates))
    for i in range(len(dist_estimates)):
        std_all[i] = np.std(dist_estimates[i])

    variance = np.amax(std_all) ** 2
    print(f"The variance estimation is \u03C3\u00b2 = {variance:.8f} m\u00b2 = {(variance * 10e4):.3f} cm\u00b2")
    
    
if __name__ == "__main__":
    main()