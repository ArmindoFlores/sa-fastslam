import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

import loader

SAMPLE = "salalab-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)


def main(save=False):
    scans = loader.from_dir(SCANS_DIR, "ls")
    
    n = 0
    _, ax = plt.subplots(1, 2)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    
    for scan in scans:
        with open(scan, "rb") as f:
            scan_info = pickle.load(f)
        
        img = np.ones((257, 257)) * 255
        scale_factor = 128 / 5
        offset = 128, 128
        
        for i, r in enumerate(scan_info["ranges"]):
            theta = scan_info["angle_min"] + scan_info["angle_increment"] * i
            x, y = round(r * np.cos(theta) * scale_factor + offset[0]), round(r * np.sin(theta) * scale_factor + offset[1])
            img[int(y)][int(x)] = 0
            
        print(scan_info)
        
        ax[0].imshow(img, cmap="gray", interpolation="nearest", extent=(-3, 3, -3, 3))
        ax[0].plot(0, 0, "ro")
        
        ax[1].imshow(img, cmap="gray", interpolation="nearest", extent=(-3, 3, -3, 3))
        ax[1].plot(0, 0, "ro")
        
        if save:
            plt.savefig(f"output/ls{str(n+1).zfill(3)}.png")
            plt.clf()
        else:
            plt.pause(0.1)
        n += 1


if __name__ == "__main__":
    main(*sys.argv[1:])
