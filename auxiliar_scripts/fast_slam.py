import os
import pickle

import numpy as np

import loader

SAMPLE = "salalab-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)

def get_timestamp(filename, prefix):
    return float(".".join(os.path.basename(filename).split(".")[:-1])[len(prefix):])

def do_fslam_iter(map_state, pos_state, *, ls=None, odom=None):
    pass

def main():
    scans = loader.from_dir(SCANS_DIR, "ls")
    odom = loader.from_dir(ODOM_DIR, "odom")
    
    map_state = np.zeros((200, 200))
    pos_state = np.zeros((200, 200))
    scan_i = 0
    odom_i = 0
    t = max(get_timestamp(scans[scan_i], "ls"), get_timestamp(odom[odom_i], "odom"))
    while True:
        ts, to = get_timestamp(scans[scan_i], "ls"), get_timestamp(odom[odom_i], "odom")
        
        if t >= ts and t >= to:           
            if odom_i == len(odom) - 1 or scan_i == len(scans) - 1:
                break
            
            ls = od = None
            next_ts, next_to = get_timestamp(scans[scan_i+1], "ls"), get_timestamp(odom[odom_i+1], "odom")
            
            if next_ts <= t:
                scan_i += 1
                with open(scans[scan_i], "rb") as f:
                    ls = pickle.load(f)
            if next_to <= t:
                odom_i += 1  
                with open(odom[odom_i], "rb") as f:
                    od = pickle.load(f)
            
            if ls is not None or od is not None:     
                do_fslam_iter(map_state, pos_state, ls=ls, odom=od)  
            
        t += 0.001
        


if __name__ == "__main__":
    main()