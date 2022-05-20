import os
import pickle

import numpy as np

import loader

SAMPLE = "salalab-16-maio"
SCANS_DIR = os.path.join("laser-scans", SAMPLE)
ODOM_DIR = os.path.join("odometry", SAMPLE)

def get_timestamp(filename, prefix):
    return float(".".join(os.path.basename(filename).split(".")[:-1])[len(prefix):])

def multivariate_gaussian(pos, mu, sigma):
    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2*np.pi)**n * sigma_det)
    
    # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def do_fslam_iter(map_state, pos_state, *, ls=None, odom=None):
    if odom is not None:
        # Use odometry to predict a new position
        pos = odom["pose"]["pose"]["position"]
        u = np.array([pos["x"], pos["y"]], dtype=np.float64)
        pos_state["muu"] += u
        pos_state["sigma"] += 1e-2

def main():
    scans = loader.from_dir(SCANS_DIR, "ls")
    odom = loader.from_dir(ODOM_DIR, "odom")
    
    shape = (200, 200)
    xx = np.linspace(-15, 15, 200)
    yy = xx.copy().T
    xx, yy = np.meshgrid(xx, yy)
    pos = np.dstack([xx, yy])
    
    n = 0
    map_state = np.zeros(shape)
    # pos_state = np.ones(shape) / (shape[0] * shape[1])  # Uniform distribution
    pos_state = {"muu": np.array([0, 0], dtype=np.float64), "sigma": 1e-12}
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
