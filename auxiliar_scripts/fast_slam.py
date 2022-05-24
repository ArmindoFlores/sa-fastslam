import os
import random
import time

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import landmark_extractor
import landmark_matching

ODOM_SIGMA = .2
LASER_SIGMA = .2


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
    return landmark_extractor.Landmark(a, b, c, start, end)

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

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def normalize(v):
    return v / np.linalg.norm(v)

def generate_odometry_data(real_movement):
    x = random.gauss(real_movement[0], ODOM_SIGMA)
    y = random.gauss(real_movement[1], ODOM_SIGMA)
    return np.array((x, y))

def generate_laser_data(state, l=360, r=100):
    increment = 2 * np.pi / l
    
    result = []
    for i in range(l):
        theta = increment * i
        v = np.array([np.cos(theta), np.sin(theta)])
        d = 0
        for distance in range(r):
            fixed_pos = state["pos"].copy()
            # fixed_pos[1] = 250 - fixed_pos[1]
            pos = np.round(fixed_pos + v * distance).astype(np.int32)
            if (0 <= pos[0] < state["map_data"].shape[0]) and (0 <= pos[1] < state["map_data"].shape[1]):
                if state["map_data"][pos[0]][pos[1]] == 0:
                    d = random.gauss(distance, LASER_SIGMA)
                    break
            else:
                break
        result.append(d)
    return result

def laser_data_to_plot(data, img, scale, offset):
    img *= 0
    dim = img.shape
    angle_increment = 2 * np.pi / len(data)
    for i, r in enumerate(data):
        theta = angle_increment * i
        pos = r * np.array((np.cos(theta), np.sin(theta))) * scale + offset
        asint = np.round(pos).astype(np.int32)
        if (0 <= asint[0] < dim[0]) and (0 <= asint[1] < dim[1]):
            img[asint[0]][asint[1]] = 255
    return img
        
def update_display(state):
    result = [state["my_pos_guess"], state["my_pos"], state["dst_pos"]]
    
    state["my_pos"].set_data(state["pos"][1], state["pos"][0])
    state["dst_pos"].set_data(state["destination"][1], state["destination"][0])
    state["my_pos_guess"].set_data(state["pos_guess"][1], state["pos_guess"][0])
    
    if state["update_ls"]:
        state["update_ls"] = False
        state["ls_img"].set_data(state["last_ls"])
        state["ls_img"].set_clim(state["last_ls"].min(), state["last_ls"].max())
        result.append(state["ls_img"])
        
    for i, landmark in enumerate(state["matcher"].valid_landmarks):
        if landmark.equation[1] != 0:
            start = 0, -landmark.equation[2] / landmark.equation[1]
            end = 250, -landmark.equation[0] / landmark.equation[1] * 250 + start[1]
        else:
            start = -landmark.equation[2] / landmark.equation[0], 0
            end = -landmark.equation[1] / landmark.equation[0] * 250 + start[0], 250
        if len(state["landmarks"]) > i:
            state["landmarks"][i].set_data([250 - start[1], 250 - end[1]], [250 - start[0], 250 - end[0]])
        else:
            state["landmarks"].append(state["ax"].plot([250 - start[1], 250 - end[1]], [250 - start[0], 250 - end[0]], color="green")[0])
    result += state["landmarks"]
        
    return result

def goes_through_wall(p1, p2, map_info):
    v = normalize(p2 - p1)
    p = p1.astype(np.float64)
    while np.any(p.astype(np.int32) != p2.astype(np.int32)):
        p += v
        asint = p.astype(np.int32)
        if map_info[asint[0]][asint[1]] == 0:
            return True
    return False
    
def update(n, state):
    if state["destination"] is None or euclidean_distance(state["destination"], state["pos"]) < 4:
        while True:
            state["destination"] = random.choice(np.argwhere(state["map_data"] == 128))
            if not goes_through_wall(state["destination"], state["pos"], state["map_data"]):
                break
        state["velocity"] = (np.random.random() + 0.5) * 0.1
    
    if n > 1000:
        real_movement = normalize(state["destination"] - state["pos"]) *  state["velocity"]
    else:
        real_movement = np.array([0, 0])
    state["pos"] += real_movement
    
    odom_data = generate_odometry_data(real_movement)
    laser_data = None
    if n % 50 == 0:
        # Only generate laser data every 50 frames
        laser_data = generate_laser_data(state)
        
    state["pos_guess"] += odom_data
    if laser_data is not None:
        state["update_ls"] = True
        laser_data_to_plot(laser_data, state["last_ls"], 1, np.array(state["last_ls"].shape) / 2)
        landmarks = landmark_extractor.extract_landmarks({
            "ranges": laser_data,
            "angle_increment": 2 * np.pi / 360,
            "angle_min":0 
        }, 20, X=5)
        state["matches"] = []
        for landmark in map(lambda l: transform_landmark(l, 250-state["pos"], np.identity(2)), landmarks):
            match = state["matcher"].observe(landmark)
            if match is not None:
                state["matches"].append(match)
        print(f"Matches: {len(state['matches'])}/{len(state['matcher'].valid_landmarks)}")
        
    return update_display(state)

def main():
    np.random.seed(int(time.time() * 13) % 2**32)
    map_data = cv2.cvtColor(cv2.imread("maps/map1.png"), cv2.COLOR_BGR2GRAY)
    image = np.zeros((250, 250))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(255 - map_data, cmap="Greys", interpolation="bilinear")
    ls_img = ax2.imshow(image, cmap="Greys", interpolation="nearest")
    
    my_pos, = ax1.plot([], [], "ro", markersize=3)
    my_pos_guess, = ax1.plot([], [], "go", markersize=3)
    dst_pos, = ax1.plot([], [], "bx")
    
    ax1.set_xlim([0, image.shape[0]])
    ax1.set_ylim([0, image.shape[1]])
    ax2.set_xlim([0, image.shape[0]])
    ax2.set_ylim([0, image.shape[1]])
    
    state = {
        "velocity": 0.5,
        "pos": np.array(image.shape) / 2 + 100,
        "pos_guess": np.array(image.shape) / 2 + 100,
        "destination": np.array(image.shape) / 2 + 100,
        "my_pos": my_pos,
        "my_pos_guess": my_pos_guess,
        "dst_pos": dst_pos,
        "map_data": map_data,
        "ls_img": ls_img,
        "last_ls": image,
        "update_ls": False,
        "ax": ax1,
        "matcher": landmark_matching.LandmarkMatcher(8, 12, 10),
        "matches": [],
        "landmarks": [],
    }
    
    
    animation.FuncAnimation(fig, lambda n: update(n, state), None, interval=15, blit=True)
    plt.show()
        

if __name__ == "__main__":
    main()
