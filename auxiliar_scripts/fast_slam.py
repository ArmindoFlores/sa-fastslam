import os
import random
import time

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import landmark_extractor
import particle_filter

ODOM_SIGMA = .001
LASER_SIGMA = .01
N = 10


def H(xr, yr, tr):
    return np.array([[1, xr * np.sin(tr) - yr * np.cos(tr)], [0, 1]])

def get_timestamp(filename, prefix):
    return float(".".join(os.path.basename(filename).split(".")[:-1])[len(prefix):])

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def normalize(v):
    return v / np.linalg.norm(v)

def generate_odometry_data(real_movement):
    x = random.gauss(real_movement[0], ODOM_SIGMA)
    y = random.gauss(real_movement[1], ODOM_SIGMA)
    z = random.gauss(real_movement[2], ODOM_SIGMA)
    return np.array((x, y, z))

def generate_laser_data(state, l=360, r=100):
    increment = 2 * np.pi / l
    
    result = []
    for i in range(l):
        theta = increment * i
        v = np.array([np.cos(theta + state["pos"][2]), np.sin(theta + state["pos"][2])])
        d = 0
        for distance in range(r):
            fixed_pos = state["pos"][:2].copy()
            # fixed_pos[1] = fixed_pos[1]
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
    # result = [state["my_pos_guess"], state["my_pos"], state["dst_pos"]]
    result = [state["my_pos"], state["my_pos_rot"], state["dst_pos"]]
    
    state["my_pos"].set_data(state["pos"][1], state["pos"][0])
    state["dst_pos"].set_data(state["destination"][1], state["destination"][0])
    # state["my_pos_guess"].set_data(state["pos_guess"][1], state["pos_guess"][0])
    state["my_pos_rot"].set_data([state["pos"][1], state["pos"][1]+20*np.sin(state["pos"][2])], [state["pos"][0], state["pos"][0]+20*np.cos(state["pos"][2])])
    
    # if state["update_ls"]:
    state["update_ls"] = False
    state["ls_img"].set_data(state["last_ls"])
    state["ls_img"].set_clim(state["last_ls"].min(), state["last_ls"].max())
    result.append(state["ls_img"])
        
    for i, landmark in enumerate(state["observed"]):
        equation = landmark.equation
        m = -equation[0] / equation[1]
        b = -equation[2] / equation[1]
        b = -m * 125 + b + 125
        # b = -m * 80 + b + 80
        start = (0, b)
        end = (250, m * 250 + b)
        if len(state["landmarks2"]) > i:
            state["landmarks2"][i].set_data([start[1], end[1]], [start[0], end[0]])
        else:
            state["landmarks2"].append(state["ax2"].plot([start[1], end[1]], [start[0], end[0]], color="green")[0])
    result += state["landmarks2"]
            
    for i, ekf in enumerate(state["matches"]):
        equation = ekf.landmark.equation
        if equation[1] != 0:
            start = 0, -equation[2] / equation[1]
            end = 250, -equation[0] / equation[1] * 250 + start[1]
        else:
            start = -equation[2] / equation[0], 0
            end = -equation[1] / equation[0] * 250 + start[0], 250
        if len(state["landmarks"]) > i:
            state["landmarks"][i].set_data([start[0], end[0]], [start[1], end[1]])
        else:
            state["landmarks"].append(state["ax"].plot([start[0], end[0]], [start[1], end[1]], color="green")[0])
    result += state["landmarks"]
    
    best_particle = None
    for i, particle in enumerate(state["particle_filter"].particles):
        if best_particle is None or particle.weight > best_particle.weight:
            best_particle = particle
        # print("Particle", i, particle.pose)
        state["particles"][i].set_data([particle.pose[1], particle.pose[0]])
    state["best_particle"][0].set_data([best_particle.pose[1], best_particle.pose[0]])
    result += state["particles"] + state["best_particle"]
        
    return result

def goes_through_wall(p1, p2, map_info):
    v = normalize(p2 - p1)
    p = p1.astype(np.float64)
    while np.any(p.astype(np.int32) != p2.astype(np.int32)):
        p += v
        asint = p.astype(np.int32)
        if not (0 < asint[0] < map_info.shape[0] and 0 < asint[1] < map_info.shape[1]):
            return False
        if map_info[asint[0]][asint[1]] == 0:
            return True
    return False
    
def update(n, state):
    # Move robot towards destination, or pick a new destination
    if state["destination"] is None or euclidean_distance(state["destination"], state["pos"][:2]) < 4:
        while True:
            state["destination"] = random.choice(np.argwhere(state["map_data"] == 128))
            if not goes_through_wall(state["destination"], state["pos"][:2], state["map_data"]):
                break
        state["velocity"] = (np.random.random() + 0.5) * 0.1
        
    # displacement = normalize(state["destination"] - state["pos"]) * state["velocity"]
    mov_vector = state["destination"] - state["pos"][:2]
    correct_orientation = np.angle(mov_vector[0] + mov_vector[1] * 1j) % (2 * np.pi)
    orientation_change = (correct_orientation - state["pos"][2]) * state["velocity"] * 0.1
    real_movement = np.array([
        state["velocity"] * np.cos(state["pos"][2] + orientation_change),
        state["velocity"] * np.sin(state["pos"][2] + orientation_change), 
        orientation_change
    ])
    state["pos"] += real_movement
    state["pos"][2] %= 2 * np.pi
    
    # Generate laser and odometry data based on the real movement
    odom_data = generate_odometry_data(real_movement)
    laser_data = None
    if n % N == 0:
        # Only generate laser data every 50 frames
        laser_data = generate_laser_data(state)
        
    # Update our particle filter  
    state["particle_filter"].sample_pose(odom_data, (ODOM_SIGMA * N)**2 * np.ones(3))
    # state["particle_filter"].sample_pose((*real_movement, 0), np.zeros(3))
    if laser_data is not None:
        state["update_ls"] = True
        laser_data_to_plot(laser_data, state["last_ls"], 1, np.array(state["last_ls"].shape) / 2)
        landmarks = landmark_extractor.extract_landmarks({
            "ranges": laser_data,
            "angle_increment": 2 * np.pi / 360,
            "angle_min": 0
        }, 2, C=20, X=2)
        state["observed"] = landmarks
        state["matches"] = []
        state["particle_filter"].observe_landmarks(landmarks)
        best_particle = None
        for particle in state["particle_filter"].particles:
            if best_particle is None or particle.weight > best_particle.weight:
                best_particle = particle
        state["matches"] = best_particle.landmark_matcher.valid_landmarks
        # print("\n".join(map(lambda l: str(l.landmark), best_particle.landmark_matcher.valid_landmarks)))
        # print("Mapped landmarks:", len(best_particle.landmark_matcher.valid_landmarks))
        
        state["particle_filter"].resample(frac=0.8)
        
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
    my_pos_rot, = ax1.plot([], [], "r")
    # my_pos_guess, = ax1.plot([], [], "go", markersize=3)
    dst_pos, = ax1.plot([], [], "bx")
    
    ax1.set_xlim([0, image.shape[0]])
    ax1.set_ylim([0, image.shape[1]])
    ax2.set_xlim([0, image.shape[0]])
    ax2.set_ylim([0, image.shape[1]])
    
    state = {
        "velocity": 0.1,
        "pos": np.resize((np.array(image.shape) / 2), 3),
        # "pos_guess": np.array(image.shape) / 2,
        "destination": np.array(image.shape) / 2,
        "my_pos": my_pos,
        "my_pos_rot": my_pos_rot,
        # "my_pos_guess": my_pos_guess,
        "dst_pos": dst_pos,
        "map_data": map_data,
        "ls_img": ls_img,
        "last_ls": image,
        "update_ls": False,
        "ax": ax1,
        "ax2": ax2,
        "particle_filter": particle_filter.ParticleFilter(100, np.array([[0.1, 0], [0, 0.03]]), H, (*(np.array(image.shape) / 2), 0), minimum_observations=6, distance_threshold=10, max_invalid_landmarks=12),
        "particles": [],
        "matches": [],
        "landmarks": [],
        "best_particle": [],
        "observed": [],
        "landmarks2": []
    }
    
    best_particle = None
    for particle in state["particle_filter"].particles:
        if best_particle is None or particle.weight > best_particle.weight:
            best_particle = particle
        p, = ax1.plot(particle.pose[0], particle.pose[1], "go", markersize=3, alpha=.1)
        state["particles"].append(p)
    c, = ax1.plot(best_particle.pose[0], best_particle.pose[1], "yo", markersize=3)
    state["best_particle"].append(c)
    
    animation.FuncAnimation(fig, lambda n: update(n, state), None, interval=15, blit=True)
    plt.show()
        

if __name__ == "__main__":
    main()
