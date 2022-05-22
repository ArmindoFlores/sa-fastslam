import math
import random

import numpy as np


def to_cartesian(theta, r):
    if r > 0.001:
        return r * math.cos(theta), r * math.sin(theta)
    else:
        return 500 * math.cos(theta), 500 * math.sin(theta)

def closest_to_line(x, y, a, b, c):
    return ((b * (b * x - a * y) - a * c) / (a**2 + b**2), (a * (-b * x + a * y) - b * c) / (a**2 + b**2))

def extract_features(ls, N=400, C=22, X=0.02, D=10):
    features = []
    D = int(round(math.radians(D) / ls["angle_increment"])) 
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
            a, c = np.linalg.lstsq(A, np.array([cartesian[i][1] for i in close_enough[:close_enough_size]]), rcond=None)[0]
            b = -1
            if m > 10:
                A = np.vstack([[cartesian[i][1] for i in close_enough[:close_enough_size]], np.ones(close_enough_size)]).T
                b, c = np.linalg.lstsq(A, np.array([cartesian[i][0] for i in close_enough[:close_enough_size]]), rcond=None)[0]
                a = -1
            
            line_points = sorted((closest_to_line(*cartesian[i], a, b, c) for i in close_enough[:close_enough_size]), key=lambda i: i[0])
            features.append((a, b, c, (line_points[0], line_points[-1])))
            
            for point in close_enough[:close_enough_size]:
                try:
                    available.remove(point)
                except ValueError:
                    pass
    return features

def extract_landmarks(ls):
    features = extract_features(ls)
    landmarks = []
    for a, b, c, (start, end) in features:
        landmark_pos = np.array(closest_to_line(0, 0, a, b, c))  
        isnew = True
        for *_, landmark in landmarks:
            if np.linalg.norm(landmark - landmark_pos) < 0.25:
                isnew = False
                break
        if isnew:
            landmarks.append((a, b, c, (start, end), landmark_pos))
    return landmarks