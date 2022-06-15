import math
import random
from re import L

import numpy as np


def closest_to_line(x, y, a, b, c):
    return ((b * (b * x - a * y) - a * c) / (a**2 + b**2), (a * (-b * x + a * y) - b * c) / (a**2 + b**2))


class Landmark:
    """This class represents a landmark in the robot's environment.
    A landmark is a line described by `ax + by + c = 0`.
    """
    __slots__ = ("_equation", "_start", "_end", "count", "_r2")
    
    def __init__(self, a, b, c, start, end, r2=0):
        """Instantiate a new Landmark object, representing a line described by `ax + by + c = 0`, starting at `start` and ending at `end`."""
        self._equation = np.array((a, b, c))
        self._start = np.array(start)
        self._end = np.array(end)
        self.count = 1
        self._r2 = r2
        
    def __repr__(self):
        return f"<Landmark {np.round(self.params(), 3)} c={self.count}>"
    
    def copy(self):
        new_landmark = Landmark(*self.equation, self._start, self._end, self._r2)
        new_landmark.count = self.count
        return new_landmark
    
    def update(self, other):
        """Update the landmark's parameters using a new observation"""
        if self._r2 < other._r2:
            self._equation = other.equation
            self._start = other._start
            self._end = other._end
        self.count += other.count
        
    def update_params(self, params):
        new_params = self.params() + params
        new_params[1] %= (2 * np.pi)
        
        sign = 1 if 0 <= new_params[1] < np.pi else -1
        if not np.isclose(abs(new_params[1]), np.pi):
            b = -1
            a = b / np.tan(new_params[1])
            c = sign * new_params[0] * np.sqrt(1 + a*a) 
            self._start = np.array([0, c])
            self._end = np.array([1, a + c])
        else:
            a = -1
            b = a * np.tan(new_params[1])
            c = sign * new_params[0] * np.sqrt(1 + b*b)
            self._start = np.array([c, 0])
            self._end = np.array([b + c, 1])
        self._equation = np.array((a, b, c))
        
    def closest_point(self, x, y):
        """Compute the closest point on the line to (`x`, `y`)"""
        p = np.array(closest_to_line(x, y, *self._equation))
        return p

    @property
    def y_defined(self):
        """True if the line can be described by `y = mx + d`"""
        return self._equation[1] == -1
    
    @property
    def x_defined(self):
        """True if the line can be described by `x = my + d`"""
        return self._equation[0] == -1
    
    @property
    def equation(self):
        """The equation that describes the line 
        `array([a, b, c])
        """
        return self._equation
    
    def params(self, pose=None):
        """Line defined an angle and a distance to point `pose`. Defaults to the world origin."""
        if pose is None:
            pose = (0, 0, 0)
        
        a, b, c = self.equation
        if c != 0:
            theta = np.arctan2(- b * c / (a**2 + b**2), - a * c / (a**2 + b**2))
        else:
            theta = np.arctan2(- b / (a**2 + b**2), - a / (a**2 + b**2))
        r = abs(c) / np.sqrt(a**2 + b**2)
        d = r - pose[0] * np.cos(theta) - pose[1] * np.sin(theta)
        phi = theta - pose[2]
        return np.array([d, phi % (2 * np.pi)])
    
    @property
    def start(self):
        """Line segment start position"""
        return self._start
    
    @property
    def end(self):
        """Line segment end position"""
        return self._end

def to_cartesian(theta, r):
    if r > 0.001:
        return r * np.array((math.cos(theta), math.sin(theta)))
    else:
        return 5000 * np.array((math.cos(theta), math.sin(theta)))

def extract_features(ls, N=400, C=22, X=0.02, D=10, S=6):
    """Extract features from laser scan data `ls`. 
    `N`, `C`, `X`, `D`, and `S` are the parameters for the RANSAC algorithm.
    """
    features = []
    D = int(round(math.radians(D) / ls["angle_increment"])) 
    rsize = len(ls["ranges"])
    cartesian = tuple(to_cartesian(ls["angle_min"] + i * ls["angle_increment"], r) for i, r in enumerate(ls["ranges"]))
    available = [i for i in range(rsize) if ls["ranges"][i] > 0.001]
    
    # Pre-allocate lists
    total = [None] * rsize
    total_size = 0
    cartesian_sample_x = [0] * rsize
    cartesian_sample_y = [0] * rsize
    cartesian_sample_size = 0
    close_enough = [None] * rsize
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
            
        representation_type = None
        
        # Choose a random angle
        theta = random.random() * 2 * np.pi
        
        # Initialize rotation matrices for theta and -theta
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        inverse_rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
            
        # Rotate points by theta and fit a line through them
        points = np.array([point for point in zip(cartesian_sample_x[:cartesian_sample_size], cartesian_sample_y[:cartesian_sample_size])])
        points_transformed = [np.dot(rotation_matrix, points[i]) for i in range(len(points))]
        xl, yl = tuple(zip(*points_transformed))
        A = np.vstack([xl, np.ones(cartesian_sample_size)]).T
        (m, o), _ = np.linalg.lstsq(A, yl, rcond=None)[:2]
        p_start = np.dot(inverse_rotation_matrix, np.array([0, o]))
        p_end = np.dot(inverse_rotation_matrix, np.array([1, m + o]))
        if p_start[0] - p_end[0] != 0:
            m = (p_start[1] - p_end[1]) / (p_start[0] - p_end[0])
            representation_type = "y(x)"
        else:
            m = (p_start[0] - p_end[0]) / (p_start[1] - p_end[1])
            representation_type = "x(y)"
        o = p_start[1] - m * p_start[0]
        
        # Find all readings within X meters of the line
        denominator = 1 / pow(m*m + 1, 0.5)
        close_enough_size = 0
        for i, (x, y) in enumerate(cartesian):
            # Remove invalid points
            if ls["ranges"][i] < 0.01:
                continue
            if representation_type == "y(x)":
                distance = abs(m * x - y + o) * denominator
            else:
                distance = abs(m * y - x + o) * denominator
            if distance < X:
                close_enough[close_enough_size] = i
                close_enough_size += 1
        
        # If the number of points close to the line is above a threshold C
        if close_enough_size > C:
            xx = np.array([cartesian[i][0] for i in close_enough[:close_enough_size]])
            yy = np.array([cartesian[i][1] for i in close_enough[:close_enough_size]])
            
            # Rotate points by the same matrix as before
            points = np.array([point for point in zip(xx, yy)])
            points_transformed = [np.dot(rotation_matrix, points[i]) for i in range(len(points))]
            xl, yl = tuple(zip(*points_transformed))
            
            # Calculate new least squares best fit line
            A = np.vstack([xl, np.ones(close_enough_size)]).T
            (m, o), residual = np.linalg.lstsq(A, yl, rcond=None)[:2]
            r2 = 1 - float(residual / (len(yy) * np.var(yl)))
            
            # Discard poorly fitted lines
            if r2 < 0.9:
                continue
            
            # Transform points back into original reference frame
            p_start = np.dot(inverse_rotation_matrix, np.array([0, o]))
            p_end = np.dot(inverse_rotation_matrix, np.array([1, m + o]))
            if p_start[0] - p_end[0] != 0:
                m = (p_start[1] - p_end[1]) / (p_start[0] - p_end[0])
                representation_type = "y(x)"
            else:
                m = (p_start[0] - p_end[0]) / (p_start[1] - p_end[1])
                representation_type = "x(y)"
            o = p_start[1] - m * p_start[0]
            if representation_type == "y(x)":
                a = m
                b = -1
                c = o
            else:
                a = -1
                b = m
                c = o
                 
            features.append((a, b, c, (p_start, p_end), r2))
            
            for point in close_enough[:close_enough_size]:
                try:
                    available.remove(point)
                except ValueError:
                    pass
    return features

def extract_landmarks(ls, T=0.25, N=400, C=18, X=0.02, D=10, S=6):
    """Extract a list of landmarks from a laser scan `ls`"""
    features = extract_features(ls, N=N, C=C, X=X, D=D, S=S)
    landmarks = []
    for a, b, c, (start, end), r2 in features:
        nlandmark = Landmark(a, b, c, start, end, r2)
        lpos = nlandmark.closest_point(0, 0)
        for i, (landmark, oldr2) in enumerate(landmarks):
            # Only add landmarks sufficiently far apart
            if np.linalg.norm(landmark.closest_point(0, 0) - lpos) < T:
                if r2 > oldr2:
                    landmarks[i] = (nlandmark, r2)
                break
        else:
            landmarks.append((nlandmark, r2))
    # print(f"Seen landmarks: {len(landmarks)}")
    return [l[0]for l in landmarks]
