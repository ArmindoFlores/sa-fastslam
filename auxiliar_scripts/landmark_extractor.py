import math
import random

import numpy as np


def closest_to_line(x, y, a, b, c):
    return ((b * (b * x - a * y) - a * c) / (a**2 + b**2), (a * (-b * x + a * y) - b * c) / (a**2 + b**2))

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def distance_on_line(p, s, f):
	return np.dot(np.linalg.pinv((f - s).reshape((2, 1))), (s - p).reshape(2, 1)), np.dot(np.linalg.pinv((f - s).reshape((2, 1))), (f - p).reshape(2, 1))

class Landmark:
    """This class represents a landmark in the robot's environment.
    A landmark is a line described by `ax + by + c = 0`.
    """
    __slots__ = ("_equation", "_start", "_end", "_count")
    
    def __init__(self, a, b, c, start, end):
        """Instantiate a new Landmark object, representing a line described by `ax + by + c = 0`, starting at `start` and ending at `end`."""
        self._equation = np.array((a, b, c))
        self._start = np.array(start)
        self._end = np.array(end)
        self._count = 1
    
    def update(self, other):
        """Update the landmark's parameters using a new observation"""
        assert self.equation[0] * other.equation[0] + self.equation[1] * other.equation[1] != 0
        
        if self.y_defined and other.y_defined or self.x_defined and other.x_defined:
            self._equation = (self._equation * self._count + other.equation * other.count) / (self._count + other.count)
        elif self.y_defined:
            print("!!!")
            a, b, c = other.equation
            self._equation = (self._equation * self._count + np.array((1/b, -1, -c/b)) * other.count) / (self._count + other.count)
        else:
            print("!!!")
            a, b, c = other.equation
            self._equation = (self._equation * self._count + np.array((-1, 1/a, -c/a)) * other.count) / (self._count + other.count)
        
        self._count += other.count
        other_start = self.closest_point(*other.start, False)
        other_end = self.closest_point(*other.end, False)
        if self.y_defined:
            if other_start[0] < self._start[0]:
                self._start = other_start
            if other_end[0] > self._end[0]:
                self._end = other_end
        elif self.x_defined:
            if other_start[1] < self._start[1]:
                self._start = other_start
            if other_end[1] > self._end[1]:
                self._end = other_end
        
    def closest_point(self, x, y, check_bounds=True):
        """Compute the closest point on the line to (`x`, `y`)"""
        p = np.array(closest_to_line(x, y, *self._equation))
        if not check_bounds:
            return p
        d1, d2 = distance_on_line(p, self.start, self.end)
        if np.sign(d1) != np.sign(d2):
            return p
        if np.sign(d1) <= 0:
            return self.end
        return self.start
    
    def distance(self, other, pos=(0, 0)):
        """A measure of how different two landmarks are"""
        return np.linalg.norm(self.closest_point(*pos) - other.closest_point(*pos))

    def intersects(self, other, threshold=0.3):
        """Check if this landmark intersects with `other`."""
        if self.equation[1] != 0:
            self_d_vector = np.array([1, -self.equation[0]/self.equation[1]])
        else:
            self_d_vector = np.array([-self.equation[1]/self.equation[0], 1])
        if other.equation[1] != 0:
            other_d_vector = np.array([1, -other.equation[0]/other.equation[1]])
        else:
            other_d_vector = np.array([-other.equation[1]/other.equation[0], 1])
        self_d_vector /= np.linalg.norm(self_d_vector)
        other_d_vector /= np.linalg.norm(other_d_vector)
        
        self_start = self.start - threshold * self_d_vector
        self_end = self.end + threshold * self_d_vector
        other_start = other.start - threshold * other_d_vector
        other_end = other.end + threshold * other_d_vector
        
        return all((
            ccw(self_start, other_start, other_end) != ccw(self_end, other_start, other_end),
            ccw(self_start, self_end, other_start) != ccw(self_start, self_end, other_end)
        ))
        # return any((
        #     all((
        #         any((
        #             (other.start[0] - threshold < self.start[0] < other.end[0] + threshold),
        #             (other.start[0] - threshold < self.end[0] < other.end[0] + threshold),
        #             (self.start[0] - threshold < other.start[0] < self.end[0] + threshold),
        #             (self.start[0] - threshold < other.end[0] < self.end[0] + threshold)
        #         )),
        #         any((
        #             (other.start[1] - threshold < self.start[1] < other.end[1] + threshold),
        #             (other.start[1] - threshold < self.end[1] < other.end[1] + threshold),
        #             (self.start[1] - threshold < other.start[1] < self.end[1] + threshold),
        #             (self.start[1] - threshold < other.end[1] < self.end[1] + threshold)
        #         ))
        #     )),
        # ))
        
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
    
    @property
    def count(self):
        """How many times the landmark has been observed"""
        return self._count
    
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
        return 500 * np.array((math.cos(theta), math.sin(theta)))

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
    """Extract a list of landmarks from a laser scan `ls`"""
    features = extract_features(ls)
    landmarks = []
    for a, b, c, (start, end) in features:
        nlandmark = Landmark(a, b, c, start, end)
        lpos = nlandmark.closest_point(0, 0, False)
        isnew = True
        for landmark in landmarks:
            # Only add landmarks sufficiently far apart
            if np.linalg.norm(landmark.closest_point(0, 0, False) - lpos) < 0.25:
                isnew = False
                break
        if isnew:
            landmarks.append(nlandmark)
    return landmarks
