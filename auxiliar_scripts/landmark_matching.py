import numpy as np
import time

import landmark_extractor


class KalmanFilter:
    def __init__(self, landmark, initial_covariance, Qt):
        self.landmark = landmark
        self.covariance = initial_covariance
        self.Qt = Qt

    def __repr__(self):
        return f"<EKF landmark={self.landmark} cov={np.linalg.norm(self.covariance)}>"
        
    def update(self, z_measured, z_predicted, H):
        Q = H.dot(self.covariance).dot(H.T) + self.Qt # * (2 - landmark.r2)
        K = self.covariance.dot(H.T).dot(np.linalg.inv(Q))
        diff = z_measured - z_predicted
        diff[1] = (diff[1] + np.pi) % (np.pi * 2) - np.pi
        self.landmark.update_params(K.dot(diff))
        self.covariance -= K.dot(H).dot(self.covariance)

class LandmarkMatcher:
    def __init__(self, Qt, minimum_observations=6, distance_threshold=0.3, max_invalid_landmarks=None):
        self._landmarks = []
        self.minimum_observations = minimum_observations
        self.distance_threshold = distance_threshold
        self.max_invalid_landmarks = max_invalid_landmarks
        self.Qt = Qt
        
    def copy(self):
        new_landmark_matcher = LandmarkMatcher(self.Qt.copy(), self.minimum_observations, self.distance_threshold, self.max_invalid_landmarks)
        for t, ekf in self._landmarks:
            landmark = ekf.landmark
            copy_landmark = landmark.copy()
            new_landmark_matcher._landmarks.append([t, KalmanFilter(copy_landmark, ekf.covariance.copy(), ekf.Qt.copy())])
        return new_landmark_matcher
        
    def observe(self, landmark, H, pose):
        closest = None
        match = None
        dsquared = self.distance_threshold**2
        
        d, phi = landmark.params()
        theta = phi + pose[2]
        new_params = np.array([d + pose[0] * np.cos(theta) + pose[1] * np.sin(theta), theta])
        worldspace_landmark = landmark_extractor.Landmark(*new_params, landmark.r2)
        p1 = worldspace_landmark.closest_point(*pose[:2])
        new_params = worldspace_landmark.params()
        
        for i, (_, ekf) in enumerate(self._landmarks):
            glandmark = ekf.landmark
            # Compute the distance between projections on both landmarks
            ld = np.sum(np.square(p1 - glandmark.closest_point(*pose[:2])))
            # rdiff = new_params[0] - glandmark.params()[0]
            # tdiff = new_params[1] - glandmark.params()[1]
            # tdiff = (tdiff + np.pi) % (np.pi * 2) - np.pi  # Wrap angles
            # ld = np.sum(np.square([rdiff, tdiff]))
            if ld < dsquared and (closest is None or ld < closest["difference"]):
                closest = {"difference": ld, "filter": ekf}
        if closest is not None:
            # Found a match
            closest["filter"].update(new_params, closest["filter"].landmark.params(), H)
            closest["filter"].landmark.count += 1
            self._landmarks[i][0] = time.time()
            
            if closest["filter"].landmark.count >= self.minimum_observations:
                match = closest["filter"]
        else:
            cp = worldspace_landmark.copy()
            self._landmarks.append([time.time(), KalmanFilter(cp, np.linalg.inv(H).T.dot(self.Qt).dot(np.linalg.inv(H)), self.Qt)])
            # self._landmarks.append([landmark, time.time()]) 
        
        # Remove oldest lowest-seen invalid landmark
        if self.max_invalid_landmarks is not None and len(self._landmarks) - len(self.valid_landmarks) > self.max_invalid_landmarks:
            to_remove = None
            for i, (age, ekf) in enumerate(self._landmarks):
                landmark = ekf.landmark
                if landmark.count < self.minimum_observations:
                    if to_remove is None:
                        to_remove = (i, age)
                    elif self._landmarks[to_remove[0]][1].landmark.count == landmark.count and to_remove[1] < age:
                        to_remove = (i, age)
            if to_remove is not None:
                self._landmarks.pop(to_remove[0])
        return match
    
    @property
    def landmarks(self):
        return tuple(map(lambda lt: lt[1], self._landmarks))
    
    @property
    def valid_landmarks(self):
        return tuple(filter(lambda l: l.landmark.count >= self.minimum_observations, map(lambda lt: lt[1], self._landmarks)))
