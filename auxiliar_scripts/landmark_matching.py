import numpy as np
import time

import landmark_extractor

LANDMARK_VAR = 0.05


class KalmanFilter:
    def __init__(self, landmark, initial_covariance, R):
        self.landmark = landmark
        self.covariance = initial_covariance
        self.R = R
        
    def update(self, z_measured, z_predicted, H):
        S = H.dot(self.covariance).dot(H.T) + self.R # * (2 - landmark.r2)
        K = self.covariance.dot(H.T).dot(np.linalg.inv(S))
        self.landmark.update_params(K.dot(z_measured - z_predicted))
        self.covariance += -K.dot(S).dot(K.T)

class LandmarkMatcher:
    def __init__(self, minimum_observations=6, distance_threshold=0.25, max_invalid_landmarks=None):
        self._landmarks = []
        self.minimum_observations = minimum_observations
        self.distance_threshold = distance_threshold
        self.max_invalid_landmarks = max_invalid_landmarks
        self.R = LANDMARK_VAR * np.identity(2)
        
    def copy(self):
        new_landmark_matcher = LandmarkMatcher(self.minimum_observations, self.distance_threshold, self.max_invalid_landmarks)
        for landmark, t, ekf in self._landmarks:
            copy_landmark = landmark.copy()
            new_landmark_matcher._landmarks.append([copy_landmark, t, KalmanFilter(copy_landmark, ekf.covariance, ekf.R)])
        return new_landmark_matcher
        
    def observe(self, landmark, H, pose):
        closest = None
        match = None
        
        worldspace_landmark = landmark_extractor.Landmark(*landmark.equation, landmark.start, landmark.end, landmark._r2)
        d, phi = landmark.params()
        theta = phi + pose[2]
        new_params = np.array([d + pose[0] * np.cos(theta) + pose[1] * np.sin(theta), theta])
        worldspace_landmark.update_params(new_params - np.array([d, phi]))
        
        for i, (glandmark, _, ekf) in enumerate(self._landmarks):
            # Compute the distance between projections on both landmarks
            ld = np.linalg.norm(worldspace_landmark.closest_point(*pose[:2]) - glandmark.closest_point(*pose[:2]))
            if ld < self.distance_threshold and (closest is None or ld < closest["difference"]):
                closest = {"difference": ld, "landmark": glandmark, "filter": ekf}
        if closest is not None:
            # Found a match
            closest["filter"].update(new_params, closest["landmark"].params(), H)
            closest["landmark"].count += 1
            self._landmarks[i][1] = time.time()
            
            # closest["landmark"].update(landmark)
            if closest["landmark"].count >= self.minimum_observations:
                match = closest["landmark"]
        else:
            cp = worldspace_landmark.copy()
            self._landmarks.append([cp, time.time(), KalmanFilter(cp, self.R, self.R)])
            # self._landmarks.append([landmark, time.time()]) 
        
        # Remove oldest lowest-seen invalid landmark
        if self.max_invalid_landmarks is not None and len(self._landmarks) - len(self.valid_landmarks) > self.max_invalid_landmarks:
            to_remove = None
            for i, (landmark, age, _) in enumerate(self._landmarks):
                if landmark.count < self.minimum_observations:
                    if to_remove is None:
                        to_remove = (i, age)
                    elif self._landmarks[to_remove[0]][0].count > landmark.count:
                        to_remove = (i, age)
                    elif self._landmarks[to_remove[0]][0].count == landmark.count and to_remove[1] > age:
                        to_remove = (i, age)
            if to_remove is not None:
                self._landmarks.pop(to_remove[0])            
        return match
    
    @property
    def landmarks(self):
        return tuple(map(lambda lt: lt[0], self._landmarks))
    
    @property
    def valid_landmarks(self):
        return tuple(filter(lambda l: l.count >= self.minimum_observations, map(lambda lt: lt[0], self._landmarks)))
