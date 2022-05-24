import numpy as np
import time


class LandmarkMatcher:
    def __init__(self, minimum_observations=6, distance_threshold=0.25, max_invalid_landmarks=None):
        self._landmarks = []
        self.minimum_observations = minimum_observations
        self.distance_threshold = distance_threshold
        self.max_invalid_landmarks = max_invalid_landmarks
        
    def observe(self, landmark):
        closest = None
        match = None
        for i, (glandmark, _) in enumerate(self._landmarks):
            # Compute the distance between projections on both landmarks
            ld = np.linalg.norm(landmark.closest_point(0, 0) - glandmark.closest_point(0, 0))
            if ld < self.distance_threshold and (closest is None or ld < closest["difference"]):
                closest = {"difference": ld, "landmark": glandmark}
        if closest is not None:
            closest["landmark"].update(landmark)
            self._landmarks[i][1] = time.time()
            if closest["landmark"].count >= self.minimum_observations:
                match = closest["landmark"]
        else:
            self._landmarks.append([landmark, time.time()]) 
        
        if self.max_invalid_landmarks is not None and len(self._landmarks) - len(self.valid_landmarks) > self.max_invalid_landmarks:
            # Remove oldest lowest-seen invalid landmark
            to_remove = None
            for i, (landmark, age) in enumerate(self._landmarks):
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
