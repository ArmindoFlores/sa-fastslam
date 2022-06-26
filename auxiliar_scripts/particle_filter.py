import numpy as np

import landmark_matching


class Particle:
    """A class describing a particle member of a particle filter.
    Contains its pose and weight.
    """
    def __init__(self, Qt, H_func, pose=None, **landmark_matcher_kwargs):
        """Instatiate a new `Particle` object, with `pose` as the starting pose."""
        if pose is None:
            self.pose = np.array([0, 0, 0], dtype=np.float64)
        else:
            self.pose = np.array(pose, dtype=np.float64)
        self.weight = 1
        self.Qt = Qt
        self.H_func = H_func
        self.landmark_matcher = landmark_matching.LandmarkMatcher(Qt=Qt, **landmark_matcher_kwargs)
        
    def __repr__(self):
        return f"<Particle pose={tuple(np.round(self.pose, 3))} weight={round(self.weight, 3)}>"

    def observe_landmark(self, landmark):
        """Try to match an observed landmark to list of previously seen ones and update its location 
        estimate and the particle's weight.
        """
        match = self.landmark_matcher.observe(landmark, self.H_func, self.pose)
        if match is not None:
            self.weigh(landmark.params(), match)
            return True
        return False
    
    def set_weight(self, weight):
        """Set the particle's weight to `weight`."""
        self.weight = weight

    def update_pose(self, r, theta, variance):
        """Update pose (AKA move the particle) according to odometry data `(r, theta)` and variance `variance`."""
        r_estimate = np.random.normal(r, np.sqrt(variance[0]))
        theta_estimate = np.random.normal(theta, np.sqrt(variance[2]))
        odom = np.array([r_estimate * np.cos(self.pose[2]), r_estimate * np.sin(self.pose[2]), theta_estimate])
        self.pose += odom

    def weigh(self, z_measured, match):
        """Weigh the particle's importance after observing a new landmark.
        `z_measured` - the observed position of the landmark
        `match` - matched landmark
        `Qt` - laser measurement variance
        """
        landmark = match.landmark
        z_predicted = landmark.params(self.pose)
        H = self.H_func(*self.pose[:2], landmark.params()[1])
        Q = H.dot(match.covariance).dot(H.T) + self.Qt
        Z = z_measured - z_predicted
        self.weight *= np.exp(-0.5 * Z.T.dot(np.linalg.inv(Q)).dot(Z)) / np.sqrt(np.linalg.det(2 * np.pi * Q))
        
    def copy(self):
        new_particle = Particle(self.Qt.copy(), self.H_func, self.pose.copy())
        new_particle.landmark_matcher = self.landmark_matcher.copy()
        return new_particle


class ParticleFilter:
    """A class describing a particle filter."""
    def __init__(self, N, Qt, H_func, initial_pose=(0, 0, 0), **landmark_matcher_kwargs):
        """Instatiate a new particle filter with `N` particles."""
        self.N = N
        self.particles = [Particle(Qt, H_func, initial_pose, **landmark_matcher_kwargs) for _ in range(N)]

    def sample_pose(self, odom, variance):
        """Update the pose of every particle according to odometry data `odom` and variance `variance`."""
        r = np.linalg.norm(odom[:2])
        theta = odom[2]
        for particle in self.particles:
            particle.update_pose(r, theta, variance)

    def observe_landmarks(self, landmarks):
        """Inform every particle of landmark observations and update their weights accordingly."""
        total = 0
        max_match = 0
        for particle in self.particles:
            total_matches = 0
            for landmark in landmarks:
                if particle.observe_landmark(landmark):
                    total_matches += 1
            if total_matches > max_match:
                max_match = total_matches
            total += total_matches
        return total, max_match

    def resample(self, N=None, frac=1):
        """Compute the next generation of `N` particles based on the importance (weight) of the previous one.
        `frac` - the fraction of particles that are chosen based on weight. The rest are sampled randomly.
        """
        if N is None:
            N = self.N
        # Not the best method
        n1 = round(frac * N)
        n2 = N - n1
        total_weight = sum((particle.weight for particle in self.particles))
        if total_weight == 0 or np.isnan(total_weight):
            for particle in self.particles:
                particle.set_weight(1)
            return
        normalized_weights = [particle.weight / total_weight for particle in self.particles]
        # if np.std(normalized_weights) < 0.01:
        #     return
        self.particles = list(np.random.choice(self.particles, n1, True, normalized_weights)) \
                       + list(np.random.choice(self.particles, n2, True)) 
        self.particles = [particle.copy() for particle in self.particles]
