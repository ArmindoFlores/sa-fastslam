import numpy as np

import landmark_matching


class Particle:
    """A class describing a particle member of a particle filter.
    Contains its pose and weight.
    """
    def __init__(self, pose=None):
        """Instatiate a new `Particle` object, with `pose` as the starting pose."""
        if pose is None:
            self.pose = np.array([0, 0, 0], dtype=np.float64)
        else:
            self.pose = np.array(pose, dtype=np.float64)
        self.weight = 1
        self.landmark_matcher = landmark_matching.LandmarkMatcher(distance_threshold=0.2, max_invalid_landmarks=15)
        
    def __repr__(self):
        return f"<Particle pose={tuple(np.round(self.pose, 3))} weight={round(self.weight, 3)}>"

    def observe_landmark(self, landmark, H_func, Qt):
        """Try to match an observed landmark to list of previously seen ones and update its location 
        estimate and the particle's weight.
        """
        match = self.landmark_matcher.observe(landmark, H_func(*self.pose), self.pose)
        if match is not None:
            self.weigh(landmark.params(), match, H_func(*self.pose), Qt)
    
    def set_weight(self, weight):
        """Set the particle's weight to `weight`."""
        self.weight = weight

    def update_pose(self, odom, variance):
        """Update pose (AKA move the particle) according to odometry data `odom` and variance `variance`."""
        self.pose += np.random.normal(odom, np.sqrt(variance)).reshape(self.pose.shape)

    def weigh(self, z_measured, match, H, Qt):
        """Weigh the particle's importance after observing a new landmark.
        `z_measured` - the observed position of the landmark
        `match` - matched landmark
        `Qt` - laser measurement variance
        """
        landmark = match.landmark
        z_predicted = landmark.params(self.pose)
        Q = H.dot(match.covariance).dot(H.T) + Qt
        Z = z_measured - z_predicted
        self.weight *= np.exp(-0.5 * Z.T.dot(np.linalg.inv(Q)).dot(Z)) / np.sqrt(np.linalg.det(2 * np.pi * Q))
        
    def copy(self):
        new_particle = Particle(self.pose.copy())
        new_particle.landmark_matcher = self.landmark_matcher.copy()
        return new_particle

class ParticleFilter:
    """A class describing a particle filter."""
    def __init__(self, N, initial_pose=(0, 0, 0)):
        """Instatiate a new particle filter with `N` particles."""
        self.N = N
        self.particles = [Particle(initial_pose) for _ in range(N)]

    def sample_pose(self, odom, variance):
        """Update the pose of every particle according to odometry data `odom` and variance `variance`."""
        for particle in self.particles:
            particle.update_pose(odom, variance)

    def observe_landmarks(self, landmarks, H_func, Qt):
        """Inform every particle of landmark observations and update their weights accordingly."""
        for particle in self.particles:
            for landmark in landmarks:
                particle.observe_landmark(landmark, H_func, Qt)

    def resample(self, N=None, frac=0.8):
        """Compute the next generation of `N` particles based on the importance (weight) of the previous one.
        `frac` - the fraction of particles that are chosen based on weight. The rest are sampled randomly.
        """
        if N is None:
            N = self.N
        # Not the best method
        n1 = round(frac * N)
        n2 = N - n1
        if np.std(np.array([particle.weight for particle in self.particles])) < 1:
            return
        total_weight = sum((particle.weight for particle in self.particles))
        if total_weight == 0 or np.isnan(total_weight):
            for particle in self.particles:
                particle.set_weight(1)
            return
        normalized_weights = [particle.weight / total_weight for particle in self.particles]
        self.particles = list(np.random.choice(self.particles, n1, True, normalized_weights)) \
                       + list(np.random.choice(self.particles, n2, True)) 
        self.particles = [particle.copy() for particle in self.particles]
