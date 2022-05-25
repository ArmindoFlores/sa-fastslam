import numpy as np


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
        
    def __repr__(self):
        return f"<Particle pose={tuple(np.round(self.pose, 3))} weight={round(self.weight, 3)}>"

    def set_weight(self, weight):
        """Set the particle's weight to `weight`."""
        self.weight = weight

    def update_pose(self, odom, variance):
        """Update pose (AKA move the particle) according to odometry data `odom` and variance `variance`."""
        self.pose += np.random.normal(odom, np.sqrt(variance)).reshape(self.pose.shape)    

    def weigh(self, zt, landmark, H, covariance, Qt):
        """Weigh the particle's importance after observing a new landmark.
        `zt` - the observed position of the landmark
        `landmark` - matched landmark
        `H`, `covariance`, `Qt` - parameters???
        """
        zt_estm = self.get_estimate(landmark)
        Q = H.dot(covariance).dot(H.T) + Qt
        w = np.exp(-0.5 * (zt - zt_estm).T.dot(np.linalg.inv(Q)).dot((zt - zt_estm))) / np.sqrt(np.linalg.det(2 * np.pi * Q))
        self.weight *= w
        
    def get_estimate(self, landmark):
        """Compute the landmark position we should observe given our position estimate"""
        if landmark.equation[0] != 0:
            theta = np.arctan(landmark.equation[1] / landmark.equation[0])
        else:
            theta = np.arctan(landmark.equation[0] / landmark.equation[1])
        r = abs(landmark.equation[2]) / np.sqrt(landmark.equation[0]**2 + landmark.equation[1]**2)
        d = r - self.pose[0] * np.cos(theta) - self.pose[1] * np.sin(theta)
        phi = theta - self.pose[2]
        return np.array([d, phi])

class ParticleFilter:
    """A class describing a particle filter."""
    def __init__(self, N):
        """Instatiate a new particle filter with `N` particles."""
        self.N = N
        self.particles = [Particle() for _ in range(N)]

    def sample_pose(self, odom, variance):
        """Update the pose of every particle according to odometry data `odom` and variance `variance`."""
        for particle in self.particles:
            particle.update_pose(odom, variance)

    def weigh_particles(self, zt, landmark, H, covariance, Qt):
        """Weigh every particle based on an observation `zt` of `landmark`."""
        for particle in self.particles:
            particle.weigh(zt, landmark, H, covariance, Qt)

    def resample(self, N, frac=0.8):
        """Compute the next generation of `N` particles based on the importance (weight) of the previous one.
        `frac` - the fraction of particles that are chosen based on weight. The rest are sampled randomly.
        """
        # Not the best method
        n1 = round(frac * N)
        n2 = N - n1
        total_weight = sum((particle.weight for particle in self.particles))
        normalized_weights = [particle.weight / total_weight for particle in self.particles]
        self.particles = list(np.random.choice(self.particles, n1, True, normalized_weights)) \
                       + list(np.random.choice(self.particles, n2, True)) 
