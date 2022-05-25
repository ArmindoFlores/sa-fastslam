import numpy as np


class Particle:
	def __init__(self, pose=None):
		if pose is None:
			self.pose = np.array([0, 0, 0])
		else:
			self.pose = np.array(pose)
		self.weight = 1
	
	def update_pose(self, odom, variance):
		self.pose += np.random.normal(odom, np.sqrt(variance))
	
class ParticleFilter:
	def __init__(self, N):
		self.N = N
		self.particles = [Particle() for _ in range(N)]

	def sample_pose(self, odom, variance):
		for particle in self.particles:
			particle.update_pose(odom, variance)

	def weigh_particles(self, H, covariance, Qt, zt, zt_estm):
		Q = H * covariance * H.T + Qt
		w = np.exp(-0.5 * (zt - zt_estm).T * np.linalg.inv(Q) * (zt - zt_estm)) / np.sqrt(np.absolute(2 * np.pi * Q))
		for i, particle in enumerate(self.particles):
			particle.weight = w[i]
	
	def resample(self, N, frac=0.8):
		# Not the best method
		n1 = round(frac * N)
		n2 = N - n1
		self.particles = list(np.random.choice(self.particles, n1, True, [particle.weight for particle in self.particles])) \
					   + list(np.random.choice(self.particles, n2, True)) 


