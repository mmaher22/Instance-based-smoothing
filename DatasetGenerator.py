import numpy as np
from scipy.stats import bernoulli, cauchy, beta, uniform

class SyntheticDatasetGenerator(object):
	def __init__(self, type = 'bernoulli', p = 0.5, rs = 22):
		self.type = type
		self.p = p
		self.set_random_seed(rs)

	#Setters
	def set_random_seed(self, rs):
		self.rs = rs
		np.random.seed(seed = self.rs)

	#Number of instances Generator
	def generate_instances_numbers(self):
		if self.type == 'bernoulli':
			N0, N1 = self.bernoulli_generator()
		else:
			N0, N1 = self.equal_generator()
		return N0, N1
	def bernoulli_generator(self):
		ys = bernoulli.rvs(self.p, size = self.num_instances)
		N0 = np.sum(ys < 1)
		N1 = self.num_instances - N0
		return N0, N1
	def equal_generator(self):
		return self.num_instances/2, self.num_instances/2

class SyntheticGaussianGenerator(SyntheticDatasetGenerator):
	
	def set_gaussian_parameters(self,mu0,mu1,sigma0,sigma1):
		self.mu0 = mu0
		self.mu1 = mu1
		self.sigma0 = sigma0
		self.sigma1 = sigma1

	#Generate final dataset
	def generate(self, num_instances):
		self.num_instances = num_instances
		N0, N1 = self.generate_instances_numbers()
		S0 = np.random.normal(self.mu0, self.sigma0, N0)
		S1 = np.random.normal(self.mu1, self.sigma1, N1)
		X = np.array(np.concatenate((S0, S1), axis=0))
		y = np.asarray([0] * N0 + [1] * N1)
		return X, y
    
class SyntheticUniformGenerator(SyntheticDatasetGenerator):
	def set_uniform_parameters(self,a0,b0,a1,b1):
		self.a0 = a0; self.a1 = a1
		self.b0 = b0; self.b1 = b1
	#Generate final dataset
	def generate(self, num_instances):
		self.num_instances = num_instances
		N0, N1 = self.generate_instances_numbers()
		S0 = uniform.rvs(self.a0, self.b0 - self.b0, size = N0)#np.random.normal(self.mu0, self.sigma0, N0)
		S1 = uniform.rvs(self.a1, self.b1 - self.a1, size = N1)#np.random.normal(self.mu1, self.sigma1, N1)
		X = np.array(np.concatenate((S0, S1), axis=0))
		y = np.asarray([0] * N0 + [1] * N1)
		return X, y
    
class SyntheticBetaGenerator(SyntheticDatasetGenerator):
	def set_beta_parameters(self,loc0,loc1,scale0,scale1):
		self.loc0 = loc0
		self.loc1 = loc1
		self.scale0 = scale0
		self.scale1 = scale1
	#Generate final dataset
	def generate(self, num_instances):
		self.num_instances = num_instances
		N0, N1 = self.generate_instances_numbers()
		S0 = beta.rvs(2, 2, loc = self.loc0, scale = self.scale0, size = N0)
		S1 = beta.rvs(2, 2, loc = self.loc1, scale = self.scale1, size = N1)
		X = np.array(np.concatenate((S0, S1), axis=0))
		y = np.asarray([0] * N0 + [1] * N1)
		return X, y
    
class SyntheticCauchyGenerator(SyntheticDatasetGenerator):
	def set_cauchy_parameters(self, mu0, mu1, scale0, scale1):
		self.mu0 = mu0
		self.mu1 = mu1
		self.scale0 = scale0
		self.scale1 = scale1
	#Generate final dataset
	def generate(self, num_instances):
		self.num_instances = num_instances
		N0, N1 = self.generate_instances_numbers()
		S0 = cauchy.rvs(loc = self.mu0, scale = self.scale0, size = N0)#np.random.normal(self.mu0, self.sigma0, N0)
		S1 = cauchy.rvs(loc = self.mu1, scale = self.scale1, size = N1)#np.random.normal(self.mu1, self.sigma1, N1)
		X = np.array(np.concatenate((S0, S1), axis=0))
		y = np.asarray([0] * N0 + [1] * N1)
		return X, y