import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
	def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
		super(RandomSampling, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)
