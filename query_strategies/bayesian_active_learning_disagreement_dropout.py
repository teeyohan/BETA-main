import numpy as np
import torch
from .strategy import Strategy


class BALDDropout(Strategy):
	def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
		super(BALDDropout, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)
		self.n_drop = 5

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		if 'BACE' in self.args.data_name:
			probs = self.predict_prob_dropout_split([self.X_tr[i] for i in idxs_unlabeled], [self.Y_tr[i] for i in idxs_unlabeled], self.n_drop)
		else:
			probs = self.predict_prob_dropout_split(self.X_tr[idxs_unlabeled], self.Y_tr[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1

		selected = U.sort()[1][:n]
		return idxs_unlabeled[selected]
