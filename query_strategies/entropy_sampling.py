import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
		super(EntropySampling, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		if 'BACE' in self.args.data_name:
			probs, embeddings = self.predict_prob_embed([self.X_tr[i] for i in idxs_unlabeled], [self.Y_tr[i] for i in idxs_unlabeled])
		else:
			probs, embeddings = self.predict_prob_embed(self.X_tr[idxs_unlabeled], self.Y_tr[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		selected = U.sort()[1][:n]
		return idxs_unlabeled[selected]
