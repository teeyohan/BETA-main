import copy
import math
import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


class AlphaMixSampling(Strategy):
	def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
		super(AlphaMixSampling, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)

	def query(self, n, idxs_prohibited=None):
		self.query_count += 1

		idxs = self.idxs_lb if idxs_prohibited is None else (self.idxs_lb + idxs_prohibited)
		idxs_unlabeled = np.arange(self.n_pool)[~idxs]
		
		if 'BACE' in self.args.data_name:
			ulb_probs, org_ulb_embedding = self.predict_prob_embed([self.X_tr[i] for i in idxs_unlabeled], [self.Y_tr[i] for i in idxs_unlabeled])
		else:
			ulb_probs, org_ulb_embedding = self.predict_prob_embed(self.X_tr[idxs_unlabeled], self.Y_tr[idxs_unlabeled])

		_, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]

		if 'BACE' in self.args.data_name:
			lb_probs, org_lb_embedding = self.predict_prob_embed([self.X_tr[i] for i in self.idxs_lb], [self.Y_tr[i] for i in self.idxs_lb])
		else:
			lb_probs, org_lb_embedding = self.predict_prob_embed(self.X_tr[self.idxs_lb], self.Y_tr[self.idxs_lb])

		ulb_embedding = org_ulb_embedding
		lb_embedding = org_lb_embedding

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

		grads = None
		alpha_cap = 0.
		while alpha_cap < 1.0:
			alpha_cap += 0.03125
			if 'BACE' in self.args.data_name:
				tmp_pred_change = self.find_candidate_set(
						lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
						Y = torch.LongTensor([int(self.Y_tr[i]) for i in self.idxs_lb]),
						grads=grads)
			else:
				tmp_pred_change = self.find_candidate_set(
						lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
						Y = self.Y_tr[self.idxs_lb],
						grads=grads)

			candidate += tmp_pred_change

			print('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

			if candidate.sum() > n:
				break

		if candidate.sum() > 0:
			print('Number of inconsistencies: %d' % (int(candidate.sum().item())))
			c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
			selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)
			selected_idxs = idxs_unlabeled[candidate][selected_idxs]
		else:
			selected_idxs = np.array([], dtype=np.int)

		if len(selected_idxs) < n:
			remained = n - len(selected_idxs)
			idx_lb = copy.deepcopy(self.idxs_lb)
			idx_lb[selected_idxs] = True
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])
			print('picked %d samples from RandomSampling.' % (remained))

		return np.array(selected_idxs)

	def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

		for i in range(self.args.n_label):
			emb = lb_embedding[Y == i]
			if emb.size(0) == 0:
				emb = lb_embedding
			anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

			alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
			# if self.args.alpha_opt:
			# 	alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap, log_prefix=str(i))
			# else:
			embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
			out, _ = self.model.clf(embedding_mix.to(self.device), embedding=True)
			out = out.detach().cpu()

			pc = out.argmax(dim=1) != pred_1

			torch.cuda.empty_cache()
			pred_change[pc] = True

		return pred_change

	def sample(self, n, feats):
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)
		return np.array(
			[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
				(cluster_idxs == i).sum() > 0])

	def generate_alpha(self, size, embedding_size, alpha_cap):
		alpha = torch.normal(
			mean=alpha_cap / 2.0,
			std=alpha_cap / 2.0,
			size=(size, embedding_size))

		alpha[torch.isnan(alpha)] = 1

		return self.clamp_alpha(alpha, alpha_cap)

	def clamp_alpha(self, alpha, alpha_cap):
		return torch.clamp(alpha, min=1e-8, max=alpha_cap)

	def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
		labels = labels.to(self.device)
		min_alpha = torch.ones(alpha.size(), dtype=torch.float)
		pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

		loss_func = torch.nn.CrossEntropyLoss(reduction='none')

		self.model.clf.eval()

		for i in range(5):
			tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
			for b in range(math.ceil(float(alpha.size(0)) / 1000000)):
				self.model.clf.zero_grad()
				start_idx = b * 1000000
				end_idx = min((b + 1) * 1000000, alpha.size(0))

				l = alpha[start_idx:end_idx]
				l = torch.autograd.Variable(l.to(self.device), requires_grad=True)
				opt = torch.optim.Adam([l], lr=0.1 / (1. if i < 5 * 2 / 3 else 10.))
				e = org_embed[start_idx:end_idx].to(self.device)
				c_e = anchor_embed[start_idx:end_idx].to(self.device)
				embedding_mix = (1 - l) * e + l * c_e

				out, _ = self.model.clf(embedding_mix, embedding=True)

				label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

				tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
				tmp_pc[start_idx:end_idx] = label_change
				pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

				tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
				min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

				clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))

				l2_nrm = torch.norm(l, dim=1)

				clf_loss *= -1

				loss = 1.0 * clf_loss + 0.01 * l2_nrm
				loss.sum().backward(retain_graph=True)
				opt.step()

				l = self.clamp_alpha(l, alpha_cap)

				alpha[start_idx:end_idx] = l.detach().cpu()

				tot_clf_loss += clf_loss.mean().item() * l.size(0)
				tot_loss += loss.mean().item() * l.size(0)
				tot_nrm += l2_nrm.mean().item() * l.size(0)

				del l, e, c_e, embedding_mix
				torch.cuda.empty_cache()

		return min_alpha.cpu(), pred_changed.cpu()