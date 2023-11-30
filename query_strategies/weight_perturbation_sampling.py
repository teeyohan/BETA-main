import copy
import math
import random
import numpy as np

from .strategy import Strategy
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


class WeightPerturbationSampling(Strategy):
    def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
        super(WeightPerturbationSampling, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)

    def query(self, n, idxs_prohibited=None):
        self.query_count += 1

        idxs = self.idxs_lb if idxs_prohibited is None else (self.idxs_lb + idxs_prohibited)
        idxs_unlabeled = np.arange(self.n_pool)[~idxs]
        if 'BACE' in self.args.data_name:
            ulb_probs, org_ulb_embedding = self.predict_prob_embed([self.X_tr[i] for i in idxs_unlabeled], [self.Y_tr[i] for i in idxs_unlabeled])
        else:
            ulb_probs, org_ulb_embedding = self.predict_prob_embed(self.X_tr[idxs_unlabeled], self.Y_tr[idxs_unlabeled])
        probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
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

        grads_weight = None
        grads_bias = None

        eps_cap = 0.
        while eps_cap < 100000:
            eps_cap += self.args.eps_cap
            if 'BACE' in self.args.data_name:
                tmp_pred_change = self.find_candidate_set(lb_embedding, ulb_embedding, pred_1, ulb_probs, eps_cap=eps_cap,
                        Y = torch.LongTensor([int(self.Y_tr[i]) for i in self.idxs_lb]),
                        grads_weight = grads_weight, grads_bias = grads_bias)
            else:
                tmp_pred_change = self.find_candidate_set(lb_embedding, ulb_embedding, pred_1, ulb_probs, eps_cap=eps_cap,
                        Y = self.Y_tr[self.idxs_lb],
                        grads_weight = grads_weight, grads_bias = grads_bias)

            candidate += tmp_pred_change
            print('With eps_cap set to %.5f, number of inconsistencies: %d' % (eps_cap, int(tmp_pred_change.sum().item())))

            if candidate.sum() > n: # 超过查询数
                break

        if candidate.sum() > 0:
            print('Number of inconsistencies: %d' % (int(candidate.sum().item())))
            if self.args.eps_select == 'random':
                selected_idxs = self.sample_random(min(n, candidate.sum().item()), int(candidate.sum().item()))
            elif self.args.eps_select == 'entropy':
                c_pbs = ulb_probs[candidate].detach()
                selected_idxs = self.sample_entropy(min(n, candidate.sum().item()), probs=c_pbs)
            elif self.args.eps_select == 'kmeans':
                c_eps = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
                selected_idxs = self.sample_kmeans(min(n, candidate.sum().item()), feats=c_eps)
            # elif self.args.eps_select == 'kmedoids':
                # c_eps = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
                # selected_idxs = self.sample_kmedoids(min(n, candidate.sum().item()), feats=c_eps)

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

    def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, eps_cap, Y, grads_weight, grads_bias):

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

        eps_weight, eps_bias = self.generate_eps(self.args.n_label, embedding_size, eps_cap)

        if self.args.eps_get == 'accumulated': # 累积扰动
            if self.model.clf.state_dict().get('classifier.1.weight') != None:
                weight = self.model.clf.state_dict()['classifier.1.weight'].detach().cpu() + eps_weight
                bias = self.model.clf.state_dict()['classifier.1.bias'].detach().cpu() + eps_bias
            else:
                weight = self.model.clf.state_dict()['classifier.linear.weight'].detach().cpu() + eps_weight
                bias = self.model.clf.state_dict()['classifier.linear.bias'].detach().cpu() + eps_bias
            out = torch.matmul(ulb_embedding, weight.T) + bias
            out = out.detach().cpu()
            pc = out.argmax(dim=1) != pred_1
        elif self.args.eps_get == 'max_optimized': # 最大损失优化扰动
            pc = self.learn_eps(ulb_embedding, pred_1, eps_weight, eps_bias, eps_cap, True)
        elif self.args.eps_get == 'min_optimized': # 最小损失优化扰动
            pc = self.learn_eps(ulb_embedding, pred_1, eps_weight, eps_bias, eps_cap, False)

        torch.cuda.empty_cache()
        pred_change[pc] = True

        return pred_change

    def sample_random(self, n, m):
        return np.array(random.sample(range(m), n))

    def sample_entropy(self, n, probs):
        return np.array((probs*torch.log(probs)).sum(1).sort()[1][:n])
    
    def sample_kmeans(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)
        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()]
            for i in range(n) if(cluster_idxs == i).sum() > 0])
    
    # def sample_kmedoids(self, n, feats):
    #     feats = feats.numpy()
    #     cluster_learner = KMedoids(n_clusters=n)
    #     cluster_learner.fit(feats)
    #     return np.array(cluster_learner.medoid_indices_)

    def generate_eps(self, size, embedding_size, eps_cap):
        eps_weight = torch.normal(mean=0, std=eps_cap/2, size=(size, embedding_size))
        eps_bias = torch.normal(mean=0, std=eps_cap/2, size=(size,))
        return self.clamp_eps(eps_weight, eps_cap), self.clamp_eps(eps_bias, eps_cap)

    def clamp_eps(self, eps, eps_cap):
        return torch.clamp(eps, min=-eps_cap, max=eps_cap)

    def learn_eps(self, org_embed, labels, eps_weight, eps_bias, eps_cap, flag):
        self.model.clf.eval()
        labels = labels.to(self.device)
        pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(self.args.eps_learning_iters):
            tot_loss = 0.
            for b in range(math.ceil(float(org_embed.size(0)) / self.args.eps_learning_batch_size)):
                start_idx = b * self.args.eps_learning_batch_size
                end_idx = min((b + 1) * self.args.eps_learning_batch_size, org_embed.size(0))

                if self.model.clf.state_dict().get('classifier.1.weight') != None:
                    weight = self.model.clf.state_dict()['classifier.1.weight'].detach().cpu() + eps_weight
                    bias = self.model.clf.state_dict()['classifier.1.bias'].detach().cpu() + eps_bias
                else:
                    weight = self.model.clf.state_dict()['classifier.linear.weight'].detach().cpu() + eps_weight
                    bias = self.model.clf.state_dict()['classifier.linear.bias'].detach().cpu() + eps_bias
                weight = torch.autograd.Variable(weight.to(self.device), requires_grad=False)
                bias = torch.autograd.Variable(bias.to(self.device), requires_grad=False)
                l_weight = torch.autograd.Variable(eps_weight.to(self.device), requires_grad=True)
                l_bias = torch.autograd.Variable(eps_bias.to(self.device), requires_grad=True)
                _weight = weight + l_weight
                _bias = bias + l_bias

                opt = torch.optim.Adam([l_weight, l_bias], lr=self.args.eps_learning_rate / (1. if i < self.args.eps_learning_iters * 2 / 3 else 10.))

                e = org_embed[start_idx:end_idx].to(self.device)
                out = torch.matmul(e, _weight.T) + _bias

                label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

                tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
                tmp_pc[start_idx:end_idx] = label_change
                pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()
                if flag:
                    clf_loss = -1 * loss_func(out, labels[start_idx:end_idx].to(self.device))
                else:
                    clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))
                clf_loss.sum().backward(retain_graph=True)
                opt.step()

                l_weight = self.clamp_eps(l_weight, eps_cap)
                l_bias = self.clamp_eps(l_bias, eps_cap)

                eps_weight = l_weight.detach().cpu()
                eps_bias = l_bias.detach().cpu()

                tot_loss += clf_loss.mean().item() * e.size(0)

                del weight, bias, l_weight, l_bias, _weight, _bias, e
                torch.cuda.empty_cache()

        return pred_changed.cpu()