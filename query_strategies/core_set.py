import numpy as np
from .strategy import Strategy
from sklearn.metrics import pairwise_distances


class CoreSet(Strategy):
    def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
        super(CoreSet, self).__init__(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)
        self.tor = 1e-4

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()

        embed = self.get_embedding(self.X_tr, self.Y_tr)
        embedding = embed.numpy()

        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen]
