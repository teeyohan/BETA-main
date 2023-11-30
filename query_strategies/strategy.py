class Strategy:
    def __init__(self, X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device):
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_va = X_va
        self.Y_va = Y_va
        self.idxs_lb = idxs_lb
        self.device = device
        self.model = model
        self.args = args
        self.n_pool = len(Y_tr)
        self.query_count = 0

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def train(self, name):
        self.model.train(name, self.X_tr, self.Y_tr, self.X_va, self.Y_va, self.idxs_lb)

    def predict(self, X_va, Y_va, X_te, Y_te):
        return self.model.predict(X_va, Y_va, X_te, Y_te)

    def predict_prob_embed(self, X, Y, eval=True):
        return self.model.predict_prob_embed(X, Y, eval)

    def get_embedding(self, X, Y):
        return self.model.get_embedding(X, Y)
