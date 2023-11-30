import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GCNDataLoader

from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def grad_clipping(net, theta):
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Training(object):
    def __init__(self, net, net_args, handler, args, device, init_model=True):
        self.net = net
        self.net_args = net_args
        self.handler = handler

        self.args = args
        self.device = device

        if init_model:
            self.clf = self.net(**self.net_args).to(self.device)
            self.initial_state_dict = copy.deepcopy(self.clf.state_dict())

    def _traincnn(self, loader_tr, loader_va, optimizer, name):
        accTrain, lossTrain, accVal, lossVal = 0., 0., 0., 0.

        self.clf.train()
        for x, y, idxs in loader_tr:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            lossTrain += loss.item()
            accTrain += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            grad_clipping(self.clf, 1.)
            optimizer.step()

        self.clf.eval()
        for x, y, idxs in loader_va:
            x, y = x.to(self.device), y.to(self.device)
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            lossVal += loss.item()
            accVal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

        return lossTrain / len(loader_tr.dataset.X), lossVal / len(loader_va.dataset.X), accTrain / len(loader_tr.dataset.X), accVal / len(loader_va.dataset.X)

    def _trainlstm(self, loader_tr, loader_va, optimizer, name):
        accTrain, lossTrain, accVal, lossVal = 0., 0., 0., 0.

        self.clf.train()
        for x, y, idxs in loader_tr:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            lossTrain += loss.item()
            accTrain += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            grad_clipping(self.clf, 1.)
            optimizer.step()

        self.clf.eval()
        for x, y, idxs in loader_va:
            x, y = x.to(self.device), y.to(self.device)
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            lossVal += loss.item()
            accVal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

        return lossTrain / len(loader_tr.dataset.X) , lossVal / len(loader_va.dataset.X), accTrain / len(loader_tr.dataset.X), accVal / len(loader_va.dataset.X)
    
    def _traingcn(self, loader_tr, loader_va, optimizer, name):
        accTrain, lossTrain, totalTrain, accVal, lossVal, totalVal = 0., 0., 0., 0., 0., 0.

        self.clf.train()
        for data in loader_tr:
            data = data.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(data)
            loss = F.cross_entropy(out, data.y, reduction='sum')
            lossTrain += loss.item()
            accTrain += (out.argmax(1) == data.y).type(torch.float).sum().item()
            totalTrain += len(data.y)
            loss.backward()
            grad_clipping(self.clf, 1.)
            optimizer.step()

        self.clf.eval()
        for data in loader_va:
            data = data.to(self.device)
            out, e1 = self.clf(data)
            loss = F.cross_entropy(out, data.y, reduction='sum')
            lossVal += loss.item()
            accVal += (out.argmax(1) == data.y).type(torch.float).sum().item()
            totalVal += len(data.y)

        return lossTrain / totalTrain, lossVal / totalVal, accTrain / totalTrain, accVal / totalVal


    def train(self, name, X_tr, Y_tr, X_va, Y_va, idxs_lb):
        n_epoch = 200 if self.args['n_epoch'] <= 0 else self.args['n_epoch']

        if not self.args['continue_training']:
            self.clf = self.net(**self.net_args).to(self.device)
            self.clf.load_state_dict(copy.deepcopy(self.initial_state_dict))

        if self.args['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        else:
            optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])

        optimizer.zero_grad()
        idxs_train = np.arange(len(Y_tr))[idxs_lb]

        if self.net_args['arch_name'] == 'gcn':
            set_tr = self.handler(root='./data', dataset = 'train', X = [X_tr[i] for i in idxs_train], Y = [Y_tr[i] for i in idxs_train])
            set_va = self.handler(root='./data', dataset = 'val', X = X_va, Y = Y_va)
            loader_tr = GCNDataLoader(set_tr, shuffle=True, **self.args['loader_tr_args'])
            loader_va = GCNDataLoader(set_va, shuffle=False, **self.args['loader_va_args'])
        elif self.net_args['arch_name'] == 'lstm':
            loader_tr = DataLoader(self.handler(X_tr[idxs_train], Y_tr[idxs_train]),
                               shuffle=True, **self.args['loader_tr_args'])
            loader_va = DataLoader(self.handler(X_va, Y_va),
                               shuffle=False, **self.args['loader_va_args'])
        else:
            loader_tr = DataLoader(self.handler(X_tr[idxs_train], Y_tr[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])
            loader_va = DataLoader(self.handler(X_va, Y_va, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_va_args'])

        self.iter = 0
        self.best_model = None
        accBest, n_stop = 0., 0

        print('Training started...')
        for epoch in range(n_epoch):
            if self.net_args['arch_name'] == 'gcn':
                lossTrain, lossVal, accTrain, accVal = self._traingcn(loader_tr, loader_va, optimizer, name)
            elif self.net_args['arch_name'] == 'lstm':
                lossTrain, lossVal, accTrain, accVal = self._trainlstm(loader_tr, loader_va, optimizer, name)
            else:
                lossTrain, lossVal, accTrain, accVal = self._traincnn(loader_tr, loader_va, optimizer, name)

            print('Epoch %3d    lossTrain: %.6f    lossVal: %.6f    accTrain: %.4f    accVal %.4f    accBest %.4f'%(epoch, lossTrain, lossVal, accTrain, accVal,accBest))

            if accTrain >= 0.99:
                print('Early Stopping! Reached max training accuracy at epoch %d ' % epoch)
                break

            if accVal > accBest:
                accBest = accVal
                n_stop = 0
            else:
                n_stop += 1

            if n_stop >= self.args['n_early_stopping']:
                print('Early Stopping! Reached max validation accuracy at epoch %d ' % epoch)
                break

    def predict(self, X_va, Y_va, X_te, Y_te):
        if self.net_args['arch_name'] == 'gcn' and self.args['mode'] == 'testing':
            set_te = self.handler(root='./data', dataset = 'test', X = X_te, Y = Y_te)
            loader = GCNDataLoader(set_te, shuffle=False, **self.args['loader_te_args'])
        elif self.net_args['arch_name'] == 'gcn' and self.args['mode'] == 'validation':
            set_va = self.handler(root='./data', dataset = 'val', X = X_va, Y = Y_va)
            loader = GCNDataLoader(set_va, shuffle=False, **self.args['loader_va_args'])
        elif self.net_args['arch_name'] == 'lstm' and self.args['mode'] == 'testing':
            loader = DataLoader(self.handler(X_te, Y_te), shuffle=False, **self.args['loader_te_args'])
        elif self.net_args['arch_name'] == 'lstm' and self.args['mode'] == 'validation':
            loader = DataLoader(self.handler(X_va, Y_va), shuffle=False, **self.args['loader_va_args'])
        elif self.args['mode'] == 'testing':
            loader = DataLoader(self.handler(X_te, Y_te, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        elif self.args['mode'] == 'validation':
            loader = DataLoader(self.handler(X_va, Y_va, transform=self.args['transform']), shuffle=False, **self.args['loader_va_args'])

        con_metric = MulticlassConfusionMatrix(num_classes=self.args['n_label']).to(self.device)
        acc_metric = MulticlassAccuracy(num_classes=self.args['n_label']).to(self.device)
        pre_metric = MulticlassPrecision(num_classes=self.args['n_label']).to(self.device)
        rec_metric = MulticlassRecall(num_classes=self.args['n_label']).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=self.args['n_label']).to(self.device)

        self.clf.eval()
        with torch.no_grad():
            if self.net_args['arch_name'] == 'gcn':
                for data in loader:
                    data = data.to(self.device)
                    out, e1 = self.clf(data)
                    pred = out.max(1)[1]
                    con = con_metric(pred, data.y)
                    acc = acc_metric(pred, data.y)
                    pre = pre_metric(pred, data.y)
                    rec = rec_metric(pred, data.y)
                    f1 = f1_metric(pred, data.y)
            else:
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    pred = out.max(1)[1]
                    con = con_metric(pred, y)
                    acc = acc_metric(pred, y)
                    pre = pre_metric(pred, y)
                    rec = rec_metric(pred, y)
                    f1 = f1_metric(pred, y)

        con = con_metric.compute()
        acc = acc_metric.compute()
        pre = pre_metric.compute()
        rec = rec_metric.compute()
        f1 = f1_metric.compute()
        con_metric.reset()
        acc_metric.reset()
        pre_metric.reset()
        rec_metric.reset()
        f1_metric.reset()

        return con.cpu(), acc.cpu(), pre.cpu(), rec.cpu(), f1.cpu()


    def predict_prob_embed(self, X, Y, eval=True):
        if self.net_args['arch_name'] == 'gcn':
            set_te = self.handler(root='./data', dataset='query', X = X, Y = Y)
            loader_te = GCNDataLoader(set_te, shuffle=False, **self.args['loader_te_args'])
        elif self.net_args['arch_name'] == 'lstm':
            loader_te = DataLoader(self.handler(X, Y),
                                shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                                shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label])
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                if self.net_args['arch_name'] == 'gcn':
                    for data in loader_te:
                        data = data.to(self.device)
                        out, e1 = self.clf(data)
                        prob = F.softmax(out, dim=1)
                        probs[data.idxs] = prob.cpu()
                        embeddings[data.idxs] = e1.cpu()
                else:
                    for x, y, idxs in loader_te:
                        x, y = x.to(self.device), y.to(self.device)
                        out, e1 = self.clf(x)
                        prob = F.softmax(out, dim=1)
                        probs[idxs] = prob.cpu()
                        embeddings[idxs] = e1.cpu()
        else:
            self.clf.train()
            if self.net_args['arch_name'] == 'gcn':
                for data in loader_te:
                    data = data.to(self.device)
                    out, e1 = self.clf(data)
                    prob = F.softmax(out, dim=1)
                    probs[data.idxs] = prob.cpu()
                    embeddings[data.idxs] = e1.cpu()
            else:
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu()
                    embeddings[idxs] = e1.cpu()

        return probs, embeddings

    def get_embedding(self, X, Y):
        if self.net_args['arch_name'] == 'gcn':
            set_te = self.handler(root='./data', dataset='query', X = X, Y = Y)
            loader_te = GCNDataLoader(set_te, shuffle=False, **self.args['loader_te_args'])
        elif self.net_args['arch_name'] == 'lstm':
            loader_te = DataLoader(self.handler(X, Y),
                                shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                                shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            if self.net_args['arch_name'] == 'gcn':
                for data in loader_te:
                    data = data.to(self.device)
                    out, e1 = self.clf(data)
                    embedding[data.idxs] = e1.cpu()
            else:
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    embedding[idxs] = e1.cpu()

        return embedding