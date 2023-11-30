import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from smiles2graph import smile_to_graph
from texts2vector import Vocab, tokenize, truncate_pad
from PIL import Image

############################################## get dataset ##############################################

def get_dataset(name, data_dir):
    if name == 'MNIST':
        return get_MNIST(data_dir)
    elif name == 'IMDB':
        return get_IMDB(data_dir)
    elif name == 'BACE':
        return get_BACE(data_dir)


def get_MNIST(data_dir):
    raw_tr = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=True, download=True)
    raw_te = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=False, download=True)
    X_tr = raw_tr.data[:-10000]
    Y_tr = raw_tr.targets[:-10000]
    X_va = raw_tr.data[-10000:]
    Y_va = raw_tr.targets[-10000:]
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te


def get_IMDB(data_dir):
    texts_TrVa, labels_TrVa = [], []
    texts_Tr, labels_Tr = [], []
    texts_Va, labels_Va = [], []
    texts_Te, labels_Te = [], []

    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'IMDB/aclImdb/train', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                texts_TrVa.append(review)
                labels_TrVa.append(1 if label == 'pos' else 0)

        folder_name = os.path.join(data_dir, 'IMDB/aclImdb/test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                texts_Te.append(review)
                labels_Te.append(1 if label == 'pos' else 0)

    for i in range(len(texts_TrVa)):
        if i % 5 == 0:
            texts_Va.append(texts_TrVa[i])
            labels_Va.append(labels_TrVa[i])
        else:
            texts_Tr.append(texts_TrVa[i])
            labels_Tr.append(labels_TrVa[i])

    train_tokens = tokenize(texts_Tr, token='word')
    val_tokens = tokenize(texts_Va, token='word')
    test_tokens = tokenize(texts_Te, token='word')
    vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    X_tr = np.array([truncate_pad(vocab[line], 512, vocab['<pad>']) for line in train_tokens])
    X_va = np.array([truncate_pad(vocab[line], 512, vocab['<pad>']) for line in val_tokens])
    X_te = np.array([truncate_pad(vocab[line], 512, vocab['<pad>']) for line in test_tokens])
    
    Y_tr = torch.tensor(labels_Tr, dtype=torch.long)
    Y_va = torch.tensor(labels_Va, dtype=torch.long)
    Y_te = torch.tensor(labels_Te, dtype=torch.long)

    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te, vocab


def get_BACE(data_dir):
    smiles, labels = [], []
    smiles_0, labels_0 = [], []
    smiles_1, labels_1 = [], []
    smiles_Trva, labels_Trva = [], [] # 1209
    smiles_Tr, labels_Tr = [], [] # 967
    smiles_Va, labels_Va = [], [] # 242
    smiles_Te, labels_Te = [], [] # 304

    with open(os.path.join(data_dir, 'BACE', 'BACE.csv'), 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i > 0:
            line = line.strip()
            _smile = line.split(',')[0]
            _label = line.split(',')[2]
            smiles.append(_smile)
            labels.append(_label)

    for i, _labels in enumerate(labels):
        if _labels == '0':
            smiles_0.append(smiles[i])
            labels_0.append(labels[i])
        else:
            smiles_1.append(smiles[i])
            labels_1.append(labels[i])
    
    # trainval-text split
    for i in range(len(smiles_0)):
        if i % 5 == 0:
            smiles_Te.append(smiles_0[i])
            labels_Te.append(labels_0[i])
        else:
            smiles_Trva.append(smiles_0[i])
            labels_Trva.append(labels_0[i])
    
    for i in range(len(smiles_1)):
        if i % 5 == 0:
            smiles_Te.append(smiles_1[i])
            labels_Te.append(labels_1[i])
        else:
            smiles_Trva.append(smiles_1[i])
            labels_Trva.append(labels_1[i])

    # train-val split
    for i in range(len(smiles_Trva)):
        if i % 5 == 0:
            smiles_Va.append(smiles_Trva[i])
            labels_Va.append(labels_Trva[i])
        else:
            smiles_Tr.append(smiles_Trva[i])
            labels_Tr.append(labels_Trva[i])

    return smiles_Tr, labels_Tr, smiles_Va, labels_Va, smiles_Te, labels_Te

############################################## get datahandler ##############################################

def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'IMDB':
        return DataHandler2
    elif name == 'BACE':
        return DataHandler3

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
        # return self.Y.size(0)

class DataHandler3(InMemoryDataset):
    def __init__(self, root, dataset, X, Y, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset = dataset
        self.process(X, Y)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def process(self, X, Y):
        data_list = []
        data_len = len(X)

        for i in range(data_len):
            smile = X[i]
            label = int(Y[i])
            c_size, features, edge_index = smile_to_graph(smile)
            processedData = DATA.Data(x = torch.FloatTensor(features),
                                      edge_index = torch.LongTensor(edge_index).transpose(1, 0),
                                      y=torch.LongTensor([label]))
            processedData.idxs = torch.LongTensor([i])
            data_list.append(processedData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)