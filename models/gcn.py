import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from models.utils import build_mlp


class GCNClassifier(torch.nn.Module):
    def __init__(self, arch_name='gcn', n_label=2, pretrained=False, dropout=0, 
                fine_tune_layers=1, in_features=None, emb_size = 256, **kwargs):
        super(GCNClassifier, self).__init__(**kwargs)
        self.n_label = n_label
        self.embedding_size = emb_size
        self.gconv1 = GCNConv(in_features, 64)
        self.gconv2 = GCNConv(64, 128)
        self.gconv3 = GCNConv(128, 256)
        self.relu = nn.ReLU()
        self.hidden_layers = build_mlp(256, (), emb_size, 
                                dropout=dropout, 
                                use_batchnorm=False, 
                                add_dropout_after=False) # input_size -> emb_size
        self.classifier = build_mlp(emb_size, (), n_label,
                            dropout=dropout,
                            use_batchnorm=False,
                            add_dropout_after=False) # emb_size -> n_label

    def forward(self, data, embedding=False):
        if embedding:
            embd = data
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.gconv1(x, edge_index)
            x = self.relu(x)
            x = self.gconv2(x, edge_index)
            x = self.relu(x)
            x = self.gconv3(x, edge_index)
            x = self.relu(x)
            x = gmp(x, batch)
            embd = self.hidden_layers(x)
        out = self.classifier(embd)
        return out, embd
    
    def get_embedding_dim(self):
        return self.embedding_size
    
    def get_classifier(self):
        return self.classifier