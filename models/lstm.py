import torch
import torch.nn as nn
from models.utils import build_mlp


class LSTMClassifier(nn.Module):
    def __init__(self, arch_name='lstm', n_label=2, pretrained=False, dropout=0,
                fine_tune_layers=1, in_features=None, emb_size=256, **kwargs):
        super(LSTMClassifier, self).__init__(**kwargs)
        self.n_label = n_label
        self.embedding_size = emb_size
        self.embedding = nn.Embedding(in_features, 64)
        self.encoder = nn.LSTM(64, 64, num_layers=1, bidirectional=True)
        self.relu = nn.ReLU()
        self.hidden_layers = build_mlp(4*64, (), emb_size, 
                                dropout=dropout, 
                                use_batchnorm=False, 
                                add_dropout_after=False)
        self.classifier = build_mlp(emb_size, (), n_label,
                            dropout=dropout,
                            use_batchnorm=False,
                            add_dropout_after=False)
        
    def forward(self, x, embedding=False):
        if embedding:
            embd = x
        else:
            x = self.embedding(x.T)
            x = self.relu(x)
            # self.encoder.flatten_parameters()
            x, _ = self.encoder(x)
            x = self.relu(x)
            x = torch.cat((x[0], x[-1]), dim=1)
            embd = self.hidden_layers(x)
        out = self.classifier(embd)
        return out, embd

    def get_embedding_dim(self):
        return self.embedding_size
    
    def get_classifier(self):
        return self.classifier
    
