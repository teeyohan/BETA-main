import torch
import torch.nn as nn
from models.utils import build_mlp


class CNNClassifier(nn.Module):
    def __init__(self, arch_name='cnn', n_label=10, pretrained=False, dropout=0,
                fine_tune_layers=1, emb_size=256, in_channels=1, **kwargs):
        super(CNNClassifier, self).__init__(**kwargs)
        self.n_label = n_label
        self.embedding_size = emb_size
        self.conv1 = nn.Conv2d(in_channels, 4*in_channels, 3)
        self.conv2 = nn.Conv2d(4*in_channels, 8*in_channels, 3)
        self.conv3 = nn.Conv2d(8*in_channels, 16*in_channels, 3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hidden_layers = build_mlp(16*4*4, (), emb_size, 
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
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = torch.flatten(x, 1)
            embd = self.hidden_layers(x)
        out = self.classifier(embd)
        return out, embd

    def get_embedding_dim(self):
        return self.embedding_size
    
    def get_classifier(self):
        return self.classifier
    
