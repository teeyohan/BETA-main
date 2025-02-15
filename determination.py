import os
import random
from tqdm import tqdm
import argparse
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from dataset import get_dataset, get_handler
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GCNDataLoader


def Determination(args):

    tag = args.query_cycle
    if tag == 'One':
        query_name = 'query_1.np'
        weight_name = 'model_round_1.pt'
        root_path = 'logs/{}/_init100_query100_rounds5/WeightPerturbationSampling_repeat{}'.format(args.data_name, args.exp_repeat)
    elif tag == 'Two':
        query_name = 'query_2.np'
        weight_name = 'model_round_2.pt'
        root_path = 'logs/{}/_init100_query100_rounds5/WeightPerturbationSampling_repeat{}'.format(args.data_name, args.exp_repeat)
    elif tag == 'Three':
        query_name = 'query_3.np'
        weight_name = 'model_round_3.pt'
        root_path = 'logs/{}/_init100_query100_rounds5/WeightPerturbationSampling_repeat{}'.format(args.data_name, args.exp_repeat)
    elif tag == 'Four':
        query_name = 'query_4.np'
        weight_name = 'model_round_4.pt'
        root_path = 'logs/{}/_init100_query100_rounds5/WeightPerturbationSampling_repeat{}'.format(args.data_name, args.exp_repeat)
    elif tag == 'Five':
        query_name = 'query_5.np'
        weight_name = 'model_round_5.pt'
        root_path = 'logs/{}/_init100_query100_rounds5/WeightPerturbationSampling_repeat{}'.format(args.data_name, args.exp_repeat)

    query_path = os.path.join(root_path, query_name)
    weight_path = os.path.join(root_path, weight_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 'IMDB' in args.data_name:
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, vocab = get_dataset(args.data_name, args.data_dir)
    else:
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = get_dataset(args.data_name, args.data_dir)
        
    idx_all = np.arange(len(Y_tr))
    idxs_query = np.load(query_path)
    idxs_unquery = np.array([i for i in idx_all if i not in idxs_query])
    sub_idxs_unquery = np.random.choice(idxs_unquery.shape[0], 500, replace=False)

    if args.data_name == 'BACE':
        set = get_handler(args.data_name)(
            root='./data', dataset='train', X = [X_tr[i] for i in sub_idxs_unquery], Y = [Y_tr[i] for i in sub_idxs_unquery])
        loader = GCNDataLoader(set, shuffle=False, **{'batch_size': 99999, 'num_workers': 0})
    elif args.data_name == 'IMDB':
        loader = DataLoader(get_handler(args.data_name)(
                            X_tr[sub_idxs_unquery], Y_tr[sub_idxs_unquery]),
                            shuffle=False, **{'batch_size': 99999, 'num_workers': 0})
    else:
        loader = DataLoader(get_handler(args.data_name)(
                    X_tr[sub_idxs_unquery], Y_tr[sub_idxs_unquery], 
                    transform=transforms.Compose([
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))])), 
                    shuffle=False, **{'batch_size': 99999, 'num_workers': 0})
    
    model = torch.load(weight_path).to(device)
    model.train()
    for data in loader:
        if args.data_name == 'BACE':
            out, _ = model(data.to(device))
        else:
            x, y, _ = data
            out, _ = model(x.to(device))

        # out = out[idxs_unquery]
        pseudo_label = torch.max(out.data, 1)[1]

        # Calculate Delta
        Delta = torch.max(out, 1)[0] - torch.sort(out, descending=True)[0][:, 1]

        # Calculate Inconsistents (gradient by average sample)
        Flips = []
        for sigma in args.sigmas:
            probs = 0.

            model.zero_grad()
            loss = nn.functional.cross_entropy(out, pseudo_label, reduction='mean')
            loss.backward(retain_graph=True)

            Norm_Gradient = model.classifier[-1].weight.grad.norm()
            print('Computing Inconsistents ...')
            for i in tqdm(range(out.size(0))):
                up = Delta[i].item()
                dowm = Norm_Gradient.item() * sigma + 1e-6
                probs += norm.cdf(up / dowm)
            Flips.append(out.size(0)-probs)
            print('When sigma = {}, Expected Flips: {:.2f}/{}'.format(sigma, out.size(0)-probs, out.size(0)))
            
        # Calculate Inconsistents (gradient by per sample)
        # Flips = []
        # for sigma in args.sigmas:
        #     probs = 0.
        #     Norm_Gradient = torch.ones(out.size(0))
        #     print('Computing Inconsistents ...')
        #     for i in tqdm(range(out.size(0))):
        #         model.zero_grad()
        #         loss = nn.functional.cross_entropy(out[i].view(1, -1), pseudo_label[i].view(1), reduction='sum')
        #         loss.backward(retain_graph=True)
        #         Norm_Gradient[i] = model.classifier[-1].weight.grad.norm()

        #         up = Delta[i].item()
        #         dowm = Norm_Gradient[i].item() * sigma + 1e-6
        #         probs += norm.cdf(up / dowm)
        #     Flips.append(out.size(0)-probs)
        #     print('When sigma = {}, Expected Flips: {:.2f}/{}'.format(sigma, out.size(0)-probs, out.size(0)))
        
        plt.title('Determination analysis on {}'.format(args.data_name), fontsize=16)

        plt.plot(args.sigmas, Flips, marker='o')

        plt.grid()
        plt.xlabel('Perturbation Magnitude',fontsize=14)
        plt.ylabel('Inconsistent',fontsize=14)
        # plt.ylim([0, 250])
        if args.data_name == 'MNIST':
            plt.xticks([0, 32.0, 64.0, 128.0, 256.0, 512.0])
        elif args.data_name == 'IMDB':
            plt.xticks([0, 1024, 2048, 4096, 8192, 16384])
        else:
            plt.xticks([0, 64.0, 128.0, 256.0, 512.0, 1024.0])
        plt.yticks(np.linspace(0, 250, 6))


        # 创建带黑色边框的子图
        fig = plt.gcf()
        sub_ax = fig.add_axes([0.36, 0.18, 0.58, 0.2])

        # 设置子图边框样式
        for spine in sub_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        # 隐藏刻度线
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])

        # 添加文本内容
        sigma_value = 0.0
        if args.data_name == 'MNIST':
            sigma_value = 0.005
        elif args.data_name == 'IMDB':
            sigma_value = 0.25
        elif args.data_name == 'BACE':
            sigma_value = 0.1

        sub_ax.text(0.04, 0.5,
                f'Inconsistent=0, Magnitude=0\nInconsistent→0+, Magnitude≈{sigma_value}\nInconsistent→N/2, Magnitude→∞',
                ha='left', va='center',
                fontsize=14,
                color='#1f77b4',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round'))

        plt.tight_layout()
        # plt.savefig('figs_and_tabs/figs/{}_{}.png'.format(data,tag), dpi=300)
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='BACE', choices=['MNIST', 'IMDB', 'BACE'])
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--query_cycle', type=str, default='One', choices=['One', 'Two', 'Three', 'Four', 'Five'])
    parser.add_argument('--exp_repeat', type=str, default='1', choices=['1', '2', '3', '4', '5'])

    args, _ = parser.parse_known_args()

    if args.data_name == 'MNIST':
        args.sigmas=[0, 8, 16, 32, 64, 128, 256, 512]
    elif args.data_name == 'IMDB':
        args.sigmas=[0, 256, 512, 1024, 2048, 4096, 8192, 16384]
    else:
        args.sigmas=[0, 16, 32, 64, 128, 256, 512, 1024]

    # args.sigmas = np.arange(0.1,1.0,0.1)

    Determination(args)
