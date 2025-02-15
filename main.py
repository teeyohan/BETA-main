import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from dataset import get_dataset, get_handler
from torchvision import transforms
import torch

from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier
from models.gcn import GCNClassifier
from models.resnet import ResNetClassifier
from models.vit import VisionTransformerClassifier
from models.training import Training

from query_strategies.random_sampling import RandomSampling
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from query_strategies.core_set import CoreSet
from query_strategies.badge_sampling import BadgeSampling
from query_strategies.gcn_sampling import GCNSampling
from query_strategies.alpha_mix_sampling import AlphaMixSampling
from query_strategies.weight_perturbation_sampling import WeightPerturbationSampling

import warnings
warnings.filterwarnings('ignore')

ALL_STRATEGIES = [
    'RandomSampling',
    'EntropySampling',
    'BALDDropout',
    'CoreSet',
    'BadgeSampling',
    'GCNSampling',
    'AlphaMixSampling',
    'WeightPerturbationSampling',
]

def save_args(args, path, name):
    config = vars(args)
    with open(os.path.join(path, name + '.json'), 'w') as f:
        hps = {key: val for key, val in config.items() if not isinstance(val, type)}
        json.dump(hps, f, indent=2)

def supervised_learning(args):
    train_parser = argparse.ArgumentParser(description="Training hyper-parameters.")

    train_parser.add_argument('--n_epoch', type=int, default=500)
    train_parser.add_argument('--optimizer', type=str, default='Adam')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--learning_rate', type=float, default=0.001)
    train_parser.add_argument('--emb_size', type=int, default=256)
    train_parser.add_argument('--dropout', type=float, default=0.1)
    train_parser.add_argument('--continue_training', type=bool, default=True)
    train_parser.add_argument('--n_early_stopping', type=int, default=50)
    train_parser.add_argument('--mode', type=str, default='testing', choices=['validation', 'testing'])

    if 'MNIST' in args.data_name:
        train_parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'vit_small', 'vit_base', 'vit_large'])
    elif 'BACE' in args.data_name:
        train_parser.add_argument('--model', type=str, default='gcn')
    else:
        train_parser.add_argument('--model', type=str, default='lstm')

    train_args, _ = train_parser.parse_known_args()

    train_params_pool = {
        'MNIST':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 50000,
             # For simpleCNN
             'transform': transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))]),
                                             
            # For Resnet and ViT
            #  'transform': transforms.Compose([transforms.Resize(32),
            #                                   transforms.Grayscale(3),
            #                                   transforms.ToTensor(),
            #                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
             'loader_tr_args': {'batch_size': train_args.batch_size},
             'loader_va_args': {'batch_size': train_args.batch_size},
             'loader_te_args': {'batch_size': train_args.batch_size},
             'optimizer_args': {'lr': train_args.learning_rate},
             'log_dir': './logs/MNIST',
             'continue_training': train_args.continue_training,
             'n_early_stopping': train_args.n_early_stopping},
        'IMDB':
            {'n_epoch': train_args.n_epoch,
             'n_label': 2,
             'n_training_set': 20000,
             'loader_tr_args': {'batch_size': train_args.batch_size},
             'loader_va_args': {'batch_size': train_args.batch_size},
             'loader_te_args': {'batch_size': train_args.batch_size},
             'optimizer_args': {'lr': train_args.learning_rate},
             'log_dir': './logs/IMDB',
             'continue_training': train_args.continue_training,
             'n_early_stopping': train_args.n_early_stopping},
        'BACE':
            {'n_epoch': train_args.n_epoch,
             'n_label': 2,
             'n_training_set': 967,
             'loader_tr_args': {'batch_size': train_args.batch_size},
             'loader_va_args': {'batch_size': train_args.batch_size},
             'loader_te_args': {'batch_size': train_args.batch_size},
             'optimizer_args': {'lr': train_args.learning_rate},
             'log_dir': './logs/BACE',
             'continue_training': train_args.continue_training,
             'n_early_stopping': train_args.n_early_stopping},
        }

    train_params = train_params_pool[args.data_name]
    train_params['optimizer'] = train_args.optimizer
    train_params['mode'] = train_args.mode

    if args.strategy == 'All':
        for strategy in ALL_STRATEGIES:
            al_train(args, train_args, train_params, strategy)
    else:
        al_train(args, train_args, train_params, args.strategy)

def al_train(args, train_args, train_params, strategy_name):
    main_path = os.path.join(args.log_dir, args.data_name)
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    general_path = os.path.join(main_path,
                                'init' + str(args.n_init_lb) + 
                                '_query' + str(args.n_query) + 
                                '_rounds' + str(args.n_round))

    if not os.path.exists(general_path):
        os.makedirs(general_path)

    for repeat in args.repeats:
        al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, repeat)

def al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, repeat):
    exp_name = strategy_name + '_repeat' + str(repeat)
    sub_path = os.path.join(general_path, exp_name)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    save_args(args, sub_path, 'args')
    save_args(train_args, sub_path, 'train_args')

    if 'IMDB' in args.data_name:
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, vocab = get_dataset(args.data_name, args.data_dir)
    else:
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = get_dataset(args.data_name, args.data_dir)
    
    train_params['emb_size'] = train_args.emb_size
    args.n_label = train_params['n_label']

    n_pool = len(Y_tr)
    n_val = len(Y_va)
    n_test = len(Y_te)

    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:args.n_init_lb]] = True

    print('number of labeled pool: {}'.format(idxs_lb.sum()))
    print('number of unlabeled pool: {}'.format(n_pool - idxs_lb.sum()))
    print('number of validation pool: {}'.format(n_val))
    print('number of testing pool: {}'.format(n_test))

    if 'BACE' in args.data_name:
        net = GCNClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'emb_size': train_params['emb_size'],
                    'in_features': 78,
                    'dropout': train_args.dropout}
    elif 'IMDB' in args.data_name:
        net = LSTMClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'emb_size': train_params['emb_size'],
                    'in_features': len(vocab),
                    'dropout': train_args.dropout}
    elif 'resnet' in train_args.model:
        net = ResNetClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': True, 'fine_tune_layers': 1,
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout}
    elif 'vit_small' in train_args.model:
        net = VisionTransformerClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': True, 'fine_tune_layers': 1,
                    'pretrained_weights': 'weights/dino_deitsmall16_pretrain.pth',
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout}
    elif 'vit_base' in train_args.model:
        net = VisionTransformerClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': True, 'fine_tune_layers': 1,
                    'pretrained_weights': 'weights/dino_vitbase16_pretrain.pth',
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout}
    elif 'vit_large' in train_args.model:
        net = VisionTransformerClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': True, 'fine_tune_layers': 1,
                    'pretrained_weights': 'weights/vitlarge16_pretrain.pth',
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout}
    else:
        net = CNNClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'emb_size': train_params['emb_size'],
                    'in_channels': 1,
                    'dropout': train_args.dropout}

    use_cuda = torch.cuda.is_available()
    print('Using %s device.' % ("cuda" if use_cuda else "cpu"))
    device = torch.device("cuda" if use_cuda else "cpu")

    handler = get_handler(args.data_name)
    model = Training(net, net_args, handler, train_params, device, init_model=True)

    cls = globals()[strategy_name]
    strategy = cls(X_tr, Y_tr, X_va, Y_va, idxs_lb, model, args, device)

    print(args.data_name)
    print('Repeat {}'.format(repeat))
    print(type(strategy).__name__)

    # round 0
    strategy.train(name='0')

    con, acc, pre, rec, f1 = strategy.predict(X_va, Y_va, X_te, Y_te)

    result_conf = pd.DataFrame(con)
    result_aprf = pd.DataFrame([[acc.item(), pre.item(), rec.item(), f1.item(), 0.]])
    result_conf.to_csv(os.path.join(general_path, strategy_name + '_conf_0_repeat' + str(repeat) + '.csv'), header=False, index=False)
    result_aprf.to_csv(os.path.join(general_path, strategy_name + '_aprf_repeat' + str(repeat) + '.csv'), header=False, index=False)
    print('Round 0\nconfusion matrix\n{}\naccuracy {:.4f}    precision {:.4f}    recall {:.4f}    f1_score {:.4f}'.format(con, acc, pre, rec, f1))

    if args.save_checkpoints:
        torch.save(strategy.model.clf, os.path.join(sub_path, 'model_round_0.pt'))
        # torch.save(strategy.model.clf.state_dict(), os.path.join(sub_path, 'model_round_0.pt'))
    np.save(open(os.path.join(sub_path, 'query_0.np'), 'wb'), idxs_tmp[idxs_lb])

    # AL rounds
    for rd in range(1, args.n_round + 1):
        print('Round {}'.format(rd))
        budget = args.n_query
        print('query budget: %d' % budget)

        start_time = time.time()
        q_idxs = strategy.query(budget)
        duration = time.time() - start_time

        if 'BACE' in args.data_name:
            query_result = torch.zeros(len(Y_tr), dtype=torch.bool)
        else:
            query_result = torch.zeros(Y_tr.size(), dtype=torch.bool)
        query_result[q_idxs] = True

        # update
        idxs_lb[q_idxs] = True
        strategy.update(idxs_lb)

        print('training with %d labeled samples.' % idxs_lb.sum())
        strategy.train(str(rd))
        con, acc, pre, rec, f1 = strategy.predict(X_va, Y_va, X_te, Y_te)

        result_conf = pd.DataFrame(con)
        result_aprf.loc[len(result_aprf)] = [acc.item(), pre.item(), rec.item(), f1.item(), duration]
        result_conf.to_csv(os.path.join(general_path, strategy_name + '_conf_{}'.format(rd) + '_repeat' + str(repeat) + '.csv'), header=False, index=False)
        result_aprf.to_csv(os.path.join(general_path, strategy_name + '_aprf' + '_repeat' + str(repeat) + '.csv'), header=False, index=False)
        print('Round {}\nconfusion matrix\n{}\naccuracy {:.4f}    precision {:.4f}    recall {:.4f}    f1_score {:.4f}'.format(rd, con, acc, pre, rec, f1))

        if args.save_checkpoints:
            torch.save(strategy.model.clf, os.path.join(sub_path, 'model_round_%d.pt' % (rd)))
            # torch.save(strategy.model.clf.state_dict(), os.path.join(sub_path, 'model_round_%d.pt' % (rd)))
        np.save(open(os.path.join(sub_path, 'query_' + str(rd) + '.np'), 'wb'), q_idxs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight Perturbation Active Learning hyper-parameters")
    parser.add_argument('--data_name', type=str, default='MNIST', choices=['MNIST', 'IMDB', 'BACE'], help='The dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='The directory of data')
    parser.add_argument('--log_dir', type=str, default='./logs', help='The directory of log')
    parser.add_argument('--save_checkpoints', type=bool, default=True, help='If save checkpoints')
    parser.add_argument('--repeats', type=int, default=[1,2,3,4,5])

    parser.add_argument('--n_init_lb', type=int, default=100, help='Initial labeled samples')
    parser.add_argument('--n_query', type=int, default=100, help='Query labeled samples')
    parser.add_argument('--n_round', type=int, default=5, help='AL round')
    parser.add_argument('--strategy', type=str, default='WeightPerturbationSampling',
                                                choices=['RandomSampling', 'EntropySampling',
                                                         'BALDDropout', 'CoreSet',
                                                         'BadgeSampling', 'GCNSampling',
                                                         'AlphaMixSampling', 'WeightPerturbationSampling', 'All'])

    parser.add_argument('--eps_cap', type=float, default=0.001, choices=[0.001, 0.01, 0.1])
    parser.add_argument('--eps_get', type=str, default='accumulated', choices=['accumulated', 'max_optimized', 'min_optimized'])
    parser.add_argument('--eps_select', type=str, default='entropy', choices=['random', 'entropy', 'kmeans'])
    parser.add_argument('--eps_learning_rate', type=float, default=0.1, help='The learning rate of finding the optimised epsilon')
    parser.add_argument('--eps_learning_iters', type=int, default=5, help='The number of iterations for learning epsilon')
    parser.add_argument('--eps_learning_batch_size', type=int, default=1000000, help='The batchsize for learning epsilon')

    args, _ = parser.parse_known_args()

    supervised_learning(args)