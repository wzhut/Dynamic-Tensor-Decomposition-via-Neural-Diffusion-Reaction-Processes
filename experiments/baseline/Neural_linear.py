import numpy as np
from numpy.core.fromnumeric import ndim
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
# from torchdiffeq import odeint
# import torch.autograd.functional as F 
import random
# import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import os
import pickle
import time
# import fire

from infrastructure.misc import *
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader

from RFF import *

# from data.real_events import EventData

# from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class Neural_linear(nn.Module):

    def __init__(self, nmod, nvec, R, nFF, batch_size):
        
        super().__init__()
        
        self.nmod = nmod
        self.nvec = nvec
        assert self.nmod == len(self.nvec)
        
        self.R = R
        self.nFF = nFF
        
        self.B = batch_size
        
        self.register_buffer('dummy', torch.tensor([]))
        
        self.Ulist = nn.ParameterList([nn.Parameter(torch.randn(self.nvec[i], R)) for i in range(self.nmod)])
        self.init_model = RFF(num_ff=self.nFF, input_dim=self.R*self.nmod)
        
        # self.W = torch.zeros([self.R,self.R], requires_grad=True)
        # torch.nn.init.xavier_normal_(self.W)
#         self.b = torch.zeros(self.R, requires_grad=True)
#         self.log_v = torch.tensor(0.0, requires_grad=True)
        #cprint('r', self.W.shape)
        #cprint('r', self.b.shape)
        #cprint('r', self.log_v.shape)
        
    def todev(self, device):
        self.to(device)
        for i in range(len(self.Ulist)):
            self.Ulist[i] = self.Ulist[i].to(device)
        self.init_model.to(device)
        self.W = torch.zeros([self.R,self.R], requires_grad=True, device=device)
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.zeros(self.R, requires_grad=True, device=device)
        self.log_v = torch.tensor(0.0, requires_grad=True, device=device)
        
    def trans_prior(self,):
        T = self.Ulist[-1].float()
        
        trans_mu = torch.matmul(T, self.W) + self.b
        I = torch.eye(T.shape[1]).to(self.dummy.device)
        trans_std = torch.exp(self.log_v)*I
        
        T = T[1:, :]
        trans_mu = trans_mu[:-1, :]
        
        #cprint('r', T.shape)
        #cprint('r', trans_mu.shape)
        
        prior_dist = torch.distributions.MultivariateNormal(loc=trans_mu, covariance_matrix=trans_std)
        
        log_prior = prior_dist.log_prob(T)
#         print(log_prior.sum())
        
#         buff = []
#         for t in range(T.shape[0]):
#             ut = T[t,:]
#             dist = torch.distributions.MultivariateNormal(loc=trans_mu[t,:], covariance_matrix=trans_std)
#             log_prob = dist.log_prob(ut)
#             buff.append(log_prob)
#         #
#         print(sum(buff))

        return log_prior.sum()
        
    def _extract_Uvec(self, b_i_n):    
        Uvec = []
        for i_n in b_i_n:
            v_i_n = []
            for i in range(self.nmod):
                v_i_n.append(self.Ulist[i][i_n[i]])
            #
            v_i_n = torch.cat(v_i_n)
            Uvec.append(v_i_n.unsqueeze(0))
        #
        Uvec = torch.cat(Uvec, dim=0) 
        return Uvec
    
    def forward_init(self, b_i_n):
        Uvec = self._extract_Uvec(b_i_n)
        y = self.init_model(Uvec)
        return y
    
    def test(self, dataset):
        dataloader_test = DataLoader(dataset, batch_size=self.B, shuffle=False, drop_last=True)
        
        soln_all = []
        ground_all = []
        
        for b_i_n, b_t_n, b_obs in dataloader_test:
            
            b_i_n, b_t_n, b_obs = \
                b_i_n.to(self.dummy.device), \
                b_t_n.to(self.dummy.device), \
                b_obs.to(self.dummy.device)
            
            ground_all.append(b_obs)
            
            pred = self.forward_init(b_i_n)
            soln_all.append(pred.squeeze())
        #
        
        soln_all = torch.cat(soln_all).view((-1, 1))
        obs = torch.cat(ground_all).to(self.dummy.device)
        # rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
        rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.sqrt((obs**2).mean())
        mae = torch.abs(obs - soln_all).mean() / torch.abs(obs).mean()
        return rmse.item(), mae.item()
    
    def eval_loss(self, b_i_n, b_obs, Nall):
        pred = self.forward_init(b_i_n).view(b_obs.shape)
        
        trans_log_prob = self.trans_prior()
        
        loss = -0.5*(Nall/self.B)*torch.sum(torch.square(pred-b_obs))+trans_log_prob
        return -loss.squeeze()
          
    def train(self, dataset_train, dataset_test, max_epochs, learning_rate, test_every=100):
        
        cprint('b', '@@@@@@@@@@  Neural Linear is being trained @@@@@@@@@@')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
        
        params = []
        for param in self.parameters():
            if param.requires_grad:
                 params.append(param)
            #
        #
        
        params = params + [self.W, self.b, self.log_v]
        
        optimizer = optim.Adam(params, lr=learning_rate)
        
        
        tr_rmse_list =[]
        te_rmse_list = []
        tr_mae_list = []
        te_mae_list = []
        rmse_tr, mae_tr = self.test(dataset_train)
        rmse_te, mae_te = self.test(dataset_test)
        tr_rmse_list.append(rmse_tr)
        te_rmse_list.append(rmse_te)
        tr_mae_list.append(mae_tr)
        te_mae_list.append(mae_te)
        
        # perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
        # perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
        
        steps = 0
        
        for epoch in tqdm(range(max_epochs)):
            
            for b, (b_i_n, b_t_n, b_obs) in enumerate(dataloader_train):
                
                b_i_n, b_t_n, b_obs = \
                    b_i_n.to(self.dummy.device), \
                    b_t_n.to(self.dummy.device), \
                    b_obs.to(self.dummy.device)
                
#                 pred = self.forward_init(b_i_n)
#                 loss = torch.mean(torch.square(pred.squeeze()-b_obs))

                loss = self.eval_loss(b_i_n, b_obs, len(dataset_train))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % test_every == 0:

                rmse_tr, mae_tr = self.test(dataset_train)
                rmse_te, mae_te = self.test(dataset_test)
                print('Epoch: {} tr_rmse: {} te:_rmse{}'.format(epoch + 1, rmse_tr, rmse_te))

                tr_rmse_list.append(rmse_tr)
                te_rmse_list.append(rmse_te)
                tr_mae_list.append(mae_tr)
                te_mae_list.append(mae_te)
        return tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list
            

        #
class Wrapped_Dataset(Dataset):
    def __init__(self, ind, T, y):
        super().__init__()
        
        self.ind = ind
        self.T = T
        self.y = y

    def __getitem__(self, index):
        
        indices = self.ind[index].astype(int)
        t = self.T[index].astype(float)
        obs = self.y[index].astype(float)
  
        return indices, t, obs
    
    def __len__(self,):
        return self.ind.shape[0]
    


def test_server5(rank=3, save=False):
    data_file = '../data/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:4'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('server_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5_extra(rank=3, save=False):
    data_file = '../data/server_10k_extra.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('server_extra_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))


def test_ctr5(rank=3):
    data_file = '../data/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    min_time = np.min(time)
    max_time = np.max(time)
    Kbins_time = 50
    ndims.append(Kbins_time)
    bins_time = np.linspace(start=min_time, stop=max_time, num=Kbins_time+1)[1:-1]
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        bin_tr_time = np.digitize(train_time, bins=bins_time).reshape((-1, 1))
        bin_te_time = np.digitize(test_time, bins=bins_time).reshape((-1, 1))

        train_ind = np.hstack([train_ind, bin_tr_time])
        test_ind = np.hstack([test_ind, bin_te_time])

        # train_y = np.log(1 + train_y)
        # test_y = np.log(1 + test_y) 

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:4'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('ctr_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_beijing5(rank=3):
    data_file = '../data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    min_time = np.min(time)
    max_time = np.max(time)
    Kbins_time = 50
    ndims.append(Kbins_time)
    bins_time = np.linspace(start=min_time, stop=max_time, num=Kbins_time+1)[1:-1]
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        bin_tr_time = np.digitize(train_time, bins=bins_time).reshape((-1, 1))
        bin_te_time = np.digitize(test_time, bins=bins_time).reshape((-1, 1))

        train_ind = np.hstack([train_ind, bin_tr_time])
        test_ind = np.hstack([test_ind, bin_te_time])

        # train_y = np.log(train_y)
        # test_y = np.log(test_y)

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:4'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('beijing_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_weather5(rank=3):
    data_file = '../data/ca_weather_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    min_time = np.min(time)
    max_time = np.max(time)
    Kbins_time = 50
    ndims.append(Kbins_time)
    bins_time = np.linspace(start=min_time, stop=max_time, num=Kbins_time+1)[1:-1]
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        bin_tr_time = np.digitize(train_time, bins=bins_time).reshape((-1, 1))
        bin_te_time = np.digitize(test_time, bins=bins_time).reshape((-1, 1))

        train_ind = np.hstack([train_ind, bin_tr_time])
        test_ind = np.hstack([test_ind, bin_te_time])

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('weather_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_traffic5(rank=3):
    data_file = '../data/ca_traffic_30k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    min_time = np.min(time)
    max_time = np.max(time)
    Kbins_time = 50
    ndims.append(Kbins_time)
    bins_time = np.linspace(start=min_time, stop=max_time, num=Kbins_time+1)[1:-1]
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        bin_tr_time = np.digitize(train_time, bins=bins_time).reshape((-1, 1))
        bin_te_time = np.digitize(test_time, bins=bins_time).reshape((-1, 1))

        train_ind = np.hstack([train_ind, bin_tr_time])
        test_ind = np.hstack([test_ind, bin_te_time])

        
        nepoch = 1000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        train_dataset = Wrapped_Dataset(train_ind, train_time, train_y)
        test_dataset = Wrapped_Dataset(test_ind, test_time, test_y)

        model = Neural_linear(len(ndims), ndims, R, 51, batch_size)
        model.todev(torch.device('cuda:4'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list = model.train(train_dataset, test_dataset, nepoch, lr, test_every=10)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('Neural_linear.txt', 'a') as f:
        f.write('traffic_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))


if __name__ == '__main__':
    # test_beijing5(2)
    # test_beijing5(3)
    # test_beijing5(5)
    # test_beijing5(7)

    # test_ctr5(2)
    # test_ctr5(3)
    # test_ctr5(5)
    # test_ctr5(7)

    test_server5_extra(2)
    test_server5_extra(3)
    test_server5_extra(5)
    test_server5_extra(7)

    # test_weather5(2)
    # test_weather5(3)
    # test_weather5(5)
    # test_weather5(7)
    # test_traffic5(2)
    # test_traffic5(3)
    # test_traffic5(5)
    # test_traffic5(7)

    # test_server5(2)
    # test_server5(3)
    # test_server5(5)
    # test_server5(7)