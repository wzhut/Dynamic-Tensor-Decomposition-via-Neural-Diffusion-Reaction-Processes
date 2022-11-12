import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
from sklearn.cluster import KMeans
import random
import pickle
# import fire
from tqdm.auto import tqdm, trange

# from baselines.kernels import KernelRBF, KernelARD
# from data.real_events import EventData

from torch.utils.data import Dataset, DataLoader

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)



#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class CPTF_time:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, B, device, jitter=1e-3, t_max=10.0):
        self.device = device
        self.Uinit = Uinit
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        self.N = y.size
        #variational posterior
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        spline_knots = 100
        spline_order = 3
        
        self.spline_params_list = []
        self.t_span = torch.linspace(0,t_max,spline_knots+1).to(self.device)
        #cprint('r', self.t_span)

        for i in range(spline_order+1):
            spline_param = torch.zeros([spline_knots,1], device=self.device, requires_grad=True)
            torch.nn.init.xavier_normal_(spline_param)
            self.spline_params_list.append(spline_param)
            #cprint('r', spline_param.shape)
        #
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        U_sub = [self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)]
        y_sub = self.y[sub_ind]

        U_prod = U_sub[0]
        Ureg = -0.5*torch.sum(torch.square(self.U[0]))
        for k in range(1, self.nmod):
            U_prod = U_prod * U_sub[k]
            Ureg = Ureg - 0.5*torch.sum(torch.square(self.U[k]))
        pred = torch.sum(U_prod, 1, keepdim=True)

        L = Ureg + 0.5*self.N*self.log_tau -0.5*torch.exp(self.log_tau)*self.N/self.B*torch.sum(torch.square(y_sub - pred))
 
        return -torch.squeeze(L)

    def pred(self, test_ind, test_time):
        inputs = [self.U[k][test_ind[:,k],:]  for k in range(self.nmod)]
        U_prod = inputs[0]
        for k in range(1, self.nmod):
            U_prod = U_prod * inputs[k]
        CP = torch.sum(U_prod, 1, keepdim=True)
        
        #cprint('b', CP.shape)
        
        spline_coeff = tuple([self.t_span] + self.spline_params_list)
        #print(spline_coeff)
        
        spline_model = NaturalCubicSpline(spline_coeff)
        
        #print(test_time.shape)
        
        lamda_t = spline_model.evaluate(test_time.squeeze())
        
        pred = CP * lamda_t
        
        #print(pred.shape)
        
        return pred


    def _callback(self, ind_te, yte, time_te):
        with torch.no_grad():
            yte = torch.tensor(yte.reshape((-1, 1)), device=self.device)
            time_te = torch.tensor(time_te.reshape((-1, 1)), device=self.device)
            pred_mean = self.pred(ind_te, time_te)
            pred_tr = self.pred(self.ind, self.time_points)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.sqrt((self.y**2).mean())
            nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.sqrt((yte**2).mean())
            mae_tr = torch.abs(pred_tr - self.y).mean() / torch.abs(self.y).mean()
            mae_te = torch.abs(pred_mean - yte).mean() / torch.abs(yte).mean()
#             print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
#                  (ls, tau, err_tr, err_te))
#             with open('sparse_gptf_res.txt','a') as f:
#                 f.write('%g '%err_te)
                
            return  nrmse_tr.item(), nrmse_te.item(), mae_tr.item(), mae_te.item()
            
            
    
    def train(self, ind_te, yte, time_te, lr, max_epochs, test_every=100):
        
        cprint('r', '@@@@@@@@@@  CPTF Time is being trained @@@@@@@@@@')
        

        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        paras = self.U + [self.log_tau] + self.spline_params_list
        #print(paras)
        tr_rmse_list = []
        te_rmse_list = []
        tr_mae_list = []
        te_mae_list = []
        
        rmse_tr, rmse_te, mae_tr, mae_te = self._callback(ind_te, yte, time_te)
        tr_rmse_list.append(rmse_tr)
        te_rmse_list.append(rmse_te)
        tr_mae_list.append(mae_tr)
        te_mae_list.append(mae_te)
        
        minimizer = Adam(paras, lr=lr)
        
        steps = 0
        
        for epoch in trange(max_epochs):
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                loss.backward(retain_graph=True)
                minimizer.step()
                curr = curr + self.B
            #print('epoch %d done'%epoch)
#             if epoch%5 == 0:
#                 self._callback(ind_te, yte, time_te)

            if (epoch + 1) % test_every == 0:
                rmse_tr, rmse_te, mae_tr, mae_te = self._callback(ind_te, yte, time_te)
                print('Epoch: {} tr_rmse: {} te:_rmse{}'.format(epoch + 1, rmse_tr, rmse_te))
                tr_rmse_list.append(rmse_tr)
                te_rmse_list.append(rmse_te)
                tr_mae_list.append(mae_tr)
                te_mae_list.append(mae_te)

        return tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list
            
            
                

def test_beijing5(rank):
    data_file = '../data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    max_time = np.max(time)
    min_time = np.min(time)
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        train_time = (train_time - min_time) / (max_time - min_time)
        test_time = (test_time - min_time) / (max_time - min_time)
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
        f.write('beijing_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_ctr5(rank):
    data_file = '../data/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        # train_y = np.log(1 + train_y)
        # test_y = np.log(1 + test_y) 

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 50
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
        f.write('ctr_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5(rank):
    data_file = '../data/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    max_time = np.max(time)
    min_time = np.min(time)
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        train_time = (train_time - min_time) / (max_time - min_time)
        test_time = (test_time - min_time) / (max_time - min_time)

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 50
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
        f.write('server_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5_extra(rank):
    data_file = '../data/server_10k_extra.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    max_time = np.max(time)
    min_time = np.min(time)
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        train_time = (train_time - min_time) / (max_time - min_time)
        test_time = (test_time - min_time) / (max_time - min_time)

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 50
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
        f.write('server_extra_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_weather5(rank):
    data_file = '../data/ca_weather_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    max_time = np.max(time)
    min_time = np.min(time)
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        train_time = (train_time - min_time) / (max_time - min_time)
        test_time = (test_time - min_time) / (max_time - min_time)

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
        f.write('weather_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_traffic5(rank):
    data_file = '../data/ca_traffic_30k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    time = np.concatenate([data[0]['tr_T'], data[0]['te_T']], axis=0)
    max_time = np.max(time)
    min_time = np.min(time)
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        train_time = (train_time - min_time) / (max_time - min_time)
        test_time = (test_time - min_time) / (max_time - min_time)

        nepoch = 5000
        R = rank
        batch_size = 1000
        
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, te_rmse_list, te_mae_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
    with open('CPTF_time.txt', 'a') as f:
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