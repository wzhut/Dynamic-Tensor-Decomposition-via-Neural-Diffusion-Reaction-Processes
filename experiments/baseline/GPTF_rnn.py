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

from kernels import KernelRBF, KernelARD
# from data.real_events import EventData

from torch.utils.data import Dataset, DataLoader

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)



#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class GPTF_rnn:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, m, B, device, jitter=1e-4):
        self.device = device
        self.Uinit = Uinit
        self.m = m
        
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)

        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        
        
#         #======================================================================#
#         self.nvec = nvec
#         self.R = R
#         self.U = [torch.rand([self.nvec[k], self.R]) for k in range(self.nmod)]
#         #
        
        #dim. of pseudo input
        self.d = 0
        for k in range(self.nmod):
            self.d = self.d + Uinit[k].shape[1]
        #init mu, L, Z
        #Zinit = self.init_pseudo_inputs()
        Zinit = np.random.rand(self.m, self.d)
        self.Z = torch.tensor(Zinit, device=self.device, requires_grad=True)
        self.N = y.size
        #variational posterior
        self.mu = torch.tensor(np.zeros([m,1]), device=self.device, requires_grad=True)
        self.L = torch.tensor(np.eye(m), device=self.device, requires_grad=True)
        #kernel parameters
        #self.log_amp = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_ls = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.jitter = torch.tensor(jitter, device=self.device)
        self.kernel = KernelRBF(self.jitter)
        
        R = self.U[0].shape[1]
        self.W = torch.zeros([R,R], device=self.device, requires_grad=True)
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.zeros(R, device=self.device, requires_grad=True)
        self.log_v = torch.tensor(0.0, device=self.device, requires_grad=True)
#         cprint('r', self.W)
#         cprint('r', self.b)
#         cprint('r', self.log_v)


    def trans_prior(self,):
        T = self.U[-1].float()
        
        trans_mu = torch.sigmoid(torch.matmul(T, self.W) + self.b)
        I = torch.eye(T.shape[1]).to(self.device)
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
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        input_emb = torch.cat([self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        #time_log = torch.log(self.time_points[sub_ind] + 1e-4)
#         input_emb = torch.cat([input_emb, time_log], 1)
        input_emb = torch.cat([input_emb], 1)
        y_sub = self.y[sub_ind]
        
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        Kmn = self.kernel.cross(self.Z, input_emb, torch.exp(self.log_ls))
        Knm = Kmn.T
        Ltril = torch.tril(self.L)
        KnmKmmInv = torch.linalg.solve(Kmm, Kmn).T
        KnmKmmInvL = torch.matmul(KnmKmmInv, Ltril)
        tau = torch.exp(self.log_tau)
        ls = torch.exp(self.log_ls)
        
        trans_log_prob = self.trans_prior()

        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = -0.5*torch.logdet(Kmm) - 0.5*torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5*torch.sum(torch.log(torch.square(torch.diag(Ltril)))) \
                + 0.5*self.N*self.log_tau - 0.5*tau*self.N/self.B*torch.sum(torch.square(y_sub - torch.matmul(KnmKmmInv, self.mu))) \
                - 0.5*tau*( self.N*(1.0+self.jitter) - self.N/self.B*torch.sum(KnmKmmInv*Knm) + self.N/self.B*torch.sum(torch.square(KnmKmmInvL)) ) \
                + 0.5*self.m - 0.5*self.N*torch.log(2.0*torch.tensor(np.pi, device=self.device)) +\
        trans_log_prob
               

        return -torch.squeeze(ELBO)

    def init_pseudo_inputs(self):
        part = [None for k in range(self.nmod)]
        for k in range(self.nmod):
            part[k] = self.Uinit[k][self.ind[:,k], :]
        X = np.hstack(part)

        X = X[np.random.randint(X.shape[0], size=self.m * 100), :]
        print(X.shape)

        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(X)
        return kmeans.cluster_centers_


    def pred(self, test_ind, test_time):
        inputs = torch.cat([self.U[k][test_ind[:,k],:]  for k in range(self.nmod)], 1)
#         inputs = torch.cat([inputs, test_time],1)
        inputs = torch.cat([inputs],1)
        #test_time_log = torch.log(test_time + 1e-4)
        #inputs = torch.cat([inputs, test_time_log],1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        pred_std = 1 + self.jitter - (Knm * torch.linalg.solve(Kmm, Knm.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view(pred_mean.shape)
        return pred_mean, pred_std
    
    def test(self, idx, T, y):
        pred_m, pred_std = self.pred(idx, T)
        rmse = torch.sqrt(((pred_m - y)**2).mean()) / torch.sqrt((y**2).mean())
        mae = torch.abs((pred_m - y)).mean() / torch.abs(y).mean()
        # ll = -0.5 * torch.exp(self.log_tau) * (pred_m - y) ** 2 + 0.5 * self.log_tau - 0.5 * np.log(2 * np.pi)
        sigma2 = torch.exp(-self.log_tau) + pred_std ** 2
        ll = -0.5 / sigma2 * (pred_m - y)**2  - 0.5 * torch.log(sigma2) - 0.5 * np.log(2 * np.pi)
        ll = ll.mean()
        return rmse, mae, ll, pred_m, pred_std


    def _callback(self, ind_te, yte, time_te):
        with torch.no_grad():
            ls = torch.exp(self.log_ls)
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te, time_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind, self.time_points)
            rmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            rmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.linalg.norm(self.y)
            nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.linalg.norm(yte)
#             print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
#                  (ls, tau, err_tr, err_te))
#             with open('sparse_gptf_res.txt','a') as f:
#                 f.write('%g '%err_te)
                
            return rmse_tr.item(), rmse_te.item(), nrmse_tr.item(), nrmse_te.item(), tau.item()
            
    
    def train(self, ind_te, yte, time_te, lr, max_epochs, test_every=100):
        
        cprint('b', '@@@@@@@@@@  GPTF RNN is being trained @@@@@@@@@@')
        
        yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        # yte = yte.reshape([-1, 1])
        time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        # time_te = time_te.reshape([-1,1])
        paras = self.U + [self.Z, self.mu, self.L, self.log_ls, self.log_tau, self.W, self.b, self.log_v]
        
        minimizer = Adam(paras, lr=lr)
        
        tr_rmse_list = []
        te_rmse_list = []
        tr_mae_list = []
        te_mae_list = []
        tr_ll_list = []
        te_ll_list = []
        tr_pred_m_list = []
        te_pred_m_list = []
        tr_pred_std_list = []
        te_pred_std_list = []

        with torch.no_grad():
            tr_rmse, tr_mae, tr_ll, tr_pred_m, tr_pred_std = self.test(self.ind, self.time_points, self.y)
            te_rmse, te_mae, te_ll, te_pred_m, te_pred_std = self.test(ind_te, time_te, yte)
            # print('Epoch: {} nELBO: {} trRMSE: {} teRMSE: {}'.format(epoch + 1, loss.item(), tr_rmse.item(), te_rmse.item()))
            tr_rmse_list.append(tr_rmse.item())
            te_rmse_list.append(te_rmse.item())
            tr_mae_list.append(tr_mae.item())
            te_mae_list.append(te_mae.item())
            tr_ll_list.append(tr_ll.item())
            te_ll_list.append(te_ll.item())
            tr_pred_m_list.append(tr_pred_m.view(-1).tolist())
            te_pred_m_list.append(te_pred_m.view(-1).tolist())
            tr_pred_std_list.append(tr_pred_std.view(-1).tolist())
            te_pred_std_list.append(te_pred_std.view(-1).tolist())
        
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
                with torch.no_grad():
                    tr_rmse, tr_mae, tr_ll, tr_pred_m, tr_pred_std = self.test(self.ind, self.time_points, self.y)
                    te_rmse, te_mae, te_ll, te_pred_m, te_pred_std = self.test(ind_te, time_te, yte)
                    print('Epoch: {} nELBO: {} trRMSE: {} teRMSE: {}'.format(epoch + 1, loss.item(), tr_rmse.item(), te_rmse.item()))
                    tr_rmse_list.append(tr_rmse.item())
                    te_rmse_list.append(te_rmse.item())
                    tr_mae_list.append(tr_mae.item())
                    te_mae_list.append(te_mae.item())
                    tr_ll_list.append(tr_ll.item())
                    te_ll_list.append(te_ll.item())
                    tr_pred_m_list.append(tr_pred_m.view(-1).tolist())
                    te_pred_m_list.append(te_pred_m.view(-1).tolist())
                    tr_pred_std_list.append(tr_pred_std.view(-1).tolist())
                    te_pred_std_list.append(te_pred_std.view(-1).tolist())
        return tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list
            
                
def test_beijing5(rank):
    data_file = '../data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    rmse_list = []
    mae_list = []
    ll_list = []
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

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('beijing_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))

def test_ctr5(rank):
    data_file = '../data/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    ll_list = []
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

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('ctr_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))

def test_server5(rank):
    data_file = '../data/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    ll_list = []
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

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('server_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))

def test_server5_extra(rank):
    data_file = '../data/server_10k_extra.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    ll_list = []
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

        
        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('server_extra_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))

def test_weather5(rank):
    data_file = '../data/ca_weather_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    ll_list = []
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('weather_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))

def test_traffic5(rank):
    data_file = '../data/ca_traffic_30k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    mae_list = []
    ll_list = []
    for fold in range(5):
        train_ind = data[fold]['tr_ind']
        train_time = data[fold]['tr_T']
        train_y = data[fold]['tr_y']
        test_ind = data[fold]['te_ind']
        test_time = data[fold]['te_T']
        test_y = data[fold]['te_y']

        nepoch = 5000
        R = rank
        batch_size = 1000
        
        m = 100
        lr = 1e-3
        
        # U = [np.random.rand(ndims[0],R), np.random.rand(ndims[1],R)]
        U = [np.random.rand(n, R) for n in ndims]
        model = GPTF_rnn(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list =  model.train(test_ind, test_y, test_time, lr, nepoch)
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
    with open('GPTF_rnn.txt', 'a') as f:
        f.write('traffic_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))


if __name__ == '__main__':
    # test_traffic5(2)
    # test_traffic5(3)
    # test_traffic5(5)
    # test_traffic5(7)
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

    # test_server5(2)
    # test_server5(3)
    # test_server5(5)
    # test_server5(7)