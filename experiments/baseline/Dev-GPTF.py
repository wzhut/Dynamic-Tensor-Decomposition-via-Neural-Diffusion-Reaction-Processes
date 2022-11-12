import numpy as np
import torch
from torch.optim import Adam
from torch.optim import LBFGS
from sklearn.cluster import KMeans
import random
from tqdm import tqdm

# from baselines.GPTF.kernels import KernelRBF, KernelARD
# from data.real_events import EventData
from kernels import KernelRBF

from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat



# from infrastructure.misc import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)



#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class GPTF:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, m, B, device, jitter=1e-4, test_every=100):
        self.device = device
        self.Uinit = Uinit
        self.m = m
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        #dim. of pseudo input
        self.d = 1
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
        self.test_every = test_every
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        input_emb = torch.cat([self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        #time_log = torch.log(self.time_points[sub_ind] + 1e-4)
        #input_emb = torch.cat([input_emb, time_log], 1)
        input_emb = torch.cat([input_emb, self.time_points[sub_ind]], 1)
        y_sub = self.y[sub_ind]
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        Kmn = self.kernel.cross(self.Z, input_emb, torch.exp(self.log_ls))
        Knm = Kmn.T
        Ltril = torch.tril(self.L)
        KnmKmmInv = torch.linalg.solve(Kmm, Kmn).T
        KnmKmmInvL = torch.matmul(KnmKmmInv, Ltril)
        tau = torch.exp(self.log_tau)
        ls = torch.exp(self.log_ls)
        
        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = -0.5*torch.logdet(Kmm) - 0.5*torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5*torch.sum(torch.log(torch.square(torch.diag(Ltril))))                 + 0.5*self.N*self.log_tau - 0.5*tau*self.N/self.B*torch.sum(torch.square(y_sub - torch.matmul(KnmKmmInv, self.mu)))                 - 0.5*tau*( self.N*(1.0+self.jitter) - self.N/self.B*torch.sum(KnmKmmInv*Knm) + self.N/self.B*torch.sum(torch.square(KnmKmmInvL)) )                 + 0.5*self.m - 0.5*self.N*torch.log(2.0*torch.tensor(np.pi, device=self.device))

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
        inputs = torch.cat([inputs, test_time],1)
        #test_time_log = torch.log(test_time + 1e-4)
        #inputs = torch.cat([inputs, test_time_log],1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        # Knn = self.kernel.matrix(inputs, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        pred_std = 1 + self.jitter - (Knm * torch.linalg.solve(Kmm, Knm.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view(pred_mean.shape)
        return pred_mean, pred_std

    def pred_np(self, test_ind, test_time):
        test_time = torch.tensor(test_time)
        inputs = torch.cat([self.U[k][test_ind[:,k],:]  for k in range(self.nmod)], 1)
        inputs = torch.cat([inputs, test_time],1)
        #test_time_log = torch.log(test_time + 1e-4)
        #inputs = torch.cat([inputs, test_time_log],1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        # Knn = self.kernel.matrix(inputs, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        pred_std = 1 + self.jitter - (Knm * torch.linalg.solve(Kmm, Knm.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view(pred_mean.shape)
        return pred_mean.tolist(), pred_std.tolist()

    def test(self, idx, T, y):
        pred_m, pred_std = self.pred(idx, T)
        rmse = torch.sqrt(((pred_m - y)**2).mean()) / torch.sqrt((y**2).mean())
        mae = torch.abs(pred_m - y).mean() / torch.abs(y).mean()
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
            err_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y))) / torch.sqrt((self.y**2).mean())
            err_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte))) / torch.sqrt((yte ** 2).mean())
            print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %                 (ls, tau, err_tr, err_te))
            #with open('sparse_gptf_res.txt','a') as f:
            #    f.write('{:.5f}, {:.5f}\n'.format(err_tr.item(), err_te.item()))
                
            return err_tr.item(), err_te.item()
    
    def train(self, ind_te, yte, time_te, lr, max_epochs, domain):
        yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        time_te = time_te.reshape([-1,1])
        paras = self.U + [self.Z, self.mu, self.L, self.log_ls, self.log_tau]
        
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
        
        for epoch in tqdm(range(max_epochs)):
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                loss.backward(retain_graph=True)
                minimizer.step()
                curr = curr + self.B

            if (epoch + 1) % self.test_every == 0:
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



def test_server5(rank=3):
    data_file = '../data/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        lr = 1e-2
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_server_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))

def test_server5_extra(rank=3):
    data_file = '../data/server_10k_extra.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        lr = 1e-2
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_server_extra_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))

def test_ctr5(rank=3):
    data_file = '../data/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        lr = 1e-2
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_ctr_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))

def test_beijing5(rank=3):
    data_file = '../data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        lr = 1e-2
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:7'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_beijing_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))

def test_weather5(rank=3):
    data_file = '../data/ca_weather_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_weather_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))

def test_traffic5(rank=3):
    data_file = '../data/ca_traffic_30k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    ndims = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
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
        lr = 1e-2
        
        U = [np.random.rand(n, R) for n in ndims]
        
        model = GPTF(train_ind, train_y, train_time, U, m, batch_size, torch.device('cuda:0'))
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train(test_ind, test_y, test_time, lr, nepoch, '')    
        rmse_list.append(np.min(te_rmse_list))
        ll_list.append(np.max(te_ll_list))
        mae_list.append(np.min(te_mae_list))
    with open('GPTF_time.txt', 'a') as f:
        f.write('GPTF_ca_traffic_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5)))


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



