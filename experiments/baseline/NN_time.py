import torch
import numpy as np
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, ModuleList, ParameterList
# from torchdiffeq import odeint_adjoint as odeint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from torch.utils import data as data_utils
from itertools import islice
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(0)
torch.random.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)

class NN(Module):
    def __init__(self, layers):
        super(NN, self).__init__()
        self.layers = layers
        self.act = Tanh()
        self.fc = ModuleList()
        for i in range(len(self.layers)-1):
            self.fc.append(Linear(self.layers[i], self.layers[i+1]))
    
    def forward(self, X):
        for i in range(len(self.layers) - 2):
            X = self.act(self.fc[i](X))
        X = self.fc[-1](X)
        return X



class ETL(Module):
    def __init__(self, nvec, dim_embedding, num_pseudo, reaction_layers,  device=torch.device('cuda:0')):
        super(ETL, self).__init__()
        self.device = device
        self.dim_embedding = dim_embedding # embedding dimension
        self.nvec = nvec
        self.nmod = len(nvec)
        self.num_node = np.sum(nvec)
        self.U = Parameter(torch.tensor(np.random.rand(self.num_node, self.dim_embedding)))
        self.f = NN([self.nmod * self.dim_embedding+1, 50, 1])

    def get_loss(self, batch_ind, batch_t, batch_y, N):
        batch_size = batch_ind.shape[0]
        pred = self.predict_(batch_ind, batch_t)
        loss = 0
        # loss -= 0.5 * N * self.log_tau
        # loss += 0.5 * torch.exp(self.log_tau) * ((pred_y - batch_y)**2).sum() / batch_size * N
        loss += 0.5 * ((batch_y - pred)**2).sum() / batch_size * N
        # loss += self.ode_func.KL_divergence()
        # loss += self.f.KL_divergence()
        return loss

    def train(self, ind, t, y, ind_te, t_te, y_te, batch_size=100, test_every=100, total_epoch=100, lr=1e-3):
        self.to(self.device)
        N = ind.shape[0]
        N_te = ind_te.shape[0]

        t = t.reshape(-1)
        ind = torch.tensor(ind, device=self.device, dtype=torch.int32)
        t = torch.tensor(t, device=self.device)
        y = torch.tensor(y, device=self.device)


        t_te = t_te.reshape(-1)
        ind_te = torch.tensor(ind_te, device=self.device, dtype=torch.int32)
        t_te = torch.tensor(t_te, device=self.device)
        y_te = torch.tensor(y_te, device=self.device)

        optimizer = Adam(self.parameters(), lr=lr)

        self.samples = []
        iter_count = 0

        nrmse_list = []
        nmae_list = []
        ll_list = []
        tr_nrmse_list = []
        tr_nmae_list = []
        tr_ll_list = []
        idx = np.arange(N)
        for epoch in tqdm(range(total_epoch)):
            np.random.shuffle(idx)
            num_batch = (N + batch_size - 1) // batch_size
            for iter in range(num_batch):
                batch_idx = idx[iter * batch_size : (iter+1) * batch_size]
                batch_ind = ind[batch_idx]
                batch_t = t[batch_idx]
                batch_y = y[batch_idx]

                optimizer.zero_grad()
                loss = self.get_loss(batch_ind, batch_t, batch_y, N)
                loss.backward()
                # print(self.ode_func[0].W.grad)
                optimizer.step()

                iter_count+=1

            if (epoch + 1) % test_every == 0:
                with torch.no_grad():
                    # testing error
                    nrmse, nmae = self.test(ind_te, t_te, y_te, 2000)
                    nrmse_list.append(nrmse.item())
                    nmae_list.append(nmae.item())
                    # ll_list.append(ll.item())
                    print('Epoch: {} NRMSE: {} NMAE: {}'.format(epoch+1, nrmse, nmae))

                    # training error
                    nrmse, nmae = self.test(ind, t, y, 2000)
                    tr_nrmse_list.append(nrmse.item())
                    tr_nmae_list.append(nmae.item())
                    # tr_ll_list.append(ll.item())
                    print('Training: NRMSE: {} NMAE: {}'.format(nrmse, nmae))
                    cur_lr = [group['lr'] for group in optimizer.param_groups][0]
                    print('Current LR:', cur_lr)


        return tr_nrmse_list, tr_nmae_list, tr_ll_list, nrmse_list, nmae_list, ll_list
    
    def test(self, ind, t, y, test_batch_size):
        N = ind.shape[0]
        num_batch = (N + test_batch_size - 1) // test_batch_size
        m_list = []
        var_list = []
        for i in range(num_batch):
            batch_ind = ind[i * test_batch_size: (i+1)*test_batch_size]
            batch_t = t[i * test_batch_size: (i+1)*test_batch_size]
            # batch_y = y[i * test_batch_size: (i+1)*test_batch_size]
            pred_m = self.predict_(batch_ind, batch_t)
            m_list.append(pred_m)
        if len(m_list) > 1:
            pred_m = torch.cat(m_list)
        else:
            pred_m = m_list[0]
        nrmse = torch.sqrt(((pred_m - y)**2).mean()) / torch.sqrt((y**2).mean())
        nmae = torch.abs(pred_m - y).mean() / torch.abs(y).mean()

        return nrmse, nmae#, ll

    # for current parameters
    def predict_(self, batch_ind, batch_t):
        U = torch.split(self.U, self.nvec, dim=0)
        embeddings = []
        for k in range(self.nmod):
            Uk = U[k]
            idx = batch_ind[:, k].long()
            # embedding_prod *= e[np.arange(batch_size).astype(np.int64), idx].view((batch_size, self.dim_embedding))
            embeddings.append(Uk[idx].view((-1, self.dim_embedding)))
        embeddings = torch.cat(embeddings + [batch_t.reshape(-1, 1)], 1)
        pred_mean = self.f(embeddings)
        return pred_mean

                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--test_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch', type=int, default=100)
    args = parser.parse_args()

    if args.data == 'beijing':
        data_file = '../data/beijing_10k.npy'
    if args.data == 'ctr':
        data_file = '../data/ctr_10k.npy'
    if args.data == 'server':
        data_file = '../data/server_10k.npy'
    if args.data == 'weather':
        data_file = '../data/weather_10k.npy'
    
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']

    data_name = args.data
    fold = args.fold
    lr = args.lr
    epoch = args.epoch
    batch = args.batch
    dim_embedding = args.rank
    test_every = args.test_every
    rmse_list = []
    mae_list = []
    for fold in range(5):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 20, 20, 20, dim_embedding])
        tr_nrmse, tr_nmae, tr_ll, nrmse, nmae, ll = model.train(ind, t, y, ind_te, t_te, y_te, test_every=test_every, total_epoch=epoch, lr=lr, batch_size=batch) # 453
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        # plt.figure()
    # plt.plot(np.arange(len(nrmse)), nrmse, label='te_nrmse')
    # plt.plot(np.arange(len(nmae)), nmae, label='te_nmae')
    # plt.plot(np.arange(len(tr_nrmse)), tr_nrmse, label='tr_nrmse')
    # plt.plot(np.arange(len(tr_nmae)), tr_nmae, label='tr_nmae')
    # plt.legend()
    # plt.savefig('{}_r_{}_f_{}.png'.format(data_name, dim_embedding, fold))
    # plt.close()
    # np.save('{}_r_{}_f_{}.npy'.format(data_name, dim_embedding, fold), {
    #                                                                     'tr_nrmse': tr_nrmse,
    #                                                                     'tr_nmae': tr_nmae,
    #                                                                     'tr_ll': tr_ll,
    #                                                                     'nrmse': nrmse,
    #                                                                     'nmae': nmae,
    #                                                                     'll': ll})
    with open('log.txt', 'a') as f:
        f.write('NN_time_{}_r_{}_:\trmse: {}\t {} \tmae: {}\t {}\n'.format(data_name, dim_embedding, np.mean(rmse_list), np.std(rmse_list) / np.sqrt(5), np.mean(mae_list), np.std(mae_list) / np.sqrt(5)))
    




