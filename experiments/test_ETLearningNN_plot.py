import numpy as np
from ETLearningNN import ETL
from scipy.io import savemat
import matplotlib.pyplot as plt
import argparse
import torch
from scipy.io import savemat

np.random.seed(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--test_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    if args.data == 'server':
        data_file = './data/server_10k.npy'
    
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
    cuda = args.cuda

    ind = data[fold]['tr_ind']
    t = data[fold]['tr_T']
    y = data[fold]['tr_y'].astype(np.float32)
    ind_te = data[fold]['te_ind']
    t_te = data[fold]['te_T']
    y_te = data[fold]['te_y'].astype(np.float32)



    model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding], torch.device('cuda:{}'.format(cuda)))
    tr_nrmse, tr_nmae, tr_ll, nrmse, nmae, ll = model.train(ind, t, y, ind_te, t_te, y_te, test_every=test_every, total_epoch=epoch, lr=lr, batch_size=batch) # 453
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
    # with open('log.txt', 'a') as f:
    #     f.write('{}_r_{}_f_{}:\trmse: {}\tmae: {}\tll: {}\n'.format(data_name, dim_embedding, fold, np.min(nrmse), np.min(nmae), np.max(ll)))
    
    
    model.load_state_dict(torch.load('opt_model_te'))

    time = np.linspace(0, 1, num=100)

    # W, L = model.ode_func.diffusion.get_W()
    # print(W)
    # print(L)
    # savemat('W.mat', {'W': W, 'L': L})

    U = model.get_trajectory(time)
    for k in range(len(nvec)):
        savemat('U_{}.mat'.format(k), {'u': U[k], 't': time})
            
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # for l in range(6):
                batch_idx = np.tile([i, j, k], [100, 1])
                batch_T = time.reshape((-1, 1))
                pred_m, pred_std = model.predict_np(batch_idx, batch_T)
                
                tr_sel = (ind[:, 0] == i) & (ind[:, 1] == j) & (ind[:, 2] == k)
                tr_tp = t[tr_sel]
                tr_yp = y[tr_sel]

                te_sel = (ind_te[:, 0] == i) & (ind_te[:, 1] == j) & (ind_te[:, 2] == k)
                te_tp = t_te[te_sel]
                te_yp = y_te[te_sel]

                savemat('i_{}_j_{}_k_{}.mat'.format(i, j, k), {
                    'pred_m': pred_m,
                    'pred_std': pred_std,
                    'tr_tp': tr_tp,
                    'tr_yp': tr_yp,
                    'te_tp': te_tp,
                    'te_yp': te_yp,
                    # 'min_te_time': np.min(te_T)
            })        


