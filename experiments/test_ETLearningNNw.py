import numpy as np
from ETLearningNNw import ETL
from scipy.io import savemat
import matplotlib.pyplot as plt
import argparse

np.random.seed(0)


def test_beijing5(dim_embedding=3):
    data_file = './data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding])
        tr_nrmse, tr_nmae, tr_ll, nrmse, nmae, ll = model.train(ind, t, y, ind_te, t_te, y_te, test_every=50, total_epoch=2000, lr=5e-2, batch_size=100) # 453
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        ll_list.append(np.max(ll))
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        plt.figure()
        plt.plot(np.arange(len(nrmse)), nrmse, label='te_nrmse')
        plt.plot(np.arange(len(nmae)), nmae, label='te_nmae')
        plt.plot(np.arange(len(tr_nrmse)), tr_nrmse, label='tr_nrmse')
        plt.plot(np.arange(len(tr_nmae)), tr_nmae, label='tr_nmae')
        plt.legend()
        plt.savefig('beijing_r_{}_f_{}.png'.format(dim_embedding, fold))
        plt.close()
    with open('log.txt', 'a') as f:
        f.write('beijing_5fold_r_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n\
                 ll: {}\t{}\n'.format(dim_embedding, 
                                     np.mean(rmse_list), 
                                     np.std(rmse_list)/np.sqrt(5), 
                                     np.mean(mae_list), 
                                     np.std(mae_list)/np.sqrt(5), 
                                     np.mean(ll_list), 
                                     np.std(ll_list)/np.sqrt(5)))

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
        data_file = './data/beijing_10k.npy'
    if args.data == 'ctr':
        data_file = './data/ctr_10k.npy'
    if args.data == 'server':
        data_file = './data/server_10k.npy'
    if args.data == 'weather':
        data_file = './data/tx_weather_10k.npy'
    if args.data == 'traffic':
        data_file = './data/ca_traffic_30k.npy'
    
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

    ind = data[fold]['tr_ind']
    t = data[fold]['tr_T']
    y = data[fold]['tr_y'].astype(np.float32)
    ind_te = data[fold]['te_ind']
    t_te = data[fold]['te_T']
    y_te = data[fold]['te_y'].astype(np.float32)

    model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding])
    tr_nrmse, tr_nmae, tr_ll, nrmse, nmae, ll = model.train(ind, t, y, ind_te, t_te, y_te, test_every=test_every, total_epoch=epoch, lr=lr, batch_size=batch) # 453
    plt.figure()
    plt.plot(np.arange(len(nrmse)), nrmse, label='te_nrmse')
    plt.plot(np.arange(len(nmae)), nmae, label='te_nmae')
    plt.plot(np.arange(len(tr_nrmse)), tr_nrmse, label='tr_nrmse')
    plt.plot(np.arange(len(tr_nmae)), tr_nmae, label='tr_nmae')
    plt.legend()
    plt.savefig('{}_r_{}_f_{}.png'.format(data_name, dim_embedding, fold))
    plt.close()
    np.save('{}_r_{}_f_{}.npy'.format(data_name, dim_embedding, fold), {
                                                                        'tr_nrmse': tr_nrmse,
                                                                        'tr_nmae': tr_nmae,
                                                                        'tr_ll': tr_ll,
                                                                        'nrmse': nrmse,
                                                                        'nmae': nmae,
                                                                        'll': ll})
    with open('log.txt', 'a') as f:
        f.write('{}_r_{}_f_{}_NNw:\trmse: {}\tmae: {}\n'.format(data_name, dim_embedding, fold, np.min(nrmse), np.min(nmae)))
    
    


