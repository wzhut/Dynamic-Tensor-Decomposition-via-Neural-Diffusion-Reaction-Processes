import numpy as np
from ETLearningNN_weighted import ETL
from scipy.io import savemat
import matplotlib.pyplot as plt
import argparse

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
    args = parser.parse_args()

    if args.data == 'beijing':
        data_file = './data/beijing_10k.npy'
    if args.data == 'ctr':
        data_file = './data/ctr_10k.npy'
    if args.data == 'server':
        data_file = './data/server_10k.npy'
    if args.data == 'traffic':
        data_file = './data/ca_traffic_30k.npy'
    if args.data == 'weather':
        data_file = './data/ca_weather_15k.npy'
    
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
        f.write('{}_r_{}_f_{}_NN:\trmse: {}\tmae: {}\n'.format(data_name, dim_embedding, fold, np.min(nrmse), np.min(nmae)))
    
    


