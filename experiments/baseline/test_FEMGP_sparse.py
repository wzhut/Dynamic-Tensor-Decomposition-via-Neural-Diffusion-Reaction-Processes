import numpy as np
from FEMGP_sparse import FreqEmbedding
from scipy.io import savemat
np.random.seed(0)

def test_beijing5(rank=3, save=False):
    data_file = '../data/beijing_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 100,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': n_node,
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        if save:
            np.save('sparse_beijing_5fold_rank_{}_res.npy'.format(rank), res)
    
    with open('log.txt', 'a') as f:
        f.write('sparse_beijing_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_ctr5(rank=3, save=False):
    data_file = '../data/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 10,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': [5, 100, 100],
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        np.save('sparse_ctr_5fold_rank_{}_res.npy'.format(rank), res)
    with open('log.txt', 'a') as f:
        f.write('sparse_ctr_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5(rank=3, save=False):
    data_file = '../data/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 10,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': n_node,
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        if save:
            np.save('sparse_server_5fold_rank_{}_res.npy'.format(rank), res)
    with open('log.txt', 'a') as f:
        f.write('sparse_server_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5_extra(rank=3, save=False):
    data_file = '../data/server_10k_extra.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 10,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': n_node,
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        if save:
            np.save('sparse_server_5fold_rank_{}_res.npy'.format(rank), res)
    with open('log.txt', 'a') as f:
        f.write('sparse_server_extra_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))


def test_weather5(rank=3, save=False):
    data_file = '../data/ca_weather_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 10,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': n_node,
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        np.save('sparse_weather_5fold_rank_{}_res.npy'.format(rank), res)
    with open('log.txt', 'a') as f:
        f.write('sparse_weather_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_traffic5(rank=3, save=False):
    data_file = '../data/ca_traffic_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    n_node = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        tr_idx = data[fold]['tr_ind']
        tr_T = data[fold]['tr_T']
        tr_y = data[fold]['tr_y']
        te_idx = data[fold]['te_ind']
        te_T = data[fold]['te_T']
        te_y = data[fold]['te_y']

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        cfg = {
            'tr_idx': tr_idx,
            'tr_T': tr_T,
            'tr_y': tr_y,
            'te_idx': te_idx,
            'te_T': te_T,
            'te_y': te_y,
            'batch_size': 1000,
            'n_epoch': 5000,
            'jitter': 1e-5,
            'lr': 1e-2,
            'test_every': 10,
            'dim_embedding_u': rank,
            'dim_embedding_v': rank,
            'n_laggauss': 10,
            'n_pseudo1': n_node,
            'n_pseudo2': 100,
            'n_node': n_node,
            'cuda': True,
            # 'n_mc': 10
        }
        model = FreqEmbedding(cfg)
        tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list = model.train()
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        res.append({'tr_rmse_list':tr_rmse_list, 
            'tr_ll_list': tr_ll_list, 
            'tr_pred_m_list': tr_pred_m_list, 
            'tr_pred_std_list': tr_pred_std_list, 
            'te_rmse_list': te_rmse_list, 
            'te_ll_list': te_ll_list, 
            'te_pred_m_list': te_pred_m_list, 
            'te_pred_std_list': te_pred_std_list,
            'tr_mae_list': tr_mae_list,
            'te_mae_list': te_mae_list})
        rmse_list.append(np.min(te_rmse_list))
        mae_list.append(np.min(te_mae_list))
        ll_list.append(np.max(te_ll_list))
        np.save('sparse_traffic_5fold_rank_{}_res.npy'.format(rank), res)
    with open('log.txt', 'a') as f:
        f.write('sparse_traffic_5fold_rank_{}:\n \
                 rmse: {}\t{}\n\
                 ll: {}\t{}\n\
                 mae: {}\t{}\n'.format(rank, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(ll_list), np.std(ll_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))


if __name__ == '__main__':
    # test_traffic5(2)
    # test_traffic5(3)
    # test_traffic5(5)
    # test_traffic5(7)
    
    # test_weather5(2)
    # test_weather5(3)
    # test_weather5(5)
    # test_weather5(7)

    test_server5_extra(2)
    test_server5_extra(3)
    test_server5_extra(5)
    test_server5_extra(7)
    