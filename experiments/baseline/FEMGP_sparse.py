import torch
import numpy as np

from torch.optim import Adam
from tqdm import tqdm
from numpy.polynomial.laguerre import laggauss
# from simulation_GP import R
from torch.linalg import solve, cholesky, inv
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import ReduceLROnPlateau


# mapping from t to u(t)

np.random.seed(0)
torch.random.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)

# exponential decay
class KernelRBF:
    def __init__(self, jitter=1e-5):
        self.jitter = jitter

    def cross3(self, X1, X2, ls):
        X1_norm = (X1 ** 2).sum(2, keepdim=True)
        X2_norm = (X2 ** 2).sum(2, keepdim=True)

        K = X1_norm - 2 * X1 @ X2.transpose(1, 2) + X2_norm.transpose(1, 2)
        K = K.unsqueeze(0)
        ls = ls.view((-1, 1, 1, 1))
        K = torch.exp(-0.5 * K / ls)
       
        return K
    
    def matrix3(self, X, ls):
        K = self.cross3(X, X, ls)
        K = K + self.jitter * torch.eye(K.shape[-1], dtype=torch.float64, device=X.device).unsqueeze(0).unsqueeze(0)
        return K

    def cross2(self, X1, X2, ls):
        X1_norm = (X1 ** 2).sum(1).reshape((-1, 1))
        X2_norm = (X2 ** 2).sum(1).reshape((-1, 1))
        K = X1_norm - 2 * X1 @ X2.T + X2_norm.T
        K = K.unsqueeze(0)
        ls = ls.view((-1, 1, 1))
        K = torch.exp(-0.5 * K / ls)
        return K

    def matrix2(self, X, ls):
        K = self.cross2(X, X, ls)
        K = K + self.jitter * torch.eye(X.shape[0], dtype=torch.float64, device=X.device).unsqueeze(0)
        return K

    def cross(self, X1, X2, ls):
        X1_norm = (X1 ** 2).sum(1).reshape((-1, 1))
        X2_norm = (X2 ** 2).sum(1).reshape((-1, 1))
        K = X1_norm - 2 * X1 @ X2.T + X2_norm.T
        K = torch.exp(-0.5 * K / ls)
        return K

    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        K = K + self.jitter * torch.eye(X.shape[0], dtype=torch.float64, device=X.device)
        return K

def init_kmeans(n_node, idx, dim_embedding, n_pseudo):
    n_mode = len(n_node)
    v = []
    Z = []
    for k in range(n_mode):
        v.append(np.random.rand(n_node[k], dim_embedding))
        sel_idx = idx[:, k]
        B = v[-1][sel_idx]
        centers = KMeans(n_clusters=n_pseudo[k]).fit(B).cluster_centers_
        # print(centers.shape)
        Z.append(centers)

    # Z = np.stack(Z)
    
    return v, Z
        
        

class FreqEmbedding:
    def __init__(self, cfg):

        self.jitter = cfg['jitter']
        
        self.tr_idx = torch.tensor(cfg['tr_idx'], dtype=torch.int64)
        self.tr_T = torch.tensor(cfg['tr_T'], dtype=torch.float64)
        self.tr_y = torch.tensor(cfg['tr_y'], dtype=torch.float64)

        self.te_idx = torch.tensor(cfg['te_idx'], dtype=torch.int64)
        self.te_T = torch.tensor(cfg['te_T'], dtype=torch.float64)
        self.te_y = torch.tensor(cfg['te_y'], dtype=torch.float64)

        self.batch_size = cfg['batch_size']

        self.lr = cfg['lr']
        self.n_epoch = cfg['n_epoch']
        self.test_every = cfg['test_every']
        self.cuda = cfg['cuda']

        self.n_laggauss = cfg['n_laggauss']
        # self.n_mc = cfg['n_mc']


        self.n_pseudo1 = cfg['n_pseudo1']
        self.n_pseudo2 = cfg['n_pseudo2']
        self.n_node = cfg['n_node'] # list of number of nodes
        self.n_mode = len(self.n_node)

        self.dim_embedding_u = cfg['dim_embedding_u'] # temporal embedding
        self.dim_embedding_v = cfg['dim_embedding_v'] # node embedding

        self.kernel = KernelRBF(self.jitter)

        x, w = laggauss(self.n_laggauss)
        x_norm = (x - np.mean(x)) / np.std(x)
        self.x_laggauss, self.w_laggauss, self.x_norm = torch.tensor(x, dtype=torch.float64), torch.tensor(w, dtype=torch.float64), torch.tensor(x_norm, dtype=torch.float64)
        

        self.log_ls_f1 = torch.tensor(np.zeros((self.n_mode, self.dim_embedding_u)), dtype=torch.float64) # K X R
        self.log_ls_f2 = torch.tensor(np.zeros((self.n_mode, self.dim_embedding_u)), dtype=torch.float64) # K X R

        init_v, init_Z = init_kmeans(self.n_node, self.tr_idx, self.dim_embedding_v, self.n_pseudo1)
        # sparse pseudo inputs and outputs poseterior
        self.Z_f = []
        self.M_f = []
        self.L_f_row = []
        self.L_f_col = []
        for k in range(self.n_mode):
            self.Z_f.append(torch.tensor(init_Z[k], dtype=torch.float64))
            self.M_f.append(torch.tensor(np.zeros((self.dim_embedding_u, self.n_pseudo1[k], self.n_laggauss)), dtype=torch.float64))
            self.L_f_row.append(torch.tensor(np.tile(np.eye(self.n_pseudo1[k]), (self.dim_embedding_u, 1)).reshape((self.dim_embedding_u, self.n_pseudo1[k], self.n_pseudo1[k])), dtype=torch.float64))
            self.L_f_col.append(torch.tensor(np.tile(np.eye(self.n_laggauss), (self.dim_embedding_u, 1)).reshape((self.dim_embedding_u, self.n_laggauss, self.n_laggauss)), dtype=torch.float64))
        # self.Z_f = torch.tensor(init_Z, dtype=torch.float64)
        # self.Z_f = torch.tensor(np.random.rand(self.n_mode, self.n_pseudo, self.dim_embedding_v), dtype=torch.float64) # K x Z x s
        # self.M_f = torch.tensor(np.random.randn(self.n_mode, self.dim_embedding_u, self.n_pseudo, self.n_laggauss), dtype=torch.float64) # K x R x Z x C
        # self.M_f = torch.tensor(np.zeros((self.n_mode, self.dim_embedding_u, self.n_pseudo1, self.n_laggauss)), dtype=torch.float64) # K x R x Z x C
        # self.L_f_row = torch.tensor(np.tile(np.eye(self.n_pseudo1), (self.n_mode * self.dim_embedding_u, 1)).reshape((self.n_mode, self.dim_embedding_u, self.n_pseudo1, self.n_pseudo1)), dtype=torch.float64) # K x R x Z x Z
        # self.L_f_col = torch.tensor(np.tile(np.eye(self.n_laggauss), (self.n_mode * self.dim_embedding_u, 1)).reshape(self.n_mode, self.dim_embedding_u, self.n_laggauss, self.n_laggauss), dtype=torch.float64) # K x R x C x C

        self.Z_g = torch.tensor(np.random.randn(self.n_pseudo2, self.dim_embedding_u * self.n_mode), dtype=torch.float64) # Z x KR
        self.m_g = torch.tensor(np.zeros((self.n_pseudo2, 1)), dtype=torch.float64)
        # self.m_g = torch.tensor(np.random.randn(self.n_pseudo, 1), dtype=torch.float64) # Z x 1
        self.L_g = torch.tensor(np.eye(self.n_pseudo2), dtype=torch.float64) # Z x Z

        self.log_ls_g = torch.tensor([0.], dtype=torch.float64)
        # self.log_ls_g = torch.tensor([np.log(self.n_mode * self.dim_embedding_u)], dtype=torch.float64)
        self.log_tau = torch.tensor([0.], dtype=torch.float64)

        # node identity embeddings
        self.v = []
        for i in range(self.n_mode):
            self.v.append(torch.tensor(init_v[i], dtype=torch.float64))
            # self.v.append(torch.tensor(np.random.rand(self.n_node[i], self.dim_embedding_v), dtype=torch.float64))


        if self.cuda:
            self.tr_idx = self.tr_idx.cuda()
            self.tr_T = self.tr_T.cuda()
            self.tr_y = self.tr_y.cuda()

            self.te_idx = self.te_idx.cuda()
            self.te_T = self.te_T.cuda()
            self.te_y = self.te_y.cuda()

            for i in range(len(self.v)):
                self.v[i] = self.v[i].cuda()

            # self.Z_f = self.Z_f.cuda()
            # self.M_f = self.M_f.cuda()
            # self.L_f_col = self.L_f_col.cuda()
            # self.L_f_row = self.L_f_row.cuda()
            for k in range(self.n_mode):
                self.Z_f[k] = self.Z_f[k].cuda()
                self.M_f[k] = self.M_f[k].cuda()
                self.L_f_row[k] = self.L_f_row[k].cuda()
                self.L_f_col[k] = self.L_f_col[k].cuda()

            self.Z_g = self.Z_g.cuda()
            self.m_g = self.m_g.cuda()
            self.L_g = self.L_g.cuda()
            
            self.x_laggauss = self.x_laggauss.cuda()
            self.x_norm = self.x_norm.cuda()
            self.w_laggauss = self.w_laggauss.cuda()

            self.log_ls_f1 = self.log_ls_f1.cuda()
            self.log_ls_f2 = self.log_ls_f2.cuda()

            self.log_ls_g = self.log_ls_g.cuda()
            self.log_tau = self.log_tau.cuda()

        # self.params = self.v + [self.Z_f, self.m_f, self.L_f, self.log_ls_f, self.log_tau] + [self.Z_g, self.m_g, self.L_g, self.log_ls_g]
        self.params = self.v + [self.log_ls_f1, self.log_ls_f2, self.log_tau, self.log_ls_g] + self.Z_f + self.M_f + self.L_f_row + self.L_f_col + [self.Z_g, self.m_g, self.L_g]
        for p in self.params:
            p.requires_grad_()        
        
        self.opt = Adam(self.params, lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.opt, 'min', patience=3, min_lr=1e-3)

    def train(self):
        N = self.tr_idx.shape[0]
        ob_idx = np.arange(N)
        n_batch = (N + self.batch_size - 1) // self.batch_size
        tr_rmse_list = []
        te_rmse_list = []
        tr_ll_list = []
        te_ll_list = []
        tr_pred_m_list = []
        te_pred_m_list = []
        tr_pred_std_list = []
        te_pred_std_list = []

        tr_mae_list = []
        te_mae_list = []
        with torch.no_grad():
                    # print(self.M_f)
            print('ls_f1: {}\tls_f2: {}\tls_g: {}\ttau:{}'.format(torch.exp(self.log_ls_f1).cpu().numpy(), torch.exp(self.log_ls_f2).cpu().numpy(), torch.exp(self.log_ls_g).cpu().numpy(), torch.exp(self.log_tau).cpu().numpy()))
            tr_rmse, tr_mae, tr_ll, tr_pred_m, tr_pred_std = self.test(self.tr_idx, self.tr_T, self.tr_y)
            te_rmse, te_mae, te_ll, te_pred_m, te_pred_std = self.test(self.te_idx, self.te_T, self.te_y)
            # print('Epoch: {} nELBO: {} trRMSE: {} teRMSE: {}'.format(epoch + 1, nELBO.item(), tr_rmse.item(), te_rmse.item()))
            tr_rmse_list.append(tr_rmse.item())
            te_rmse_list.append(te_rmse.item())
            tr_ll_list.append(tr_ll.item())
            te_ll_list.append(te_ll.item())
            tr_pred_m_list.append(tr_pred_m.view(-1).tolist())
            te_pred_m_list.append(te_pred_m.view(-1).tolist())
            tr_pred_std_list.append(tr_pred_std.view(-1).tolist())
            te_pred_std_list.append(te_pred_std.view(-1).tolist())

            tr_mae_list.append(tr_mae)
            tr_mae_list.append(te_mae)

        for epoch in tqdm(range(self.n_epoch)):
            np.random.shuffle(ob_idx)
            for i in range(n_batch):
                batch_ob_idx = ob_idx[i * self.batch_size: (i+1) * self.batch_size]
                batch_idx = self.tr_idx[batch_ob_idx]
                batch_T = self.tr_T[batch_ob_idx]
                batch_y = self.tr_y[batch_ob_idx]

                self.opt.zero_grad()
                nELBO = self.get_nELBO(batch_idx, batch_T, batch_y)
                nELBO.backward()
                self.opt.step()
            
            if (epoch + 1) % self.test_every == 0:
                with torch.no_grad():
                    # print(self.M_f)
                    print('ls_f1: {}\tls_f2: {}\tls_g: {}\ttau:{}'.format(torch.exp(self.log_ls_f1).cpu().numpy(), torch.exp(self.log_ls_f2).cpu().numpy(), torch.exp(self.log_ls_g).cpu().numpy(), torch.exp(self.log_tau).cpu().numpy()))
                    tr_rmse, tr_mae, tr_ll, tr_pred_m, tr_pred_std = self.test(self.tr_idx, self.tr_T, self.tr_y)
                    te_rmse, te_mae, te_ll, te_pred_m, te_pred_std = self.test(self.te_idx, self.te_T, self.te_y)
                    print('Epoch: {} nELBO: {} trRMSE: {} teRMSE: {}'.format(epoch + 1, nELBO.item(), tr_rmse.item(), te_rmse.item()))
                    tr_rmse_list.append(tr_rmse.item())
                    te_rmse_list.append(te_rmse.item())
                    tr_ll_list.append(tr_ll.item())
                    te_ll_list.append(te_ll.item())
                    tr_pred_m_list.append(tr_pred_m.view(-1).tolist())
                    te_pred_m_list.append(te_pred_m.view(-1).tolist())
                    tr_pred_std_list.append(tr_pred_std.view(-1).tolist())
                    te_pred_std_list.append(te_pred_std.view(-1).tolist())

                    tr_mae_list.append(tr_mae.item())
                    te_mae_list.append(te_mae.item())
                    self.scheduler.step(tr_rmse.item())
        return tr_rmse_list, tr_mae_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_mae_list, te_ll_list, te_pred_m_list, te_pred_std_list

    def get_nELBO(self, batch_idx, batch_T, batch_y):
        batch_size = batch_idx.shape[0]
        KL = 0
        U = []
        x = self.x_laggauss.view((1, -1)) # 1 x C
        w = self.w_laggauss.view((1, -1)) # 1 x C
        coef = torch.cos(x * batch_T) * w# batch_size x C
        coef = 2 * coef.view((1, -1, self.n_laggauss)) # 1 x batch_size x C
        # coef = coef.view((1, -1, self.n_laggauss)) / np.pi # 1 x batch_size x C
        
        W = self.x_norm.view((1, -1, 1)) # 1 x C x 1
        for k in range(self.n_mode):
            # input
            V = self.v[k][batch_idx[:, k], :].view((-1, 1, self.dim_embedding_v)) # batch_size x 1 x s
            # pseudo inputs and outputs
            Z = self.Z_f[k].view((1, self.n_pseudo1[k], -1)) # 1 x Z x s
            M = self.M_f[k].view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_laggauss)) # R x 1 x Z x C
            MT = M.transpose(2, 3) # R x 1 x C x Z
            Ltril_row = torch.tril(self.L_f_row[k]) # R x Z x Z
            Ltril_col = torch.tril(self.L_f_col[k]) # R x C x C
            S_row = Ltril_row @ Ltril_row.transpose(1, 2) # R x Z x Z
            S_col = Ltril_col @ Ltril_col.transpose(1, 2) # R x C x C 
            S_row = S_row.view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_pseudo1[k])) # R x 1 x Z x Z
            S_col = S_col.view((self.dim_embedding_u, 1, self.n_laggauss, self.n_laggauss)) # R x 1 x C x C

           
            # kernels
            ls1 = torch.exp(self.log_ls_f1[k]) # R
            ls2 = torch.exp(self.log_ls_f2[k]) # R

            kZZ = self.kernel.matrix3(Z, ls1) # R x 1 x Z x Z
            kVV = self.kernel.matrix3(V, ls1) # R x batch_size x 1 x 1
            kZV = self.kernel.cross3(Z, V, ls1) # R x batch_size x Z x 1
            kWW = self.kernel.matrix3(W, ls2) # R x 1 x C x C

            alpha = solve(kZZ, kZV) # R x batch_size x Z x 1
            alphaT = alpha.transpose(2, 3) # R x batch_size x 1 x Z
            m = MT @ alpha # R x batch_size x C x 1
            kWW_diag = torch.diagonal(kWW, dim1=2, dim2=3).view((self.dim_embedding_u, 1, self.n_laggauss, 1))
            S_col_diag = torch.diagonal(S_col, dim1=2, dim2=3).view((self.dim_embedding_u, 1, self.n_laggauss, 1))
            # Sigma = kWW * (kVV - alphaT @ kZZ @ alpha) + S_col * (alphaT @ S_row @ alpha) # R x batch_size x C x C
            # Sigma = Sigma + 1e-5 * torch.eye(self.n_laggauss, dtype=torch.float64, device=Sigma.device).view((1, 1, self.n_laggauss, self.n_laggauss))
            # Ltril = cholesky(Sigma)
            # u = m + Ltril @ torch.empty_like(m).normal_() # R x batch_size x C x 1
            # sigma = torch.diagonal(Sigma, dim1=2, dim2=3).view((self.dim_embedding_u, batch_size, self.n_laggauss, 1))
            sigma = kWW_diag *(kVV - alphaT @ kZZ @ alpha) + S_col_diag * (alphaT @ S_row @ alpha) # R x batch_size x C x 1 

            u = m + torch.empty_like(m).normal_() * torch.sqrt(sigma)
            u = (coef * u.view((self.dim_embedding_u, batch_size, self.n_laggauss))).sum(2).T # batch_size x R
            U.append(u)
            # KL
            # print(torch.diagonal(solve(kWW, S_col), dim1=2, dim2=3).sum(2).shape)
                         # R x 1 x C x C
            KL += 0.5 * ( (torch.diagonal(solve(kWW, S_col), dim1=2, dim2=3).sum(2) * torch.diagonal(solve(kZZ, S_row), dim1=2, dim2=3).sum(2)).sum() +
                        # R x 1 x C x C       R x 1 x C x Z     R x 1 x Z x Z    R x 1 x Z x C
                        torch.diagonal(solve(kWW, MT) @ solve(kZZ, M), dim1=2, dim2=3).sum() +
                                                                                                                            # R x C x C                                                           R x Z x Z
                        self.n_pseudo1[k] * torch.logdet(kWW).sum() + self.n_laggauss * torch.logdet(kZZ).sum() - self.n_pseudo1[k] * torch.log(torch.diagonal(Ltril_col, dim1=1, dim2=2)**2).sum() - self.n_laggauss * torch.log(torch.diagonal(Ltril_row, dim1=1, dim2=2)**2).sum()
                    )
        # print(KL)
        # input()
        U = torch.cat(U, dim=1) # batch_size x KR
        # print(U)
        U = U.view((batch_size, 1, -1)) # batch_size x 1 x KR

        ls = torch.exp(self.log_ls_g)
        Z = self.Z_g.view((1, self.n_pseudo2, -1)) # 1 x Z x KR
        # print(Z)
        m_g = self.m_g.view((1, self.n_pseudo2, 1)) # 1 x Z x 1
        kZZ = self.kernel.matrix3(Z, ls).view((1, self.n_pseudo2, self.n_pseudo2)) #  1 x Z x Z
        kZU = self.kernel.cross3(Z, U, ls).view((batch_size, self.n_pseudo2, 1)) #  batch_size x Z x 1
        # print(kZU)
        # input()
        kUU = self.kernel.matrix3(U, ls).view((batch_size, 1, 1)) #  batch_size x 1 x 1
        Ltril = torch.tril(self.L_g) # Z x Z
        S = (Ltril @ Ltril.T).view((1, self.n_pseudo2, self.n_pseudo2)) # 1 x Z x Z

        alpha = solve(kZZ, kZU) # batch_size x Z x 1
        alphaT = alpha.transpose(1, 2) # batch_size x 1 x Z
        m = alphaT @ m_g # batch_size x 1 x 1
        sigma = kUU - alphaT @ (kZZ - S) @ alpha # batch_size x 1 x 1
        g = m + torch.empty_like(m).normal_() * torch.sqrt(sigma)
        g = g.view((-1, 1)) # batch_size x 1

        # KL
        kZZ_inv = inv(kZZ)
        m_gT = m_g.transpose(1, 2)
        KL += 0.5 * (
                (kZZ_inv * S).sum() + (m_gT @ kZZ_inv @ m_g).sum()  
                + torch.logdet(kZZ).sum() 
                - torch.log(torch.diag(Ltril)**2).sum()
        )
        N = self.tr_y.shape[0]
        LL = 0.5 * N * self.log_tau - 0.5 * torch.exp(self.log_tau) * N / batch_size * ((batch_y-g)**2).sum() 
        ELBO = LL - KL
        return -ELBO

    def pred(self, batch_idx, batch_T, batch_y):
        batch_size = batch_idx.shape[0]
        U = []
        x = self.x_laggauss.view((1, -1)) # 1 x C
        w = self.w_laggauss.view((1, -1)) # 1 x C
        beta = torch.cos(x * batch_T) * w# batch_size x C
        beta = 2 * beta.view((1, -1, self.n_laggauss)) # 1 x batch_size x C
        # beta = beta.view((1, -1, self.n_laggauss)) / np.pi

        
        # W = self.x_norm.view((1, -1, 1)) # 1 x C x 1
        for k in range(self.n_mode):
            # input
            V = self.v[k][batch_idx[:, k], :].view((-1, 1, self.dim_embedding_v)) # batch_size x 1 x s
            # pseudo inputs and outputs
            Z = self.Z_f[k].view((1, self.n_pseudo1[k], -1)) # 1 x Z x s
            M = self.M_f[k].view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_laggauss)) # R x 1 x Z x C
            MT = M.transpose(2, 3) # R x 1 x C x Z
            Ltril_row = torch.tril(self.L_f_row[k]) # R x Z x Z
            Ltril_col = torch.tril(self.L_f_col[k]) # R x C x C
            S_row = Ltril_row @ Ltril_row.transpose(1, 2) # R x Z x Z
            S_col = Ltril_col @ Ltril_col.transpose(1, 2) # R x C x C 
            S_row = S_row.view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_pseudo1[k])) # R x 1 x Z x Z
            S_col = S_col.view((self.dim_embedding_u, 1, self.n_laggauss, self.n_laggauss)) # R x 1 x C x C

            # kernels
            ls1 = torch.exp(self.log_ls_f1[k]) # R
            # ls2 = torch.exp(self.log_ls_f2[k]) # R

            kZZ = self.kernel.matrix3(Z, ls1) # R x 1 x Z x Z
            # kVV = self.kernel.matrix3(V, ls1) # R x batch_size x 1 x 1
            kZV = self.kernel.cross3(Z, V, ls1) # R x batch_size x Z x 1
            # kWW = self.kernel.matrix3(W, ls2) # R x 1 x C x C

            alpha = solve(kZZ, kZV) # R x batch_size x Z x 1
            alphaT = alpha.transpose(2, 3) # R x batch_size x 1 x Z
            m = MT @ alpha # R x batch_size x C x 1
            # Sigma = kWW * (kVV - alphaT @ kZZ @ alpha) + S_col * (alphaT @ S_row @ alpha) # R x batch_size x C x C
            # Ltril = cholesky(Sigma)
            u = m #+ Ltril @ torch.empty_like(m).normal_() # R x batch_size x C x 1
            u = (beta * u.view((self.dim_embedding_u, batch_size, self.n_laggauss))).sum(2).T # batch_size x R
            U.append(u)

        U = torch.cat(U, dim=1) # batch_size x KR
        U = U.view((batch_size, 1, -1)) # batch_size x 1 x KR

        ls = torch.exp(self.log_ls_g)
        Z = self.Z_g.view((1, self.n_pseudo2, -1)) # 1 x Z x KR
        m_g = self.m_g.view((1, self.n_pseudo2, 1)) # 1 x Z x 1
        kZZ = self.kernel.matrix3(Z, ls).view((1, self.n_pseudo2, self.n_pseudo2)) #  1 x Z x Z
        kZU = self.kernel.cross3(Z, U, ls).view((batch_size, self.n_pseudo2, 1)) #  batch_size x Z x 1
        kUU = self.kernel.matrix3(U, ls).view((batch_size, 1, 1)) #  batch_size x 1 x 1
        Ltril = torch.tril(self.L_g) # Z x Z
        S = (Ltril @ Ltril.T).view((1, self.n_pseudo2, self.n_pseudo2)) # 1 x Z x Z

        alpha = solve(kZZ, kZU) # batch_size x Z x 1
        alphaT = alpha.transpose(1, 2) # batch_size x 1 x Z
        m = alphaT @ m_g # batch_size x 1 x 1
        m = m.view((-1, 1))
        sigma = kUU - alphaT @ (kZZ - S) @ alpha # batch_size x 1 x 1
        std = torch.sqrt(sigma)
        std = std.view((-1, 1))
        # g = m #+ torch.empty_like(m).normal_() * torch.sqrt(sigma)
        # g = g.view((-1, 1)) # batch_size x 1

        # se = ((g - batch_y) ** 2).sum()
        return m, std

    def test(self, idx, T, y):
        N = idx.shape[0]
        n_batch = (N + self.batch_size - 1) // self.batch_size
        pred_m = []
        pred_std = []
        for i in range(n_batch):
            batch_idx = idx[i * self.batch_size: (i+1) * self.batch_size]
            batch_T = T[i * self.batch_size: (i+1) * self.batch_size]
            batch_y = y[i * self.batch_size: (i+1) * self.batch_size]
            m, std = self.pred(batch_idx, batch_T, batch_y)
            pred_m.append(m)
            pred_std.append(std)
        pred_m = torch.cat(pred_m, dim=0)
        pred_std = torch.cat(pred_std, dim=0)

        rmse = torch.sqrt(((pred_m - y)**2).mean()) / torch.sqrt((y**2).mean())
        mae = torch.abs(pred_m - y).mean() / torch.abs(y).mean()
        # ll = -0.5 * torch.exp(self.log_tau) * (pred_m - y) ** 2 + 0.5 * self.log_tau - 0.5 * np.log(2 * np.pi)
        sigma2 = torch.exp(-self.log_tau) + pred_std ** 2
        ll = -0.5 / sigma2 * (pred_m - y)**2  - 0.5 * torch.log(sigma2) - 0.5 * np.log(2 * np.pi)
        ll = ll.mean()
        return rmse, mae, ll, pred_m, pred_std
        # rmse = 
    
    def get_trajectory(self, mode, node, time):
        with torch.no_grad():
            T = torch.tensor(time, dtype=torch.float64).view((-1, 1))
            if self.cuda:
                T = T.cuda()
            k = mode
            i = node
            x = self.x_laggauss.view((1, -1)) # 1 x C
            w = self.w_laggauss.view((1, -1)) # 1 x C
            beta = torch.cos(x * T) * w# T x C
            beta = 2 * beta.view((1, -1, self.n_laggauss)) # 1 x T x C

            W = self.x_norm.view((1, -1, 1)) # 1 x C x 1

            V = self.v[k][i, :].view((-1, 1, self.dim_embedding_v)) # 1 x 1 x s
            # pseudo inputs and outputs
            Z = self.Z_f[k].view((1, self.n_pseudo1[k], -1)) # 1 x Z x s
            M = self.M_f[k].view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_laggauss)) # R x 1 x Z x C
            MT = M.transpose(2, 3) # R x 1 x C x Z
            Ltril_row = torch.tril(self.L_f_row[k]) # R x Z x Z
            Ltril_col = torch.tril(self.L_f_col[k]) # R x C x C
            S_row = Ltril_row @ Ltril_row.transpose(1, 2) # R x Z x Z
            S_col = Ltril_col @ Ltril_col.transpose(1, 2) # R x C x C 
            S_row = S_row.view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_pseudo1[k])) # R x 1 x Z x Z
            S_col = S_col.view((self.dim_embedding_u, 1, self.n_laggauss, self.n_laggauss)) # R x 1 x C x C

            # kernels
            ls1 = torch.exp(self.log_ls_f1[k]) # R
            ls2 = torch.exp(self.log_ls_f2[k]) # R

            kZZ = self.kernel.matrix3(Z, ls1) # R x 1 x Z x Z
            kVV = self.kernel.matrix3(V, ls1) # R x 1 x 1 x 1
            kZV = self.kernel.cross3(Z, V, ls1) # R x 1 x Z x 1
            kWW = self.kernel.matrix3(W, ls2) # R x 1 x C x C

            alpha = solve(kZZ, kZV) # R x 1 x Z x 1
            alphaT = alpha.transpose(2, 3) # R x 1 x 1 x Z
            m = MT @ alpha # R x 1 x C x 1
            kWW_diag = torch.diagonal(kWW, dim1=2, dim2=3).view((self.dim_embedding_u, 1, self.n_laggauss, 1))
            S_col_diag = torch.diagonal(S_col, dim1=2, dim2=3).view((self.dim_embedding_u, 1, self.n_laggauss, 1))
            sigma = kWW_diag *(kVV - alphaT @ kZZ @ alpha) + S_col_diag * (alphaT @ S_row @ alpha) # R x 1 x C x 1 
            # Ltril = cholesky(Sigma)
            u = m #+ Ltril @ torch.empty_like(m).normal_() # R x 1 x C x 1
            # 1 x T x C       R x 1 x C       R x T x C
            u = (beta * u.view((self.dim_embedding_u, 1, self.n_laggauss))).sum(2).T # T x R
            # 
            sigma = (beta**2 * sigma.view((self.dim_embedding_u, 1, self.n_laggauss))).sum(2).T
            std = torch.sqrt(sigma) # T X R
        return u.tolist(), std.tolist()


    def pred_np(self, batch_idx, batch_T, batch_y):
        with torch.no_grad():
            batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
            batch_T = torch.tensor(batch_T, dtype=torch.float64)
            if self.cuda:
                batch_idx = batch_idx.cuda()
                batch_T = batch_T.cuda()
            batch_size = batch_idx.shape[0]
            U = []
            x = self.x_laggauss.view((1, -1)) # 1 x C
            w = self.w_laggauss.view((1, -1)) # 1 x C
            beta = torch.cos(x * batch_T) * w# batch_size x C
            beta = 2 * beta.view((1, -1, self.n_laggauss)) # 1 x batch_size x C
            # beta = beta.view((1, -1, self.n_laggauss)) / np.pi

            
            # W = self.x_norm.view((1, -1, 1)) # 1 x C x 1
            for k in range(self.n_mode):
                # input
                V = self.v[k][batch_idx[:, k], :].view((-1, 1, self.dim_embedding_v)) # batch_size x 1 x s
                # pseudo inputs and outputs
                Z = self.Z_f[k].view((1, self.n_pseudo1[k], -1)) # 1 x Z x s
                M = self.M_f[k].view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_laggauss)) # R x 1 x Z x C
                MT = M.transpose(2, 3) # R x 1 x C x Z
                Ltril_row = torch.tril(self.L_f_row[k]) # R x Z x Z
                Ltril_col = torch.tril(self.L_f_col[k]) # R x C x C
                S_row = Ltril_row @ Ltril_row.transpose(1, 2) # R x Z x Z
                S_col = Ltril_col @ Ltril_col.transpose(1, 2) # R x C x C 
                S_row = S_row.view((self.dim_embedding_u, 1, self.n_pseudo1[k], self.n_pseudo1[k])) # R x 1 x Z x Z
                S_col = S_col.view((self.dim_embedding_u, 1, self.n_laggauss, self.n_laggauss)) # R x 1 x C x C

                # kernels
                ls1 = torch.exp(self.log_ls_f1[k]) # R
                # ls2 = torch.exp(self.log_ls_f2[k]) # R

                kZZ = self.kernel.matrix3(Z, ls1) # R x 1 x Z x Z
                # kVV = self.kernel.matrix3(V, ls1) # R x batch_size x 1 x 1
                kZV = self.kernel.cross3(Z, V, ls1) # R x batch_size x Z x 1
                # kWW = self.kernel.matrix3(W, ls2) # R x 1 x C x C

                alpha = solve(kZZ, kZV) # R x batch_size x Z x 1
                alphaT = alpha.transpose(2, 3) # R x batch_size x 1 x Z
                m = MT @ alpha # R x batch_size x C x 1
                # Sigma = kWW * (kVV - alphaT @ kZZ @ alpha) + S_col * (alphaT @ S_row @ alpha) # R x batch_size x C x C
                # Ltril = cholesky(Sigma)
                u = m #+ Ltril @ torch.empty_like(m).normal_() # R x batch_size x C x 1
                u = (beta * u.view((self.dim_embedding_u, batch_size, self.n_laggauss))).sum(2).T # batch_size x R
                U.append(u)

            U = torch.cat(U, dim=1) # batch_size x KR
            U = U.view((batch_size, 1, -1)) # batch_size x 1 x KR

            ls = torch.exp(self.log_ls_g)
            Z = self.Z_g.view((1, self.n_pseudo2, -1)) # 1 x Z x KR
            m_g = self.m_g.view((1, self.n_pseudo2, 1)) # 1 x Z x 1
            kZZ = self.kernel.matrix3(Z, ls).view((1, self.n_pseudo2, self.n_pseudo2)) #  1 x Z x Z
            kZU = self.kernel.cross3(Z, U, ls).view((batch_size, self.n_pseudo2, 1)) #  batch_size x Z x 1
            kUU = self.kernel.matrix3(U, ls).view((batch_size, 1, 1)) #  batch_size x 1 x 1
            Ltril = torch.tril(self.L_g) # Z x Z
            S = (Ltril @ Ltril.T).view((1, self.n_pseudo2, self.n_pseudo2)) # 1 x Z x Z

            alpha = solve(kZZ, kZU) # batch_size x Z x 1
            alphaT = alpha.transpose(1, 2) # batch_size x 1 x Z
            m = alphaT @ m_g # batch_size x 1 x 1
            m = m.view((-1, 1))
            sigma = kUU - alphaT @ (kZZ - S) @ alpha # batch_size x 1 x 1
            std = torch.sqrt(sigma)
            std = std.view((-1, 1))
            # g = m #+ torch.empty_like(m).normal_() * torch.sqrt(sigma)
            # g = g.view((-1, 1)) # batch_size x 1

            # se = ((g - batch_y) ** 2).sum()
            return m.tolist(), std.tolist()
