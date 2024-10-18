# mainly calculate by numba on CPU, SLOW!
import numba
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

@numba.njit(
    cache=True,
    locals={"_val": numba.float32, "res": numba.float32, "res_vnode": numba.float32},
)
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()
        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode] : indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            if r[vnode] >= alpha_eps * deg[vnode] and vnode not in q:
                q.append(vnode)
    return list(p.keys()), list(p.values())    


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, n_nodes, topk):
    nodes = np.arange(n_nodes)
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]

    PPR_eps_k = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        PPR_eps_k[i, js[i]] = vals[i]
    
    # 不对称归一化
    row_list, col_list = PPR_eps_k.nonzero()  # 确保加上括号
    deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
    deg_inv_sqrt = 1.0 / deg_sqrt
    
    for row, col in zip(row_list, col_list):
        PPR_eps_k[row, col] *= deg_sqrt[row] * deg_inv_sqrt[col]

    return PPR_eps_k


class PersonalizedPageRankGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_nodes, concat=False):
        super(PersonalizedPageRankGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_nodes = n_nodes
        self.concat = concat
        self.alpha = 0.25
        self.epsilon = 1e-4
        self.k = 32
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1))) # GAT的a
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a_ppr = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.xavier_uniform_(self.a_ppr, gain=1.414)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def _prepare_attentional_mechanism_input(self, h, PPR_eps_k):
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2.T
        e += self.a_ppr * PPR_eps_k
        return self.leakyrelu(e)

    def forward(self, h, adj):
        HW = torch.mm(h, self.W)
        alpha = float(self.alpha)
        epsilon = float(self.epsilon)
        n_nodes = int(self.n_nodes)
        topk = int(self.k)

        adj_np = adj.cpu().numpy()
        csr_adj = sp.csr_matrix(adj_np)
        indptr = csr_adj.indptr
        indices = csr_adj.indices
        deg = adj_np.sum(1)
        PPR_eps_k = calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, n_nodes, topk)
        PPR_eps_k = torch.tensor(PPR_eps_k, dtype=torch.float32, device=h.device)  # 将 PPR_eps_k 移动到和 h 相同的设备

        e = self._prepare_attentional_mechanism_input(HW, PPR_eps_k)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, HW)
        return h_prime
