# Pure PyTorch implecation on GPU, a little faster but still slow
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def calc_ppr_topk_parallel_gpu(adj, alpha, epsilon, topk):
    # adj 是稠密邻接矩阵，首先将其转为稀疏表示
    csr_adj = adj.to_sparse()
    indptr = csr_adj.indices()[0]
    indices = csr_adj.indices()[1]
    deg = adj.sum(1)
    
    # 初始化输出张量
    n_nodes = adj.size(0)
    PPR_eps_k = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=adj.device)
    
    alpha_eps = alpha * epsilon
    
    for inode in range(n_nodes):
        p = torch.zeros(n_nodes, dtype=torch.float32, device=adj.device)
        r = torch.zeros(n_nodes, dtype=torch.float32, device=adj.device)
        r[inode] = alpha
        q = [inode]
        
        while len(q) > 0:
            unode = q.pop()
            res = r[unode]
            p[unode] += res
            r[unode] = 0
            neighbors = indices[indptr[unode]:indptr[unode + 1]]
            deg_unode = deg[unode]
            for vnode in neighbors:
                _val = (1 - alpha) * res / deg_unode
                r[vnode] += _val
                if r[vnode] >= alpha_eps * deg[vnode] and vnode not in q:
                    q.append(vnode)
        
        # 获取前 topk
        topk_vals, topk_indices = torch.topk(p, topk)
        PPR_eps_k[inode, topk_indices] = topk_vals
    
    # 对 PPR_eps_k 进行归一化操作
    deg_sqrt = torch.sqrt(torch.clamp(deg, min=1e-12))
    deg_inv_sqrt = 1.0 / deg_sqrt
    
    row_list, col_list = PPR_eps_k.nonzero(as_tuple=True)
    PPR_eps_k[row_list, col_list] *= deg_sqrt[row_list] * deg_inv_sqrt[col_list]
    
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
        topk = int(self.k)

        # 直接在 GPU 上计算 PPR_eps_k
        PPR_eps_k = calc_ppr_topk_parallel_gpu(adj, alpha, epsilon, topk)
        e = self._prepare_attentional_mechanism_input(HW, PPR_eps_k)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, HW)
        return h_prime
