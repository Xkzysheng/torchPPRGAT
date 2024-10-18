# based on Sparse Tensor calculation, not apprantly faster than torch_PPRGAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch.cuda.amp import autocast

# 将稠密 adj 矩阵转换为 torch_sparse 支持的格式
def to_torch_sparse(adj):
    if not adj.is_sparse:
        adj = adj.to_sparse()  # 将稠密张量转换为稀疏张量
    coalesce = adj.coalesce()
    indices = coalesce.indices()
    values = coalesce.values()
    return torch.sparse_coo_tensor(indices, values, adj.size(), device=adj.device)


# 稀疏矩阵的乘法，用于替换 PyTorch 的默认稀疏乘法
def sparse_mm(adj, h):
    # 将邻接矩阵转换为稀疏格式
    adj_sparse = to_torch_sparse(adj)
    adj_sparse = adj_sparse.to_sparse_csr()
    h = h.to_sparse_csr()
    # 执行稀疏矩阵乘法
    result = adj_sparse @ h
    # 如果输入的 h 原本是 float16，则将结果转换回 float16
    if h.dtype == torch.float32:
        result = result.to(torch.float16)
    return result


# 使用批处理方式在GPU上计算PPR
def calc_ppr_topk_parallel_gpu_batch(adj, alpha, epsilon, topk, batch_size=64):
    csr_adj = adj.to_sparse()
    indptr = csr_adj.indices()[0]
    indices = csr_adj.indices()[1]
    deg = adj.sum(1)
    
    n_nodes = adj.size(0)
    PPR_eps_k = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=adj.device)
    alpha_eps = alpha * epsilon

    for batch_start in range(0, n_nodes, batch_size):
        batch_end = min(batch_start + batch_size, n_nodes)
        batch_nodes = range(batch_start, batch_end)
        
        for inode in batch_nodes:
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
            
            topk_vals, topk_indices = torch.topk(p, topk)
            PPR_eps_k[inode, topk_indices] = topk_vals

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
        alpha = float(self.alpha)
        epsilon = float(self.epsilon)
        topk = int(self.k)

        # 使用 CUDA 流并行执行 PPR 和 HW 计算
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            PPR_eps_k = calc_ppr_topk_parallel_gpu_batch(adj, alpha, epsilon, topk, batch_size=64)
        
        with torch.cuda.stream(stream2):
            HW = torch.mm(h.half(), self.W.half())

        # 等待所有流完成操作
        torch.cuda.synchronize()

        # 准备注意力机制输入，使用混合精度
        with autocast():
            e = self._prepare_attentional_mechanism_input(HW, PPR_eps_k)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            h_prime = sparse_mm(adj.half(), HW.half()).float()

        return h_prime.to_dense()
