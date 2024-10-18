# Pure PyTorch implecation on GPU
# Calculate directly by PPR = a(I-(1-a)D^-1A)^-1 and selected k largest values for PPR's each line
# Fast and good!
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def get_PPR_topk(self, adj, k):
        alpha = self.alpha
        I_N = torch.eye(self.n_nodes, device='cuda')
        deg = adj.sum(1)
        D_inv = torch.diag(1.0 / deg)
        PPR = alpha*torch.inverse(I_N-(1-alpha)*D_inv@adj)
        topk_values, topk_indices = torch.topk(PPR, k, dim=1)
        # 创建一个掩码矩阵，所有 top-k 索引位置设为 True，其余位置设为 False
        topk_mask = torch.zeros_like(PPR, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        # 只保留 adj 非零位置上的元素，并将其他位置设为 0
        PPR_topk = torch.where(topk_mask & (adj != 0), PPR, torch.zeros_like(PPR))
        return PPR_topk

    def _prepare_attentional_mechanism_input(self, h, PPR_topk):
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2.T
        e += self.a_ppr * PPR_topk
        return self.leakyrelu(e)

    def forward(self, h, adj):
        HW = h @ self.W
        alpha = float(self.alpha)
        epsilon = float(self.epsilon)
        topk = int(self.k)

        # 直接在 GPU 上计算 PPR_topk
        PPR_topk = self.get_PPR_topk(adj, self.k)
        e = self._prepare_attentional_mechanism_input(HW, PPR_topk)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = attention @ HW
        return h_prime
