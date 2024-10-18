# 2-layer PPRGAT model for node classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_PPRGATConv import PersonalizedPageRankGraphAttentionLayer as PPRGATLayer
# from slow_PPRGATConv import PersonalizedPageRankGraphAttentionLayer as PPRGATLayer
# from torch_PPRGATConv import PersonalizedPageRankGraphAttentionLayer as PPRGATLayer
# from beta_PPRGATConv import PersonalizedPageRankGraphAttentionLayer as PPRGATLayer

class PPRGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_nodes):
        """Dense version of PPRGAT (Single head, Without multi-heads)."""
        super(HAT, self).__init__()
        # hidden_layer
        self.attentions = PPRGATLayer(nfeat, nhid, n_nodes=n_nodes, concat=False)
        # out_layer
        self.out_att = PPRGATLayer(nhid, nclass, n_nodes=n_nodes, concat=False)

    def forward(self, x, adj):
        x = self.attentions(x, adj)
        x = F.elu(x)
        x = self.out_att(x, adj)
        # 分类问题，最后一层过一遍softmax
        return x # 配合crossentropy_loss (隐含softmax实现）
