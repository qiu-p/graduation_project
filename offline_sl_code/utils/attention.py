import numpy as np
import random
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

import os
# import sys
# sys.path.insert(0, os.getcwd())

from ipdb import set_trace

ColumnFeatureNum = 8

"""
    三种 state 表示形式:
        matrix state: 2*num_column, 第0维是32compressor个数,第1维是22compressor个数
        mask state(image): 将matrix state根据stage展开,作为mask函数以及Resnet的输入
        seq state: 每个column建模为一个element,所有columns一起建模成sequence,每个column的特征由手工设计18维特征
"""

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            # Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        # input (batch_size, sequence lenghth, node feature dim)
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        # return (
        #     h,  # (batch_size, graph_size, embed_dim)
        #     h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        # )
        # h_graph = torch.max(h, dim=1)[0] + torch.mean(h, dim=1) # max tends to lead to gradient exploding and logprob output nan
        h_graph = torch.mean(h, dim=1)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h_graph  # average to get embedding of graph, (batch_size, embed_dim)
        )

class AttentionValueNet(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_dim,
        n_layers,
        node_dim,
        out_dim,
        non_linear='relu'
    ):
        # TODO: encoder add positional encoding
        super(AttentionValueNet, self).__init__()
        # graph attention encoder
        self.encoder = GraphAttentionEncoder(
            n_heads,
            hidden_dim,
            n_layers,
            node_dim=node_dim
        )
        # MLP decoder
        if non_linear == "tanh":
            self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, out_dim)
            )
        elif non_linear == "relu":
            self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim)
            )
        elif non_linear == "relu2":
            self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim)
            )
            # self.decoder = nn.Sequential(
            #         nn.Linear(hidden_dim, 1)
            # )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """

        encoder_h, encoder_h_avg = self.encoder(inputs)
        out = self.decoder(encoder_h_avg) # (mean, log_std)
        return out

if __name__ == '__main__':
    n_heads = 8
    hidden_dim = 128
    n_layers = 8
    node_dim = 8
    device = 'cuda:5'

    # embed_dim = 13
    # n_encode_layers = 2
    # normalization = 'batch'
    # device = 'cuda:0'

    # encoder = GraphAttentionEncoder(
    #     n_heads,
    #     embed_dim,
    #     n_encode_layers
    # ).to(device)
    
    # inputs = torch.randn((1,100,13), dtype=torch.float, device=device)

    # outputs, _ = encoder(inputs)
    # print(outputs.shape)

    inputs1 = torch.randn((1,32,8), dtype=torch.float, device=device)
    inputs2 = torch.randn((1,32,8), dtype=torch.float, device=device)
    seq_inputs1 = torch.cat([inputs1, inputs2])
    seq_inputs2 = torch.cat([inputs2, inputs1])
    
    policy = AttentionValueNet(
        n_heads,
        hidden_dim,
        n_layers,
        node_dim
    ).to(device)
    # policy = LSTMValueNet(
    #     8, 128
    # ).to(device)
    a = policy(inputs1)
    print(a)
    b = policy(inputs2)
    print(b)
    
    # repeat_inputs = inputs.repeat(25,1,1)
    # print(repeat_inputs.shape)
    c = policy(seq_inputs1)
    print(c)
    d = policy(seq_inputs2)
    print(d)
    policy.eval()
    for _ in range(2):
        e = policy(inputs1)
        print(e)
    policy.train()
    for _ in range(2):
        e = policy(inputs1)
        print(e)
    # inputs = np.random.rand(1,32,18)
    # policy = AttentionGaussianPolicy(
    #     n_heads,
    #     hidden_dim,
    #     n_layers,
    #     node_dim,
    #     device=device
    # ).to(device)
    # a = policy.action(inputs, deterministic=True)
    # print(a)
    # repeat_inputs = np.repeat(inputs, 25, axis=0)
    # print(repeat_inputs.shape)
    # b = policy.action(repeat_inputs, deterministic=True)
    # print(b)
    # value = LSTMValueNet(
    #     18, 128
    # ).to(device)
    # a = value(inputs)
    # print(a.shape)