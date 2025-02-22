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

from o0_global_const import PartialProduct

LOG_STD_MAX = 2
LOG_STD_MIN = -20
ColumnFeatureNum = 18

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
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
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

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class AttentionDQNPolicy(nn.Module):
    def __init__(
        self,
        n_heads=8,
        hidden_dim=128,
        n_layers=8,
        node_dim=18,
        bit_width='8_bits', int_bit_width=8, str_bit_width=8,
        num_classes=252, EPS_START=0.9, EPS_END=0.10, 
        EPS_DECAY=1000, MAX_STAGE_NUM=16, device='cpu', is_column_mask=False
    ):
        # TODO: encoder add positional encoding
        super(AttentionDQNPolicy, self).__init__()
        # graph attention encoder
        self.encoder = GraphAttentionEncoder(
            n_heads,
            hidden_dim,
            n_layers,
            node_dim=node_dim
        )
        # MLP decoder
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
        )
        # hyperparameter
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width
        self.str_bit_width = str_bit_width
        self.num_classes = num_classes
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.device = device
        # is column mask
        self.is_column_mask = is_column_mask
        if is_column_mask:
            num_column_mask = int(self.int_bit_width*2*4)
            self.column_mask = np.ones(
                num_column_mask
            )
            for i in range(16):
                self.column_mask[i*4:(i+1)*4] = 0
                self.column_mask[num_column_mask-(i+1)*4:num_column_mask-i*4] = 0
            self.column_mask = (self.column_mask!=0)
            print(self.column_mask)

    def mask(self, state):
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)
        #state = torch.reshape(state, (2,int(self.int_bit_width*2)))
        state_np = state.cpu().numpy()
        np.savetxt('./build/state.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.int_bit_width*2))*4)
        #initial_state = state
        pp = np.zeros(int(self.int_bit_width*2))
        for i in range(int(self.int_bit_width*2)):
            pp[i] = PartialProduct[self.bit_width][i]

        for i in range(2,int(self.int_bit_width*2)):
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #initial_pp = pp
        for i in range(2,int(self.int_bit_width*2)):
            if (pp[i] == 2):
                legal_act.append((i,0))
                if (state[1][i] >= 1):
                    legal_act.append((i,3))
            if (pp[i] == 1):
                if (state[0][i] >= 1):
                    legal_act.append((i,2))
                if (state[1][i] >= 1):
                    legal_act.append((i,1))
        for act_col, action in legal_act:
            mask[act_col * 4 + action] = 1

        mask = (mask!=0)
        return torch.from_numpy(mask)

    def mask_with_legality(self, state):
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)

        #state = torch.reshape(state, (2,int(self.int_bit_width*2)))
        try:
            state_np = state.numpy()
        except:
            state_np = state.cpu().numpy()
        np.savetxt('./build/state.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.int_bit_width*2))*4)
        #initial_state = state
        pp = np.zeros(int(self.int_bit_width*2)+1)
        for i in range(int(self.int_bit_width*2)+1):
            pp[i] = PartialProduct[self.bit_width][i]

        for i in range(2,int(self.int_bit_width*2)):
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #initial_pp = pp
        for i in range(2,int(self.int_bit_width*2)):
            if (pp[i] == 2):
                legal_act.append((i,0))
                if (state[1][i] >= 1):
                    legal_act.append((i,3))
            if (pp[i] == 1):
                if (state[0][i] >= 1):
                    legal_act.append((i,2))
                if (state[1][i] >= 1):
                    legal_act.append((i,1))
        
        for act_col, action in legal_act:
            #state = initial_state
            df = pd.read_csv("./build/state.txt", header=None, sep=' ')
            df = df.to_numpy()
            state = torch.tensor(df)
            #pp = initial_pp
            #total column number cannot exceed 31
            pp = np.zeros(int(self.int_bit_width*2)+1)
            for i in range(int(self.int_bit_width*2)+1):
                pp[i] = PartialProduct[self.bit_width][i]

            for i in range(2,int(self.int_bit_width*2)):
                pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

            #change the CT structure
            if action == 0:
                state[1][act_col] = state[1][act_col] + 1
                pp[act_col] = pp[act_col] - 1
                pp[act_col+1] = pp[act_col+1] + 1
            elif action == 1:
                state[1][act_col] = state[1][act_col] - 1
                pp[act_col] = pp[act_col] + 1
                pp[act_col+1] = pp[act_col+1] - 1
            elif action == 2:
                state[1][act_col] = state[1][act_col] + 1
                state[0][act_col] = state[0][act_col] - 1
                pp[act_col] = pp[act_col] + 1
            elif action == 3:
                state[1][act_col] = state[1][act_col] - 1
                state[0][act_col] = state[0][act_col] + 1
                pp[act_col] = pp[act_col] - 1

            #legalization
            # mask 值为1 代表这个动作合法，为0代表不合法；
            for i in range(act_col+1,int(self.int_bit_width*2)+1):
                if (pp[i] == 1 or pp[i] == 2):
                    mask[act_col * 4 + action] = 1
                    break
                #column number restriction
                elif (i == int(self.int_bit_width*2)):
                    mask[act_col * 4 + action] = 1
                    break
                elif (pp[i] == 3):
                    state[0][i] = state[0][i] + 1
                    pp[i+1] = pp[i+1] + 1
                    pp[i] = 1           
                elif (pp[i] == 0):
                    if (state[1][i] >= 1):
                        state[1][i] = state[1][i] - 1
                        pp[i+1] = pp[i+1] -1
                        pp[i] = 1
                    else:
                        state[0][i] = state[0][i] - 1
                        pp[i+1] = pp[i+1] -1
                        pp[i] = 2
        #state = torch.tensor(state)
        index = torch.zeros((int(self.int_bit_width*2))*4)
        mask = (mask!=0)

        for i in range (0,(int(self.int_bit_width*2))*4):
            index[i] = i
        index = torch.masked_select(index, torch.from_numpy(mask))
        df = pd.read_csv("./build/state.txt", header=None, sep=' ')
        df = df.to_numpy()
        state = torch.tensor(df)
        for action in index:
            next_state = self.transition(np.array(state), action)
            next_state = np.reshape(next_state, (2,int(self.int_bit_width*2)))
            ct32, ct22, pp, stage_num = self.merge(next_state, 0)
            if stage_num >= self.MAX_STAGE_NUM: # 大于 max stage num 的动作也会被mask掉啊，约束这么多
                mask[int(action)] = 0
        mask = (mask!=0)
        # add column mask
        if self.is_column_mask:
            for i in range(len(self.column_mask)):
                if not self.column_mask[i]:
                    mask[i] = 0
            mask = (mask!=0)
        return torch.from_numpy(mask)

    def transition(self, state, action):
        state = np.reshape(state, (2,int(self.int_bit_width*2)))
        action = int(action)
        act_col = int(action // 4)
        action = int(action % 4)
        #total column number cannot exceed int(self.int_bit_width*2)
        pp = np.zeros(int(self.int_bit_width*2)+1)

        # partial products generated by the booth encoding
        for i in range(int(self.int_bit_width*2)+1):
            pp[i] = PartialProduct[self.bit_width][i]

        for i in range(1,int(self.int_bit_width*2)): # state 1 - (2self.int_bit_width-1) 有效列；第零列是补齐，方便运算；
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #change the CT structure, 执行动作，更新state记录的compressor 结构，以及partial product，partial product应该是用来legal的
        if action == 0:
            state[1][act_col] = state[1][act_col] + 1
            pp[act_col] = pp[act_col] - 1
            pp[act_col+1] = pp[act_col+1] + 1
        elif action == 1:
            state[1][act_col] = state[1][act_col] - 1
            pp[act_col] = pp[act_col] + 1
            pp[act_col+1] = pp[act_col+1] - 1
        elif action == 2:
            state[1][act_col] = state[1][act_col] + 1
            state[0][act_col] = state[0][act_col] - 1
            pp[act_col] = pp[act_col] + 1
        elif action == 3:
            state[1][act_col] = state[1][act_col] - 1
            state[0][act_col] = state[0][act_col] + 1
            pp[act_col] = pp[act_col] - 1
        #legalization
        # 第i列的动作只会影响第i列和第i+1列的partial product；
        for i in range(act_col+1,int(self.int_bit_width*2)):
            if (pp[i] == 1 or pp[i] == 2):
                break
            elif (pp[i] == 3):
                state[0][i] = state[0][i] + 1
                pp[i+1] = pp[i+1] + 1
                pp[i] = 1           
            elif (pp[i] == 0):
                if (state[1][i] >= 1):
                    state[1][i] = state[1][i] - 1
                    pp[i+1] = pp[i+1] -1
                    pp[i] = 1
                else:
                    state[0][i] = state[0][i] - 1
                    pp[i+1] = pp[i+1] -1
                    pp[i] = 2
        state = np.reshape(state, (1,2,int(self.int_bit_width*2))) # 这里的state 为什么reshape呢？
        return state 

    def merge(self, raw_state, thread_num=0):
        state = np.zeros_like(raw_state)
        state[0] = raw_state[0]
        state[1] = raw_state[1]
        #print(state)
        #merge
        stage_num = 0
        ct32 = np.zeros([1,int(self.int_bit_width*2)])
        ct22 = np.zeros([1,int(self.int_bit_width*2)])
        ct32[0] = state[0]
        ct22[0] = state[1]
        pp = np.zeros([1,int(self.int_bit_width*2)])
        pp[0] = PartialProduct[self.bit_width][:-1]

        for i in range(1,int(self.int_bit_width*2)):
            j = 0
            while(j <= stage_num):
                #print(stage_num)
                ct32[j][i] = state[0][i]
                ct22[j][i] = state[1][i]
                if (j==0):
                    pp[j][i] = pp[j][i]
                else:
                    pp[j][i] = pp[j-1][i] + ct32[j-1][i-1] + ct22[j-1][i-1]
                    
                if ((ct32[j][i]*3 + ct22[j][i]*2) <= pp[j][i]):
                    pp[j][i] = pp[j][i] - ct32[j][i]*2 - ct22[j][i-1]
                    state[0][i] = state[0][i] - ct32[j][i]
                    state[1][i] = state[1][i] - ct22[j][i]
                    break
                else :
                    if(j == stage_num):
                        stage_num += 1
                        ct32 = np.r_[ct32,np.zeros([1,int(self.int_bit_width*2)])]
                        ct22 = np.r_[ct22,np.zeros([1,int(self.int_bit_width*2)])]
                        pp = np.r_[pp,np.zeros([1,int(self.int_bit_width*2)])]
                        
                    if(pp[j][i]%3 == 0):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            ct22[j][i] = 0
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    if(pp[j][i]%3 == 1):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            ct22[j][i] = 0
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    if(pp[j][i]%3 == 2):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            if (ct22[j][i] >= 1):
                                ct22[j][i] = 1
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    pp[j][i] = pp[j][i] - ct32[j][i]*2 - ct22[j][i]
                    state[0][i] = state[0][i] - ct32[j][i]
                    state[1][i] = state[1][i] - ct22[j][i]
                j = j + 1
        sum = ct32.sum() + ct22.sum()
        sum = int(sum)

        #write to file
        file_name = './build/ct_test' + str(thread_num) + '.txt'
        f = open(file_name, mode = 'w')
        f.write(str(self.str_bit_width) + ' ' + str(self.str_bit_width))
        f.write('\n')
        f.write(str(sum))
        f.write('\n')
        for i in range(0,stage_num+1):
            for j in range(0,int(self.int_bit_width*2)):
                for k in range(0,int(ct32[i][int(self.int_bit_width*2)-1-j])): 
                    f.write(str(int(self.int_bit_width*2)-1-j))
                    f.write(' 1')
                    f.write('\n')
                for k in range(0,int(ct22[i][int(self.int_bit_width*2)-1-j])): 
                    f.write(str(int(self.int_bit_width*2)-1-j))
                    f.write(' 0')
                    f.write('\n')
        return ct32, ct22, pp, stage_num

    def _get_estimated_delay(self, ct32_i, ct22_i):
        try:
            nonzero32 = list(np.nonzero(ct32_i))[-1]
            min_delay_32 = nonzero32[-1]
        except:
            print(f"warning!!! ct32_i zero: {ct32_i}")
            min_delay_32 = 0
        
        try:
            nonzero31 = list(np.nonzero(ct22_i))[-1]
            min_delay_22 = nonzero31[-1]
        except:
            print(f"warning!!! ct22_i zero: {ct22_i}")
            min_delay_22 = 0
        
        return max(min_delay_32, min_delay_22)

    def _process_state(self, state, state_mask, ct32, ct22):
        # input matrix state and state mask
        # output sequence state
        """
            state feature vector definition
            [
                pp(1), position(1), mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                last_column: (4)
                next_column: (4)
            ]
        """
        num_column = state.shape[1]
        state_features = np.zeros(
            (num_column, ColumnFeatureNum)
        )
        column_features = np.zeros(
            (num_column, 4)
        )
        # get partial product information   
        initial_partial_product = PartialProduct[self.bit_width]
        for i in range(num_column):
            # pp
            state_features[i,0] = initial_partial_product[i]
            # position
            state_features[i,1] = i
            # mask
            cur_column_mask = state_mask[4*i:4*(i+1)]
            state_features[i,2:6] = np.array(cur_column_mask, dtype=np.float)
            
            # column features
            column_features[i,0] = state[0,i]
            column_features[i,1] = state[1,i]
            column_features[i,2] = 3*state[0,i] + 2*state[1,i]
            # i-th column 32 delay
            estimated_delay = self._get_estimated_delay(ct32[:,i], ct22[:,i])
            column_features[i,3] = estimated_delay
        
        for i in range(num_column):
            state_features[i,6:10] = column_features[i,:]
            if i == 0:
                state_features[i,10:14] = np.zeros((1,4))
            else:
                state_features[i,10:14] = column_features[i-1,:]
            
            if i == (num_column - 1):
                state_features[i,14:18] = np.zeros((1,4))
            else:
                state_features[i,14:18] = column_features[i+1,:]
        
        return state_features

    def _get_mask_state(self, ct32, ct22, stage_num):
        ct32_state = torch.tensor(np.array([ct32]))
        ct22_state = torch.tensor(np.array([ct22]))
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
            ct32_state = torch.cat((ct32_state, zeros), dim=1)
            ct22_state = torch.cat((ct22_state, zeros), dim=1)
        mask_state = torch.cat((ct32_state, ct22_state), dim=0).float()        
        return mask_state

    def select_action(self, state, steps_done, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        # compressor tree
        ct32, ct22, pp, stage_num = self.merge(state.cpu(), 0)
        # mask state 
        mask_state = self._get_mask_state(ct32, ct22, stage_num)
        # mask
        mask = self.mask_with_legality(mask_state)
        simple_mask = self.mask(mask_state)
        # seq state
        seq_state = self._process_state(
            state.cpu().numpy(), mask.numpy(), ct32, ct22
        )
        if deterministic:
            eps_threshold = 0.
        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold
        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(state.shape)
                mask = mask.to(self.device)
                seq_state = torch.tensor(seq_state).unsqueeze(0).float()
                q = self(seq_state, state_mask=mask)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action = q_distribution.sample()
                else:
                    action = q.max(1)[1]
                return action.view(1, 1), info
        else:
            index = torch.zeros((int(self.int_bit_width*2))*4)
            for i in range (0,(int(self.int_bit_width*2))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info

    def forward(self, x, is_target=False, state_mask=None):
        """
            x shape: (batch size, seq len, feature dim)
        """
        x = x.to(self.device)
        assert state_mask is not None 
        mask = state_mask
        h, h_avg = self.encoder(x)
        output = self.decoder(h_avg)
        output = output.masked_fill(~mask.to(self.device),-1000)
        return output

class AttentionGaussianPolicy(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_dim,
        n_layers,
        node_dim,
        bit_width
    ):
        # TODO: encoder add positional encoding
        super(AttentionGaussianPolicy, self).__init__()
        # graph attention encoder
        self.encoder = GraphAttentionEncoder(
            n_heads,
            hidden_dim,
            n_layers,
            node_dim=node_dim
        )
        # MLP decoder
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
        )
        
        self.bit_width = bit_width

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """

        encoder_h, encoder_h_avg = self.encoder(inputs)
        out = self.decoder(encoder_h_avg) # (mean, log_std)
        return out

    def get_mean_std(self, states):
        out = self.forward(states)
        mean, log_std = torch.chunk(out,2,-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.expand(mean.shape)

        return mean, log_std

    def _get_estimated_delay(self, ct32_i, ct22_i):
        try:
            nonzero32 = list(np.nonzero(ct32_i))
            min_delay_32 = nonzero32[-1]
        except:
            print(f"warning!!! ct32_i zero: {ct32_i}")
            min_delay_32 = 0
        
        try:
            nonzero31 = list(np.nonzero(ct22_i))
            min_delay_22 = nonzero31[-1]
        except:
            print(f"warning!!! ct32_i zero: {ct22_i}")
            min_delay_22 = 0
        
        return max(min_delay_32, min_delay_22)

    def _process_state(self, state, state_mask, ct32, ct22):
        # input matrix state and state mask
        # output sequence state
        """
            state feature vector definition
            [
                pp(1), position(1), mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                last_column: (4)
                next_column: (4)
            ]
        """
        num_column = state.shape[1]
        state_features = np.zeros(
            (num_column, ColumnFeatureNum)
        )
        column_features = np.zeros(
            (num_column, 4)
        )
        # get partial product information   
        initial_partial_product = PartialProduct[self.bit_width]
        for i in range(num_column):
            # pp
            state_features[i,0] = initial_partial_product[i]
            # position
            state_features[i,1] = i
            # mask
            cur_column_mask = state_mask[4*i:4*(i+1)]
            state_features[i,2:6] = float(cur_column_mask)
            
            # column features
            column_features[i,0] = state[0,i]
            column_features[i,1] = state[1,i]
            column_features[i,2] = 3*state[0,i] + 2*state[1,i]
            # i-th column 32 delay
            estimated_delay = self._get_estimated_delay(ct32[:,i:i+1], ct22[:,i:i+1])
            column_features[i,3] = estimated_delay
        
        for i in range(num_column):
            state_features[i,6:10] = column_features[i,:]
            if i == 0:
                state_features[i,10:14] = np.zeros((1,4))
            else:
                state_features[i,10:14] = column_features[i-1,:]
            
            if i == (num_column - 1):
                state_features[i,14:18] = np.zeros((1,4))
            else:
                state_features[i,14:18] = column_features[i+1,:]
        
        return state_features

    def action(self, state, state_mask, deterministic=False):
        # process states to get seq states
        seq_state = self._process_state(state, state_mask)
        # input seq states: (batch_size, seq len, node dim)
        mean, log_std = self.get_mean_std(seq_state)
        std = torch.exp(log_std)
        # normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            sample = Normal(torch.zeros_like(mean), torch.ones_like(mean)).sample()
            action = mean + std * sample
        
        tanh_action = torch.tanh(action)

        return tanh_action

    def log_prob(self, states, action=None, pretanh_action=None):
        if pretanh_action is None:
            assert action is not None
            pretanh_action = torch.log((1+action)/(1-action) +1e-6) / 2
        else:
            assert pretanh_action is not None
            action = torch.tanh(pretanh_action)
        mean, log_std = self.get_mean_std(states)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        pre_log_prob = normal.log_prob(pretanh_action)
        log_prob = pre_log_prob.sum(-1, keepdim=True) - torch.log(1 - action * action + 1e-6).sum(-1, keepdim=True)
        info = {}
        info['pre_log_prob'] = pre_log_prob
        info['mean'] = mean
        info['std'] = std
        info['entropy'] = normal.entropy()

        return log_prob, info 

if __name__ == '__main__':
    n_heads = 8
    hidden_dim = 128
    n_layers = 8
    node_dim = 18
    device = 'cuda:3'

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

    inputs = torch.randn((1,100,18), dtype=torch.float, device=device)
    policy = AttentionGaussianPolicy(
        n_heads,
        hidden_dim,
        n_layers,
        node_dim
    ).to(device)
    a = policy.action(inputs)
    print(a)