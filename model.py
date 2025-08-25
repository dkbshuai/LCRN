# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:54:27 2024

@author: 60183
"""
import torch
from torch import nn
from torch.nn import Parameter
from functools import partial
from torch.nn import Linear
import torch.nn.functional as F


def setEmbedingModel(d_list,embed_dim):
    return nn.ModuleList([Mlp(d,embed_dim,embed_dim)for d in d_list])

def setDecoderModel(embed_dim,d_list):
    return nn.ModuleList([Mlp(embed_dim,d)for d in d_list])

class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm1d(out_dim)

        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        return out

class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.z_layer = Linear(dims[1], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_b0(self.z_layer(enc_h2))
        
        return z

class Model(nn.Module):
    def __init__(self, input_len, d_list, num_classes, embed_dim, dropout, exponent):
        super().__init__()
        
        dims = []
        for n_dim in d_list:

            linshidims = []
            for idim in range(1):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1000)
            dims.append(linshidims)
            
        self.encoder_list = nn.ModuleList([encoder(d_list[i], dims[i], embed_dim) for i in range(len(d_list))])
        self.view_num = input_len
        self.weights = Parameter(torch.softmax(torch.randn([1,self.view_num,1]),dim=1))
        self.exponent = exponent
        self.classification = nn.Linear(embed_dim, num_classes)
        self.act = nn.Sigmoid()
        self.MLP_QK = nn.Linear(embed_dim, 2*embed_dim)

        
        
    def forward(self,x,mask, mode):
        # x[v,n,d]
        # mask[bs, v]
        
        if mode =='train':
            for i,X in enumerate(x):
                mask_len = int(0.25 * X.size(-1))

                st = torch.randint(low=0,high=X.size(-1)-mask_len-1,size=(X.size(0),))
                # print(st,st+mask_len)
                mv = torch.ones_like(X)
                for j,e in enumerate(mv): 
                    mv[j,st[j]:st[j]+mask_len] = 0
                x[i] = x[i].mul(mv)
        
        B = mask.shape[0]
        individual_zs = []
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(x[enc_i])
            individual_zs.append(z_i)
            # summ += torch.diag(we[:, enc_i]).mm(z_i)
        z = torch.stack(individual_zs,dim=1) #[n v d]
        x_qk = z
        x_weighted = torch.pow(self.weights.expand(B,-1,-1),self.exponent)
        x_weighted_mask = torch.softmax(x_weighted.masked_fill(mask.unsqueeze(2)==0, -1e9),dim=1) #[B, self.view_num, 1]
        assert torch.sum(torch.isnan(x_weighted_mask)).item() == 0
        
        z = z.mul(x_weighted_mask)
        #print("z", z.shape)
        logi = self.classification(F.relu(z)) 
        logi = torch.einsum('bvd->bd',logi)
        pred_x = self.act(logi)
            
        return pred_x, x_qk

    
def get_model(input_len, d_list, num_classes, embed_dim=512, dropout=0., exponent=2):

    model = Model(input_len, d_list, num_classes, embed_dim, dropout, exponent)
    
    return model
