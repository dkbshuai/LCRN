# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:14:16 2024

@author: 60183
"""

import torch
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import matplotlib.pyplot as plt


class Loss(nn.Module):
    def __init__(self,alpha,gamma,device):
        super(Loss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def contrastive_loss(self, x_qk, label, inc_V_ind, inc_L_ind):
        # x = [bs, v, dim] 
        # label = [bs, c]  c = dim
        # inc_V_ind = [bs, v]
        # inc_L_ind = [bs, c]
        
        q, k = x_qk.chunk(2, dim = -1)
        x_q = F.normalize(q, p=2, dim=-1)
        x_k = F.normalize(k, p=2, dim=-1)
        x_q, x_k = x_q.transpose(0,1), x_k.transpose(0,1)  #[v,bs,d]
        
        v = x_q.size(0)
        
        view_loss_1 = 0
        view_loss_2 = 0
        view_loss_3 = 0
        view_loss_4 = 0
        
        for i in range(v):
            for j in range(i+1, v):
                x_q_1, x_q_2 = x_q[i,:,:], x_q[j,:,:]
                x_k_1, x_k_2 = x_k[i,:,:], x_k[j,:,:]
                mask_v1, mask_v2 = inc_V_ind[:, i], inc_V_ind[:, j]
                mask_both_miss = mask_v1.mul(mask_v2).bool()
                x_q_1, x_q_2 = x_q_1[mask_both_miss], x_q_2[mask_both_miss]
                x_k_1, x_k_2 = x_k_1[mask_both_miss], x_k_2[mask_both_miss]
                n = x_q_1.size(0)
                
                
                label_mask = label[mask_both_miss]
                inc_L_ind_mask = inc_L_ind[mask_both_miss]
                valid_labels_sum = torch.matmul(inc_L_ind_mask.float(), inc_L_ind_mask.float().T)
                
                label_or_mun = label_mask.unsqueeze(2) + label_mask.T.unsqueeze(0)
                label_or_mun = torch.sum(torch.where(label_or_mun > 1, 1, label_or_mun),dim=1)
                
                label_matrix = (torch.matmul(label_mask, label_mask.T).mul(valid_labels_sum) / (label_or_mun.mul(valid_labels_sum) + 1e-9)).fill_diagonal_(0)
                
                label_matrix = torch.where(label_matrix == 1, 1e-5, label_matrix)
                label_matrix = torch.where(label_matrix == 0, 1, label_matrix)
                label_matrix = torch.where(label_matrix < 1, 0, label_matrix)
                
                similarity_mat_1 = (torch.matmul(x_q_1, x_k_2.T)/ self.alpha) *label_matrix
                similarity_mat_2 = (torch.matmul(x_q_2, x_k_1.T)/ self.alpha) *label_matrix
                similarity_mat_3 = (torch.matmul(x_q_1, x_k_1.T)/ self.alpha) *label_matrix
                similarity_mat_4 = (torch.matmul(x_q_2, x_k_2.T)/ self.alpha) *label_matrix
                new_label = torch.tensor(range(0,n)).to(self.device)
                view_loss_1 += self.criterion(similarity_mat_1, new_label)/n
                view_loss_2 += self.criterion(similarity_mat_2, new_label)/n
                view_loss_3 += self.criterion(similarity_mat_3, new_label)/n
                view_loss_4 += self.criterion(similarity_mat_4, new_label)/n
        #return view_loss_1
        #return view_loss_2
        #return (view_loss_1 + view_loss_2)/2
        return (view_loss_1 + view_loss_2 + self.gamma*view_loss_3 + self.gamma*view_loss_4)/4
        #return (self.gamma*view_loss_3 + self.gamma*view_loss_4)/2
        #return (view_loss_1 + view_loss_3 + view_loss_4)/3