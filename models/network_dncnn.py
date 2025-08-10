# -*- coding: utf-8 -*-
import torch.nn as nn
from models.util_converse import Converse_Block



"""
# --------------------------------------------
# implementation of Converse DnCNN
# --------------------------------------------
"""  
class Converse_DnCNN(nn.Module):

    def __init__(self, in_nc=1, kernel_size=7, dim=64, padding=6, padding_mode='circular', eps=1e-5, nb_blocks=20):
        super(Converse_DnCNN, self).__init__()
         
        self.dim = dim
        self.kernel_size = kernel_size  
        self.padding = padding  
        self.padding_mode = padding_mode
        self.eps = eps
        self.nb_blocks = nb_blocks

        self.m_head = [nn.Conv2d(in_nc, dim, 1, 1, 0)]
        self.m_body = [Converse_Block(dim, dim, kernel_size=self.kernel_size, scale=1, padding=padding, padding_mode=padding_mode, eps=1e-5) for i in range(20)]
        self.m_tail = [nn.Conv2d(dim, in_nc, 1, 1, 0)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_tail = nn.Sequential(*self.m_tail)  

    def forward(self, x):
        x = self.m_head(x)
        x = self.m_body(x)
        x = self.m_tail(x)
        return x

