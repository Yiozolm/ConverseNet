# -*- coding: utf-8 -*-
import torch.nn as nn
from models.util_converse import ConverseBlock


"""
# --------------------------------------------
# implementation of Converse DnCNN
# --------------------------------------------
"""  
class ConverseDnCNN(nn.Module):

    def __init__(self, in_channels=1, kernel_size=7, num_features=64, padding=6, padding_mode='circular', eps=1e-5, num_blocks=20):
        super(ConverseDnCNN, self).__init__()
         
        self.num_features = num_features
        self.kernel_size = kernel_size  
        self.padding = padding  
        self.padding_mode = padding_mode
        self.eps = eps
        self.num_blocks = num_blocks

        self.m_head = [nn.Conv2d(in_channels, num_features, 1, 1, 0)]
        self.m_body = [ConverseBlock(num_features, num_features, kernel_size=self.kernel_size, scale=1, padding=padding, padding_mode=padding_mode, eps=1e-5) for i in range(20)]
        self.m_tail = [nn.Conv2d(num_features, in_channels, 1, 1, 0)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_tail = nn.Sequential(*self.m_tail)  

    def forward(self, x):
        x = self.m_head(x)
        x = self.m_body(x)
        x = self.m_tail(x)
        return x

