# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

# @author ZhangMinghui,Southeast University
#
# Module Function: Provide non-linear-activation functions
# which have not be included in torch.nn module

"""
Module includes:
        __all__ = ['Swish',
                   'Mish']
"""


##############################################################################
# Different Activation Function
##############################################################################
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self,x):
        x = x * torch.sigmoid(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        x = x * torch.tanh(F.softplus(x))
        return x