import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

from matplotlib import pyplot as plt
import numpy as np
import math


""" Quantize ReLU """

class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        self.acti = nn.ReLU()
        self.quantize = quantize
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)                                            # relu
        if self.quantize:
            x = qfn.apply(x, self.abit)
        return x


class qfn(torch.autograd.Function):
    @staticmethod                                                                   # decorator: staticmethod ???
    def forward(ctx, input, k):                                                     # ctx ??
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


""" Quantize activation """
class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.abit = a_bit
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            num_levels = 2**self.abit - 1
            S = 1 / num_levels
            activation_q = qfn.apply(x, self.abit)
            activation_q_int = activation_q * num_levels
        return activation_q, activation_q_int, S



""" Quantize weight"""
class weight_quantize_fn(nn.Module):
    def __init__(self, bit_list):                                                   # bit_list: from any-precision
        super(weight_quantize_fn, self).__init__()
        # self.bit_list = bit_list
        # self.wbit = self.bit_list
        self.wbit = bit_list
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:                                                         # 32-bit: w_q_int, S 수정
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            max_val = torch.max(torch.abs(torch.tanh(x))).detach()                  # expectation -> max_val 수정
            num_levels = 2**self.wbit - 1
            S = max_val / float(num_levels)

            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5                # w': normalize & transform

            weight_q_int = qfn.apply(weight, self.wbit)                             # w_Q': [0, 1]

            weight_q = 2 * weight_q_int - 1                                         # w_Q: [-1, +1]
            weight_q = weight_q * max_val

        return weight_q, weight_q_int*num_levels, S, num_levels

# ver.2: Channel-wise available
class weight_quantize_fn_ver2(nn.Module):
    def __init__(self, bit, per_channel):
        super().__init__()
        self.wbit = bit
        self.per_channel = per_channel

    def forward(self, x):
        if self.wbit == 32:
            if self.per_channel:
                E = torch.mean(torch.abs(x), dim=list(range(1, x.dim()))).detach()
                w_q = torch.zeros_like(x)
                for i in range(x.shape[0]):
                    weight = torch.tanh(x[i])
                    weight = weight / torch.max(torch.abs(weight))
                    w_q[i] = weight * E[i]
            else:
                E = torch.mean(torch.abs(x)).detach()
                weight = torch.tanh(x)
                weight = weight / torch.max(torch.abs(weight))
                w_q = weight * E   
        else:
            if self.per_channel:
                max_val = torch.abs(torch.tanh(x)).detach()
                for i in range(1, x.dim()):
                    max_val = torch.max(max_val, dim=i, keepdim=True).values
                num_levels = 2**self.wbit - 1
                self.max_val = max_val

                w_q = torch.zeros_like(x)
                w_q_int = torch.zeros_like(x)
                for i in range(x.shape[0]):
                    m = max_val[i]
                    weight = torch.tanh(x[i])
                    weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5        # w'

                    w_q_int[i] = qfn.apply(weight, self.wbit)                       # w_Q': [0, 1], interval=1/256
                    w_q[i] = (2*w_q_int[i] - 1) * m                                 # w_Q: [-1, +1], interval=1/128
            else:
                max_val = torch.max(torch.abs(torch.tanh(x))).detach()
                self.max_val = max_val
                num_levels = 2**self.wbit - 1

                weight = torch.tanh(x)
                weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5            # w'               

                w_q_int = qfn.apply(weight, self.wbit)                              # w_Q': [0, 1], interval=1/256

                w_q = 2 * w_q_int - 1                                               # w_Q: [-1, +1], interval=1/128
                w_q = w_q * max_val
            
            S = max_val / float(num_levels)

        return w_q, w_q_int*num_levels, S, num_levels

""" Quantize Partial Sum """
class psum_quantize_fn(nn.Module):
    def __init__(self, bit_list, num_sigma=6, per_channel=False):
        super(psum_quantize_fn, self).__init__()
        self.pbit = bit_list
        assert self.pbit <= 8 or self.pbit == 32
        self.num_sigma = num_sigma                                                  # how many sigma in the range
        self.per_channel = per_channel

    def forward(self, x):
        if self.pbit == 32:                                                         # 32-bit: w_q_int, S 수정
            psum_q = x
        else:
            num_levels = 2**self.pbit - 1
            if self.per_channel:
                mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)
                sigma = torch.std(x, dim=(0, 2, 3), keepdim=True)
            else:
                mu = torch.mean(x)
                sigma = torch.std(x)

            self.mu = mu
            self.sigma = sigma

            self.thd_neg = mu - self.num_sigma * sigma
            self.thd_pos = mu + self.num_sigma * sigma

            x = torch.clamp(x, min=self.thd_neg, max=self.thd_pos)
            self.clip = x
            psum_q = qfn.apply(x, self.pbit)
        return psum_q

""" Switchable batch normalization """
class BatchNorm2d_Q(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, a_bit, w_bit, num_features):                                 # a_bit != w_bit 수정
        super(BatchNorm2d_Q, self).__init__()
        self.abit = a_bit
        self.wbit = w_bit
        self.bn_dict = nn.ModuleDict()
        self.bn_dict[str(a_bit)] = nn.BatchNorm2d(num_features, eps=1e-4)           # w_bit -> a_bit  수정
        """
        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')
        """
        
    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x


""" Quantized linear """
class Linear_Q(nn.Linear):
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias=bias)
        self.w_bit = w_bit
        self.w_quantize_fn = weight_quantize_fn(self.w_bit)                         # weight qnt function
        self.b_quantize_fn = weight_quantize_fn(self.w_bit)                         # bias qnt function
        
        self.w_q, self.w_q_int, self.scaling_factor, _ = self.w_quantize_fn(self.weight)
        self.b_q, self.b_q_int, _, _ = self.b_quantize_fn(self.weight)

    def forward(self, input, order=None):
        self.w_q, self.w_q_int, self.scaling_factor, _ = self.w_quantize_fn(self.weight)
        self.b_q, self.b_q_int, _, _ = self.b_quantize_fn(self.bias)

        return F.linear(input, self.w_q, self.b_q)
