"""Adapted from https://github.com/zhutmost/lsq-net/blob/master/quan/quantizer/lsq.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


# Simpler version
class LsqWeight(nn.Module):
    def __init__(self, bit, per_channel):
        super().__init__()
        self.wbit = bit
        self.thd_neg = -2**(bit-1)
        self.thd_pos = 2**(bit-1)-1
        self.per_channel = per_channel
        self.sf = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            # self.sf = nn.Parameter(
            #     x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))       # LSQ paper v1
            self.sf = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True))       # LSQ paper v2
        else:
            # self.sf = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)) # LSQ paper v1
            self.sf = nn.Parameter(x.detach().abs().mean())                             # LSQ paper v2

    def forward(self, x):
        # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)                        # LSQ paper v1
        s_grad_scale = 1e-4                                                             # 1e-5 ~ 1e-2 depending on granularity
        s_scale = grad_scale(self.sf, s_grad_scale)

        w_q_int = torch.clamp(round_pass(x / s_scale), self.thd_neg, self.thd_pos) + 2**(self.wbit-1)
        w_q = (w_q_int-2**(self.wbit-1)) * s_scale

        return w_q, w_q_int, s_scale


class LsqPsum(nn.Module):
    def __init__(self, bit, num_sigma, per_channel, oc):
        super().__init__()
        self.psbit = bit
        self.num_sigma = num_sigma
        self.thd_neg = -2**(bit-1)
        self.thd_pos = 2**(bit-1)-1
        self.per_channel = per_channel
        self.oc = oc

        num_levels = 2**bit-1

        # 1/num_levels ~ 1
        if self.per_channel:
            self.sf = nn.Parameter(torch.ones(1, self.oc, 1, 1))
        else:
            self.sf = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)                        # LSQ paper v1
        s_grad_scale = 1e-1                                                             # 1e-2 ~ 1 depending on granularity
        s_scale = grad_scale(self.sf, s_grad_scale)
        
        if self.per_channel:
            mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            sigma = torch.std(x, dim=(0, 2, 3), keepdim=True)
        else:
            mu = torch.mean(x)
            sigma = torch.std(x)
        x_clip = torch.clamp(x, mu - self.num_sigma * sigma, mu + self.num_sigma * sigma)
        x_q_int = round_pass(x_clip/s_scale)   # clip by num_sigma

        center = torch.round(torch.mean(x_q_int))

        if self.psbit == 1.5: # symmetric quantizer
            # center = center + 0.5
            x_q_int = torch.where(x_q_int > center+0.5, (center+1).clone().detach(),
                      torch.where(x_q_int < center-0.5, (center-1).clone().detach(),
                                  torch.tensor(center, device=x.device)))
        else:
            thd_neg = center - 2**(self.psbit-1)
            thd_pos = center + 2**(self.psbit-1)-1
            x_q_int = torch.clamp(x_q_int, thd_neg, thd_pos)

        x_q = x_q_int * s_scale
        return x_q


# # return integer output
# class LsqPsum_int(nn.Module):
#     def __init__(self, bit, num_sigma, per_channel, oc):
#         super().__init__()
#         self.psbit = bit
#         self.num_sigma = num_sigma
#         self.thd_neg = -2**(bit-1)                                  # [-4, +3]
#         self.thd_pos = 2**(bit-1)-1
#         self.per_channel = per_channel                              # whether col-wise or not
#         self.oc = oc

#         num_levels = 2**bit-1
#         if self.per_channel:
#             self.sf = nn.Parameter(torch.ones(1, self.oc, 1, 1)*100/num_levels)
#         else:
#             self.sf = nn.Parameter(torch.tensor([100/num_levels]))

#     def forward(self, x):
#         # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)                        # LSQ paper v1
#         s_grad_scale = 1e-1                                                             # LSQ paper v2: activation 1e-1
#         s_scale = grad_scale(self.sf, s_grad_scale)
#         if self.per_channel:
#             mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)
#             sigma = torch.std(x, dim=(0, 2, 3), keepdim=True)
#         else:
#             mu = torch.mean(x)
#             sigma = torch.std(x)

#         x_clip = torch.clamp(x, mu - self.num_sigma * sigma, mu + self.num_sigma * sigma)   # clip by num_sigma

#         # x_q_int = round_pass(x_clip/s_scale)
#         x_q_int = torch.clamp(round_pass(x_clip / s_scale), self.thd_neg, self.thd_pos)
#         x_q = x_q_int * s_scale

#         return x_q_int
