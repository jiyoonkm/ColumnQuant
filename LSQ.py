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


class LsqPsum(nn.Module):
    def __init__(self, bit, num_sigma, per_channel, oc):
        super().__init__()
        self.psbit = bit
        self.num_sigma = num_sigma
        self.thd_neg = 0                                  # [-4, +3]
        self.thd_pos = 2**(bit)-1
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
        s_grad_scale = 1e-2                                                             # 1e-2 ~ 1 depending on granularity
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
            thd_neg = center - 1
            thd_pos = center + 1
            x_q_int = torch.clamp(x_q_int, thd_neg, thd_pos)
            x_q = x_q_int * s_scale
        else:
            if self.psbit==1:
                thd_neg = center
                thd_pos = center + 1
            else:
                thd_neg = center - 2**(self.psbit-1)
                thd_pos = center + 2**(self.psbit-1) - 1
            x_q_int = torch.clamp(x_q_int, thd_neg, thd_pos)
            x_q = x_q_int * s_scale

        x_q = x_q_int * s_scale
        return x_q

''' for conv layer v3's: round to pre-determined combination of scale factors '''
class LsqWeight_2b(nn.Module):
    def __init__(self, bit, per_channel):       # bit: weight precision of each array
        super().__init__()
        self.wbit = bit
        self.thd_neg = -2**bit+1
        self.thd_pos = 2**bit-1
        self.per_channel = per_channel
        self.sH = nn.Parameter(torch.ones(1))
        self.sL = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.sH = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True))       # LSQ paper v2
            self.sL = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)/2)
        else:
            self.sH = nn.Parameter(x.detach().abs().mean())                             # LSQ paper v2
            self.sL = nn.Parameter(x.detach().abs().mean()/2)                             # LSQ paper v2

    def round_pass(self, x, sH, sL):
        quantized = torch.zeros_like(x)
        quantized_soft = torch.zeros_like(x)

        if self.per_channel:
            C = x.size(0)
            x_flat = x.view(C, -1)  # Shape: (C, N)

            # Number of elements per channel
            N = x_flat.size(1)

            sH_exp = sH.view(C, 1)  # Shape: (C, 1)
            sL_exp = sL.view(C, 1)  # Shape: (C, 1)

            # Define quantization coefficients for 3-bit quantization (9 levels)
            coefficients = torch.tensor([
                [-1, -1],  # -sH - sL
                [-1,  0],  # -sH
                [-1,  1],  # -sH + sL
                [ 0, -1],  # -sL
                [ 0,  0],  # 0
                [ 0,  1],  # sL
                [ 1, -1],  # sH - sL
                [ 1,  0],  # sH
                [ 1,  1],  # sH + sL
            ], dtype=torch.float32, device=x.device)  # Shape: (9, 2)

            # Compute quantization levels: (C, 9)
            quant_levels = sH_exp * coefficients[:, 0].view(1, 9) + sL_exp * coefficients[:, 1].view(1, 9)  # Shape: (C, 9)

            x_exp = x_flat.unsqueeze(2)  # Shape: (C, N, 1)

            quant_levels_exp = quant_levels.unsqueeze(1)  # Shape: (C, 1, 9)
            quant_levels_exp = quant_levels_exp.expand(-1, N, -1)  # Shape: (C, N, 9)

            distances = torch.abs(x_exp - quant_levels_exp)  # Shape: (C, N, 9)
            indices = torch.argmin(distances, dim=2)  # Shape: (C, N)

            quantized = torch.gather(quant_levels_exp, 2, indices.unsqueeze(2)).squeeze(2)  # Shape: (C, N)

            # Soft quantization
            alpha=10
            weights = F.softmax(-alpha*distances, dim=2)  # Shape: (C, N, 9)
            quantized_soft = torch.sum(weights * quant_levels_exp, dim=2)  # Shape: (C, N)

            # Reshape back to original dimensions
            quantized = quantized.view_as(x)
            quantized_soft = quantized_soft.view_as(x)

        else:
            quantization_levels = torch.stack([-(sH + sL), -sH, -(sH - sL), -sL, torch.tensor(0.0, device=x.device),
                                    sL, (sH - sL), sH, (sH + sL)])

            # Compute distances and find the closest quantization level
            distances = torch.abs(x.unsqueeze(-1) - quantization_levels)

            indices = torch.argmin(distances, dim=-1)

            # Use the indices to map to the quantization levels
            quantized = quantization_levels[indices]

            alpha=10
            weights = torch.softmax(-alpha*distances, dim=-1)
            quantized_soft = torch.sum(weights * quantization_levels, dim=-1)

        y_grad = quantized_soft
        # y_grad = x

        return (quantized - y_grad).detach() + y_grad

    def forward(self, x):
        # s_grad_scale = 1e-1                                                             # 1e-5 ~ 1e-2 depending on granularity
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

        sH_scale = grad_scale(self.sH, s_grad_scale)
        sL_scale = grad_scale(self.sL, s_grad_scale)

        w_q = self.round_pass(x, sH_scale, sL_scale)

        return w_q, sH_scale, sL_scale

class LsqWeight_3b(nn.Module):
    def __init__(self, bit, per_channel):       # bit: weight precision of each array
        super().__init__()
        self.wbit = bit
        self.thd_neg = -2**bit+1
        self.thd_pos = 2**bit-1
        self.per_channel = per_channel
        self.sH = nn.Parameter(torch.ones(1))
        self.sM = nn.Parameter(torch.ones(1))
        self.sL = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.sH = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True))       # LSQ paper v2
            self.sM = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)/2)       # LSQ paper v2
            self.sL = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)/4)
        else:
            self.sH = nn.Parameter(x.detach().abs().mean()*4/3)                             # LSQ paper v2
            self.sM = nn.Parameter(x.detach().abs().mean()*2/3)                             # LSQ paper v2
            self.sL = nn.Parameter(x.detach().abs().mean()/3)                             # LSQ paper v2


    def round_pass(self, x, sH, sM, sL):
        quantized = torch.zeros_like(x)
        quantized_soft = torch.zeros_like(x)

        # Set quantization levels
        if self.per_channel:
            C = x.size(0)
            x_flat = x.view(C, -1)  # Shape: (C, N)

            # Number of elements per channel
            N = x_flat.size(1)

            sH_exp = sH.view(C, 1)  # Shape: (C, 1)
            sM_exp = sM.view(C, 1)  # Shape: (C, 1)
            sL_exp = sL.view(C, 1)  # Shape: (C, 1)

            # Define all possible combinations of sH, sM, sL with coefficients -1, 0, 1
            coeff_H = torch.tensor([-1, 0, 1], dtype=torch.float32, device=x.device)
            coeff_M = torch.tensor([-1, 0, 1], dtype=torch.float32, device=x.device)
            coeff_L = torch.tensor([-1, 0, 1], dtype=torch.float32, device=x.device)


            # Create meshgrid
            coeff_H_grid, coeff_M_grid, coeff_L_grid = torch.meshgrid(coeff_H, coeff_M, coeff_L, indexing='ij')
            # Flatten and stack to get (26, 3)
            coefficients = torch.stack([coeff_H_grid.flatten(),
                                        coeff_M_grid.flatten(),
                                        coeff_L_grid.flatten()], dim=1)  # Shape: (27, 3)

            # Compute quantization levels: (C, 27)
            quant_levels = (sH_exp * coefficients[:, 0].view(1, 27) +
                            sM_exp * coefficients[:, 1].view(1, 27) +
                            sL_exp * coefficients[:, 2].view(1, 27))  # Shape: (C, 26)

            x_exp = x_flat.unsqueeze(2)  # Shape: (C, N, 1)

            quant_levels_exp = quant_levels.unsqueeze(1)  # Shape: (C, 1, 26)
            quant_levels_exp = quant_levels_exp.expand(-1, N, -1)  # Shape: (C, N, 26)

            distances = torch.abs(x_exp - quant_levels_exp)  # Shape: (C, N, 26)
            indices = torch.argmin(distances, dim=2)  # Shape: (C, N)

            quantized = torch.gather(quant_levels_exp, 2, indices.unsqueeze(2)).squeeze(2)  # Shape: (C, N)

            # Soft quantization
            alpha = 20
            weights = F.softmax(-alpha*distances, dim=2)  # Shape: (C, N, 26)
            quantized_soft = torch.sum(weights * quant_levels_exp, dim=2)  # Shape: (C, N)

            # Reshape back to original dimensions
            quantized = quantized.view_as(x)
            quantized_soft = quantized_soft.view_as(x)

        else:
            quantization_levels = torch.stack([-(sH+sM+sL), -(sH+sM), -(sH-sM+sL), -(sH+sM-sL), -(sH+sL), -sH, -(sH-sM), -(sH-sM-sL),
                                               -(sM+sL), -(sH-sL), -sM, -(sM-sL), -sL, torch.tensor(0.0, device=x.device),
                                               (sH+sM+sL), (sH+sM), (sH-sM+sL), (sH+sM-sL), (sH+sL), sH, (sH-sM), (sH-sM-sL),
                                               (sM+sL), (sH-sL), sM, (sM-sL), sL])

            # Compute distances and find the closest quantization level
            distances = torch.abs(x.unsqueeze(-1) - quantization_levels)

            indices = torch.argmin(distances, dim=-1)

            # Use the indices to map to the quantization levels
            quantized = quantization_levels[indices]
            quantized = torch.clamp(quantized, x.detach().min(), x.detach().max())

            alpha=10
            weights = torch.softmax(-alpha*distances, dim=-1)
            quantized_soft = torch.sum(weights * quantization_levels, dim=-1)

        y_grad = quantized_soft

        return (quantized - y_grad).detach() + y_grad

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5) / 10
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

        sH_scale = grad_scale(self.sH, s_grad_scale)
        sM_scale = grad_scale(self.sM, s_grad_scale)
        sL_scale = grad_scale(self.sL, s_grad_scale)

        w_q = self.round_pass(x, sH_scale, sM_scale, sL_scale)

        return w_q, sH_scale, sL_scale


''' for conv layer v2's: absolte-valued weight '''
class LsqWeight_v2(nn.Module):
    def __init__(self, bit, per_channel):
        super().__init__()
        self.wbit = bit
        self.thd_neg = 0
        self.thd_pos = 2**bit-1
        self.per_channel = per_channel                              # whether col-wise or not
        self.sf = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.sf = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True))       # LSQ paper v2
        else:
            self.sf = nn.Parameter(x.detach().abs().mean())                             # LSQ paper v2

    def forward(self, x):
        s_grad_scale = 1e-4                                                             # LSQ paper v2: 1e-2 ~ 1e-4 / tried 1e-5
        s_scale = grad_scale(self.sf, s_grad_scale)

        w_q_int = torch.clamp(round_pass(x / s_scale), self.thd_neg, self.thd_pos)
        w_q = w_q_int * s_scale

        return w_q, w_q_int, s_scale

''' for conv layer v1's '''
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
