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
import torch
import torch.nn as nn
import torch.nn.functional as F

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

class LsqWeight_v3(nn.Module):
    def __init__(self, bit, per_channel=False):       # bit: weight precision of each array
        super().__init__()
        self.wbit = bit
        self.thd_neg = -2**bit+1
        self.thd_pos = 2**bit-1
        self.per_channel = per_channel
        
        self.num_scales = bit
        
        # Initialize scale parameters
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.num_scales)
        ])

    def init_from(self, x, *args, **kwargs):
        base_scale = None
        
        if self.per_channel:
            base_scale = x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)       # LSQ paper v2
        else:
            base_scale = x.detach().abs().mean()                             # LSQ paper v2
        
        # Initialize each scale parameter (with decreasing values)
        for i in range(self.num_scales):
            scale_factor = 1.0 / (2 ** i)  # Scale reduction factor
            self.scales[i] = nn.Parameter(base_scale * scale_factor)

    def generate_quantization_coefficients(self, device):
        # Generate all permutations of coefficients using meshgrid
        coeff_values = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)
        meshgrids = torch.meshgrid(*[coeff_values for _ in range(self.num_scales)], indexing='ij')
        
        # Flatten and stack the grid points
        coefficients = torch.stack([grid.flatten() for grid in meshgrids], dim=1)

        return coefficients

    def round_pass(self, x, scaled_params):
        quantized = torch.zeros_like(x)
        quantized_soft = torch.zeros_like(x)
        
        # Generate quantization coefficients
        coefficients = self.generate_quantization_coefficients(x.device)
        num_levels = coefficients.shape[0]
        
        # Channel- or Column-wise quantization
        if self.per_channel:
            
            C = x.size(0)
            x_flat = x.view(C, -1)  # Shape: (C, N)
            N = x_flat.size(1)
            
            # Expand scale parameters
            scaled_params_expanded = [s.view(C, 1) for s in scaled_params]
            
            # Calculate quantization levels: (C, num_levels)
            quant_levels = torch.zeros((C, num_levels), device=x.device)
            
            for i in range(len(scaled_params_expanded)):
                if i < coefficients.shape[1]:
                    quant_levels += scaled_params_expanded[i] * coefficients[:, i].view(1, -1)
            
            x_exp = x_flat.unsqueeze(2)
            quant_levels_exp = quant_levels.unsqueeze(1).expand(-1, N, -1)
            
            # Calculate distances and find closest quantization level
            distances = torch.abs(x_exp - quant_levels_exp)
            indices = torch.argmin(distances, dim=2)
            
            # Get quantized values
            quantized = torch.gather(quant_levels_exp, 2, indices.unsqueeze(2)).squeeze(2)
            
            # Soft quantization (for gradients)
            alpha = 10 + self.wbit * 2  # Adjust softmax temperature
            weights = F.softmax(-alpha * distances, dim=2)
            quantized_soft = torch.sum(weights * quant_levels_exp, dim=2)
            
            # Reshape back to original dimensions
            quantized = quantized.view_as(x)
            quantized_soft = quantized_soft.view_as(x)
            
        else:
            # Calculate quantization levels
            quant_levels = torch.zeros(num_levels, device=x.device)
            
            for i in range(len(scaled_params)):
                if i < coefficients.shape[1]:
                    quant_levels += scaled_params[i] * coefficients[:, i]
            
            # Calculate distances and find closest quantization level
            distances = torch.abs(x.unsqueeze(-1) - quant_levels)
            indices = torch.argmin(distances, dim=-1)
            
            # Get quantized values
            quantized = quant_levels[indices]
            
            # Clamp to input range (optional)
            quantized = torch.clamp(quantized, x.detach().min(), x.detach().max())
            
            # Soft quantization (for gradients)
            alpha = 10 + self.wbit * 2
            weights = torch.softmax(-alpha * distances, dim=-1)
            quantized_soft = torch.sum(weights * quant_levels, dim=-1)
        
        y_grad = quantized_soft
        return (quantized - y_grad).detach() + y_grad

    def forward(self, x):
        # Calculate gradient scale: 1e-5 ~ 1e-1 recommended in LSQ
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5) / (2 ** (self.wbit - 2))
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5) / (2 ** (self.wbit - 2))
        
        scaled_params = [grad_scale(s, s_grad_scale) for s in self.scales]
        
        w_q = self.round_pass(x, scaled_params)
        
        return w_q


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
