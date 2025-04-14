import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.batchnorm import BatchNorm2d

import numpy as np
import math

from LSQ import *
from utils import split4d, im2col_weight, weightTile_HxW

class SplitConv4Pim_group(torch.nn.Module):
    def __init__(self, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, stride=1, padding=1, bias=False, isRow=True, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.num_splits = int(w_bit/split_bit)
        self.w_mode = w_mode

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.stride = stride
        self.padding = padding
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        if bias:
            self.bias = nn.Parameter(torch.randn(self.oc))
        else:
            self.bias = None

        self.isRow = isRow

        num_ic = min(math.floor(self.N / (self.kernel_size**2)), self.ic)
        self.groups = math.ceil(self.ic / num_ic)

        # Initialize the weights using He initialization
        self.weight = nn.Parameter(torch.randn(self.oc*self.groups, self.ic, self.kernel_size, self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        split_conv = []
        self.bit_shift = torch.empty(self.num_splits)
        for s in range(self.num_splits):

            if self.w_mode == 'Array':
                # v1: simpler version
                conv_module = Conv4Pim_group_arr(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)
                # v2: weight-decomposed version
                conv_module = Conv4Pim_group_arr_v2(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)
                # v3: v2 + LsqWeight v3
                conv_module = Conv4Pim_group_arr_v3(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)

            elif self.w_mode == 'Layer':
                # v1: simpler version
                conv_module = Conv4Pim_group_split(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)
                # v2: weight-decomposed version
                conv_module = Conv4Pim_group_split_v2(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)
                # v3: v2 + LsqWeight v3 assuming binary cells (w_bit: 2b or 3b available)
                conv_module = Conv4Pim_group_split_v3(self.w_bit, self.split_bit, s, self.ps_bit, self.num_sigma, self.psum_mode, self.ic, self.oc, self.kernel_size, self.N, self.groups, self.stride, self.padding, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt)

            conv_module.weight = self.weight
            conv_module.bias = self.bias
            conv_module.quant_ready()
            
            split_conv.append(conv_module)
            self.bit_shift[s] = (2**self.split_bit)**(self.num_splits-s-1)
        self.split_conv = nn.ModuleList(split_conv)

    def forward(self, input):
        output = 0
        for s in range(len(self.split_conv)):
            self.split_conv[s].psumOpt = self.psumOpt
            output += self.split_conv[s](input) * self.bit_shift[s]
        return output

''' v3: v2 + LsqWeight v3 assuming binary cells (w_bit: 2b or 3b available) '''
class Conv4Pim_group_split_v3(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(self.ic, self.N // (self.kernel_size**2))

        if self.w_bit==2:
            self.weight_quantize_fn = LsqWeight_2b(self.w_bit, self.w_per_ch)
        elif w_bit==3:
            self.weight_quantize_fn = LsqWeight_3b(self.w_bit, self.w_per_ch)

        self.w_q = 0

        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)


        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn_p = []
                ps_quan_fn_n = []
                for _ in range(self.groups):
                    ps_quan_fn_p.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                    ps_quan_fn_n.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn_p = nn.ModuleList(ps_quan_fn_p)
                self.ps_quan_fn_n = nn.ModuleList(ps_quan_fn_n)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn_p = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)
                self.psum_quantize_fn_n = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        self.weight_quantize_fn.init_from(self.weight)

    def forward(self, input):
        img_num, _, input_h, input_w = input.shape

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' 1. Quantize weight: Round to pre-determined combinations of scale factors '''
        self.w_q = self.weight_quantize_fn(self.weight)

        ''' POSITIVE WEIGHT '''

        ''' 2. Decompose: Get positive one '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w_split_p = torch.relu(self.w_q)

        ''' 3. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        output_p = 0

        if self.w_per_ch:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_p[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)

                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_p[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_p, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))
        else:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_p[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_p[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_p, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))

        self.arr_output = torch.stack(out_list, dim=0)

        ''' 4. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_p += self.ps_quan_fn_p[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_p += self.arr_output[i]
                output_p = self.psum_quantize_fn_p(output_p)
        else:
            for i in range(self.groups):
                output_p += self.arr_output[i]

        ''' NEGATIVE  WEIGHT '''

        ''' 2. Decompose: Get negative one '''
        self.w_split_n = torch.relu(-self.w_q)

        ''' 3. Group Conv '''
        output_n = 0

        if self.w_per_ch:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_n[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_n[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_n, stride=self.stride, bias=self.bias, padding=1, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))
        else:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_n[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_n[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_n, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))

        self.arr_output = torch.stack(out_list, dim=0)

        ''' 4. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_n += self.ps_quan_fn_n[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_n += self.arr_output[i]
                output_n = self.psum_quantize_fn_n(output_n)
        else:
            for i in range(self.groups):
                output_n += self.arr_output[i]

        return output_p - output_n


class Conv4Pim_group_arr_v3(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(self.ic, self.N // (self.kernel_size**2))

        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)
        self.w_split = 0

        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn_p = []
                ps_quan_fn_n = []
                for _ in range(self.groups):
                    ps_quan_fn_p.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                    ps_quan_fn_n.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn_p = nn.ModuleList(ps_quan_fn_p)
                self.ps_quan_fn_n = nn.ModuleList(ps_quan_fn_n)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn_p = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)
                self.psum_quantize_fn_n = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)
        self.num_arrays = self.weight_tiler.arrays

        # Array-wise weight quantizers
        w_quan_fn = []
        for i in range(self.num_arrays):
            if self.w_bit==2:
                w_quan_fn.append(LsqWeight_2b(self.w_bit, self.w_per_ch))
            elif self.w_bit==3:
                w_quan_fn.append(LsqWeight_3b(self.w_bit, self.w_per_ch))
        self.w_quan_fn = nn.ModuleList(w_quan_fn)

        self.row_slide = self.weight_tiler.row_slide
        self.col_slide = self.weight_tiler.col_slide

        num_oc = self.weight_tiler.num_oc

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic
        empty = self.weight_tiler.pd[3]

        # Initialize the quantizers
        for j in range(self.row_slide):
            if j==self.row_slide-1:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn[self.col_slide*j+k].init_from(temp_weight)
            else:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn[self.col_slide*j+k].init_from(temp_weight)

    def forward(self, input):
        img_num, _, input_h, input_w = input.shape
        self.weight_p = torch.relu(self.weight)
        self.weight_n = torch.relu(-self.weight)

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' POSITIVE WEIGHT '''
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight_p)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic

        col_slide = self.weight_tiler.col_slide
        row_slide = self.weight_tiler.row_slide
        num_oc = self.weight_tiler.num_oc
        empty = self.weight_tiler.pd[3]                                         # the number of empty rows

        if self.isRow:
            arr_oc = self.N
            if self.N >= self.oc:
                arr_oc = self.oc
        else:
            arr_oc = self.oc

        ''' 3. Weight Quantization '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Make a stack of weight
        w_list = [[] for _ in range(row_slide)]
        for j in range(row_slide):
            if j==row_slide-1:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q = self.w_quan_fn[col_slide*j+k](temp_weight)
                    temp_weight_q = w_q
                    w_list[j].append(temp_weight_q)
            else:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q = self.w_quan_fn[col_slide*j+k](temp_weight)
                    temp_weight_q = w_q
                    w_list[j].append(temp_weight_q)

            w_list[j] = torch.cat(w_list[j], dim=1)
        self.w_split_p = torch.stack(w_list, dim=0)                                 # quantized & splitted weight

        ''' 4. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output_p = 0

        out_list = [[] for _ in range(row_slide)]
        for i in range(row_slide):
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_p[i][:-arr_oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list[i] = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_p[i][-arr_oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list[i].append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_p[i], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list[i] = list(group_out.chunk(self.groups, dim=1))
            out_list[i] = torch.stack(out_list[i], dim=0)
        self.arr_output_p = torch.cat(out_list, dim=2)

        ''' 5. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_p += self.ps_quan_fn_p[i](self.arr_output_p[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_p += self.arr_output_p[i]
                output_p = self.psum_quantize_fn_p(output_p)
        else:
            for i in range(self.groups):
                output_p += self.arr_output_p[i]


        ''' NEGATIVE '''
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight_n)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic

        col_slide = self.weight_tiler.col_slide
        row_slide = self.weight_tiler.row_slide
        num_oc = self.weight_tiler.num_oc
        empty = self.weight_tiler.pd[3]                                         # the number of empty rows

        if self.isRow:
            arr_oc = self.N
            if self.N >= self.oc:
                arr_oc = self.oc
        else:
            arr_oc = self.oc

        ''' 3. Weight Quantization '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Make a stack of weight
        w_list = [[] for _ in range(row_slide)]
        for j in range(row_slide):
            if j==row_slide-1:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q = self.w_quan_fn[col_slide*j+k](temp_weight)
                    temp_weight_q = w_q
                    w_list[j].append(temp_weight_q)
            else:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q = self.w_quan_fn[col_slide*j+k](temp_weight)
                    temp_weight_q = w_q
                    w_list[j].append(temp_weight_q)

            w_list[j] = torch.cat(w_list[j], dim=1)
            self.w_split_n = torch.stack(w_list, dim=0)


        ''' 4. Group Conv '''
        output_n = 0

        out_list = [[] for _ in range(row_slide)]
        for i in range(row_slide):
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split_n[i][:-arr_oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list[i] = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split_n[i][-arr_oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list[i].append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split_n[i], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list[i] = list(group_out.chunk(self.groups, dim=1))
            out_list[i] = torch.stack(out_list[i], dim=0)
        self.arr_output_n = torch.cat(out_list, dim=2)

        ''' 5. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_n += self.ps_quan_fn_n[i](self.arr_output_n[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_n += self.arr_output_n[i]
                output_n = self.psum_quantize_fn_n(output_n)
        else:
            for i in range(self.groups):
                output_n += self.arr_output_n[i]

        return output_p - output_n


''' v2: weight decomposition considered '''
class Conv4Pim_group_split_v2(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(self.ic, self.N // (self.kernel_size**2))

        self.weight_quantize_fn_p = LsqWeight_v2(self.w_bit, self.w_per_ch)
        self.weight_quantize_fn_n = LsqWeight_v2(self.w_bit, self.w_per_ch)

        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn_p = []
                ps_quan_fn_n = []
                for _ in range(self.groups):
                    ps_quan_fn_p.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                    ps_quan_fn_n.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn_p = nn.ModuleList(ps_quan_fn_p)
                self.ps_quan_fn_n = nn.ModuleList(ps_quan_fn_n)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn_p = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)
                self.psum_quantize_fn_n = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        self.weight_quantize_fn_p.init_from(self.weight)
        self.weight_quantize_fn_n.init_from(self.weight)

    def forward(self, input):
        self.weight_p = torch.relu(self.weight)
        self.weight_n = torch.relu(-self.weight)

        img_num, _, input_h, input_w = input.shape

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' POSITIVE '''
        ''' 1. Weight Quantization '''
        self.w_q, self.w_q_int, self.scale_factor = self.weight_quantize_fn_p(self.weight_p)

        ''' 2. Split '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # simpler LsqWeight
        self.w_split = (self.splitter(self.w_q_int)[self.idx]).to(device)    # split weight tensor

        ''' 3. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        output_p = 0

        if self.w_per_ch:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))
        else:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))

        self.arr_output = torch.stack(out_list, dim=0)

        ''' 4. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_p += self.ps_quan_fn_p[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_p += self.arr_output[i]
                output_p = self.psum_quantize_fn_p(output_p)
        else:
            for i in range(self.groups):
                output_p += self.arr_output[i]

        ''' NEGATIVE '''
        ''' 1. Weight Quantization '''
        self.w_q, self.w_q_int, self.scale_factor = self.weight_quantize_fn_n(self.weight_n)

        ''' 2. Split '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # simpler LsqWeight
        self.w_split = (self.splitter(self.w_q_int)[self.idx]).to(device)    # split weight tensor

        ''' 3. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        output_n = 0

        if self.w_per_ch:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=1, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))
        else:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))

        self.arr_output = torch.stack(out_list, dim=0)

        ''' 4. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_n += self.ps_quan_fn_n[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_n += self.arr_output[i]
                output_n = self.psum_quantize_fn_n(output_n)
        else:
            for i in range(self.groups):
                output_n += self.arr_output[i]

        return output_p - output_n

class Conv4Pim_group_arr_v2(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(self.ic, self.N // (self.kernel_size**2))

        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn_p = []
                ps_quan_fn_n = []
                for _ in range(self.groups):
                    ps_quan_fn_p.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                    ps_quan_fn_n.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn_p = nn.ModuleList(ps_quan_fn_p)
                self.ps_quan_fn_n = nn.ModuleList(ps_quan_fn_n)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn_p = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)
                self.psum_quantize_fn_n = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)
        self.num_arrays = self.weight_tiler.arrays

        # Array-wise weight quantizers
        w_quan_fn_p = []
        w_quan_fn_n = []
        for i in range(self.num_arrays):
            w_quan_fn_p.append(LsqWeight_v2(self.w_bit, self.w_per_ch))
            w_quan_fn_n.append(LsqWeight_v2(self.w_bit, self.w_per_ch))
        self.w_quan_fn_p = nn.ModuleList(w_quan_fn_p)
        self.w_quan_fn_n = nn.ModuleList(w_quan_fn_n)

        # Initialize the quantizers
        self.row_slide = self.weight_tiler.row_slide
        self.col_slide = self.weight_tiler.col_slide

        num_oc = self.weight_tiler.num_oc

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic
        empty = self.weight_tiler.pd[3]

        for j in range(self.row_slide):
            if j==self.row_slide-1:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn_p[self.col_slide*j+k].init_from(temp_weight)
                    self.w_quan_fn_n[self.col_slide*j+k].init_from(temp_weight)
            else:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn_p[self.col_slide*j+k].init_from(temp_weight)
                    self.w_quan_fn_n[self.col_slide*j+k].init_from(temp_weight)


    def forward(self, input):
        img_num, _, input_h, input_w = input.shape
        self.weight_p = torch.relu(self.weight)
        self.weight_n = torch.relu(-self.weight)

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' POSITIVE '''
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(torch.relu(self.weight))

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic

        ''' 3. Weight Quantization '''
        col_slide = self.weight_tiler.col_slide
        row_slide = self.weight_tiler.row_slide
        num_oc = self.weight_tiler.num_oc
        empty = self.weight_tiler.pd[3]                                         # the number of empty rows

        if self.isRow:
            arr_oc = self.N
            if self.N >= self.oc:
                arr_oc = self.oc
        else:
            arr_oc = self.oc

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Make a stack of weight
        w_list = [[] for _ in range(row_slide)]
        move = self.splitter(torch.tensor([2**(self.w_bit-1)]))[0].item()
        for j in range(row_slide):
            if j==row_slide-1:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf= self.w_quan_fn_p[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            else:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf = self.w_quan_fn_p[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            w_list[j] = torch.cat(w_list[j], dim=1)
        self.w_split = torch.stack(w_list, dim=0)                                 # quantized & splitted weight

        ''' 4. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output_p = 0

        out_list = [[] for _ in range(row_slide)]
        for i in range(row_slide):
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[i][:-arr_oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list[i] = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[i][-arr_oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list[i].append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split[i], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list[i] = list(group_out.chunk(self.groups, dim=1))
            out_list[i] = torch.stack(out_list[i], dim=0)
        self.arr_output = torch.cat(out_list, dim=2)

        ''' 5. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_p += self.ps_quan_fn_p[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_p += self.arr_output[i]
                output_p = self.psum_quantize_fn_p(output_p)
        else:
            for i in range(self.groups):
                output_p += self.arr_output[i]

        ''' NEGATIVE '''
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(torch.relu(-self.weight))

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic

        ''' 3. Weight Quantization '''
        col_slide = self.weight_tiler.col_slide
        row_slide = self.weight_tiler.row_slide
        num_oc = self.weight_tiler.num_oc
        empty = self.weight_tiler.pd[3]                                         # the number of empty rows

        if self.isRow:
            arr_oc = self.N
            if self.N >= self.oc:
                arr_oc = self.oc
        else:
            arr_oc = self.oc

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Make a stack of weight
        w_list = [[] for _ in range(row_slide)]
        move = self.splitter(torch.tensor([2**(self.w_bit-1)]))[0].item()
        for j in range(row_slide):
            if j==row_slide-1:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf= self.w_quan_fn_n[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            else:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf = self.w_quan_fn_n[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            w_list[j] = torch.cat(w_list[j], dim=1)
        self.w_split = torch.stack(w_list, dim=0)                                 # quantized & splitted weight

        ''' 4. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output_n = 0

        out_list = [[] for _ in range(row_slide)]
        for i in range(row_slide):
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[i][:-arr_oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list[i] = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[i][-arr_oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list[i].append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split[i], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list[i] = list(group_out.chunk(self.groups, dim=1))
            out_list[i] = torch.stack(out_list[i], dim=0)
        self.arr_output = torch.cat(out_list, dim=2)

        ''' 5. Psum Quantization '''
        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output_n += self.ps_quan_fn_n[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output_n += self.arr_output[i]
                output_n = self.psum_quantize_fn_n(output_n)
        else:
            for i in range(self.groups):
                output_n += self.arr_output[i]

        return output_p - output_n

# Simpler version
class Conv4Pim_group_split(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(math.floor(self.N / (self.kernel_size**2)), self.ic)

        self.weight_quantize_fn = LsqWeight(self.w_bit, self.w_per_ch)
        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn = []
                for _ in range(self.groups):
                    ps_quan_fn.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn = nn.ModuleList(ps_quan_fn)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        self.weight_quantize_fn.init_from(self.weight)

    def forward(self, input):
        img_num, _, input_h, input_w = input.shape

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' 1. Quantization '''
        self.w_q, self.w_q_int, self.scale_factor = self.weight_quantize_fn(self.weight)

        ''' 2. Split '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.w_split = (self.splitter(self.w_q_int)[self.idx]).to(device)    # split weight tensor
        if self.idx==0:
            move = self.splitter(torch.tensor([2**(self.w_bit-1)]))[0].item()
            self.w_split = self.w_split - move


        ''' 5. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        output = 0

        if self.w_per_ch:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor[:-self.oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor[-self.oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))
        else:
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[:-self.oc, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[-self.oc:, ...] * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding)
                out_list.append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split * self.scale_factor, stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list = list(group_out.chunk(self.groups, dim=1))

        self.arr_output = torch.stack(out_list, dim=0)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output += self.ps_quan_fn[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output += self.arr_output[i]
                output = self.psum_quantize_fn(output)
        else:
            for i in range(self.groups):
                output += self.arr_output[i]

        return output

class Conv4Pim_group_arr(nn.Module):
    def __init__(self, w_bit, split_bit, idx, ps_bit, num_sigma, psum_mode, in_planes, planes, kernel_size, N, groups, stride, padding=1, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super().__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.idx = idx

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.ic = in_planes
        self.oc = planes
        self.kernel_size = kernel_size
        self.N = N
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.num_ic = min(math.floor(self.N / (self.kernel_size**2)), self.ic)

        self.weight_tiler = weightTile_HxW(self.N*self.groups, self.N, self.ic, self.oc*self.groups, self.kernel_size, self.isRow)
        self.splitter = split4d(self.w_bit, self.split_bit)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                ps_quan_fn = []
                for _ in range(self.groups):
                    ps_quan_fn.append(LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc))
                self.ps_quan_fn = nn.ModuleList(ps_quan_fn)
            elif self.psum_mode == 'Layer':
                self.psum_quantize_fn = LsqPsum(self.ps_bit, self.num_sigma, self.ps_per_ch, self.oc)

    def quant_ready(self):
        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)
        self.num_arrays = self.weight_tiler.arrays

        # Array-wise weight quantizers
        w_quan_fn = []
        for i in range(self.num_arrays):
            w_quan_fn.append(LsqWeight(self.w_bit, self.w_per_ch))
        self.w_quan_fn = nn.ModuleList(w_quan_fn)

        # Initialize the quantizers
        self.row_slide = self.weight_tiler.row_slide
        self.col_slide = self.weight_tiler.col_slide

        num_oc = self.weight_tiler.num_oc

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic
        empty = self.weight_tiler.pd[3]

        for j in range(self.row_slide):
            if j==self.row_slide-1:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn[self.col_slide*j+k].init_from(temp_weight)
            else:
                for k in range(self.col_slide):
                    if k==self.col_slide-1:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[self.col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    self.w_quan_fn[self.col_slide*j+k].init_from(temp_weight)


    def forward(self, input):
        img_num, _, input_h, input_w = input.shape

        # conv output shape
        h = math.floor((input_h-self.kernel_size+2*self.padding)/self.stride)+1
        w = math.floor((input_w-self.kernel_size+2*self.padding)/self.stride)+1

        ''' 1. Mapping '''
        self.kernel_stretched = im2col_weight(self.weight)

        ''' 2. Tiling '''
        self.im2col_weight = self.weight_tiler(self.kernel_stretched)

        if self.ic % self.num_ic:
            self.last_ic = self.ic % self.num_ic                                # number of input channels in the last array
        else:
            self.last_ic = self.num_ic

        ''' 3. Quantization '''
        col_slide = self.weight_tiler.col_slide
        row_slide = self.weight_tiler.row_slide
        num_oc = self.weight_tiler.num_oc
        empty = self.weight_tiler.pd[3]                                         # the number of empty rows

        if self.isRow:
            arr_oc = self.N
            if self.N >= self.oc:
                arr_oc = self.oc
        else:
            arr_oc = self.oc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make a stack of weight
        w_list = [[] for _ in range(row_slide)]
        move = self.splitter(torch.tensor([2**(self.w_bit-1)]))[0].item()
        for j in range(row_slide):
            if j==row_slide-1:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.last_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:num_oc-empty, :self.num_ic*(self.kernel_size**2)].reshape(num_oc-empty, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf= self.w_quan_fn[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    if self.idx==0:
                        w_split = w_split - move
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            else:
                for k in range(col_slide):
                    if k==col_slide-1:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.last_ic*(self.kernel_size**2)].reshape(num_oc, self.last_ic, self.kernel_size, self.kernel_size)
                    else:
                        temp_weight = self.im2col_weight[col_slide*j+k][:, :self.num_ic*(self.kernel_size**2)].reshape(num_oc, self.num_ic, self.kernel_size, self.kernel_size)
                    w_q, w_int, sf = self.w_quan_fn[col_slide*j+k](temp_weight)
                    w_split = (self.splitter(w_int)[self.idx]).to(device)
                    if self.idx==0:
                        w_split = w_split - move
                    temp_weight_q = w_split*sf
                    w_list[j].append(temp_weight_q)
            w_list[j] = torch.cat(w_list[j], dim=1)
        self.w_split = torch.stack(w_list, dim=0)                                 # quantized & splitted weight

        ''' 4. Group Conv '''
        img = torch.cat([input]*self.groups, dim=1)                             # expand input for group conv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = 0

        out_list = [[] for _ in range(row_slide)]
        for i in range(row_slide):
            if self.ic % self.num_ic:
                group_out = F.conv2d(img[:, :-self.ic, ...], self.w_split[i][:-arr_oc, ...], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups-1)
                out_list[i] = list(group_out.chunk(self.groups-1, dim=1))              # chunk out: tuple
                last_out = F.conv2d(img[:, -self.ic:, ...], self.w_split[i][-arr_oc:, ...], stride=self.stride, bias=self.bias, padding=self.padding)
                out_list[i].append(last_out)

            else:
                group_out = F.conv2d(img, self.w_split[i], stride=self.stride, bias=self.bias, padding=self.padding, groups=self.groups)
                out_list[i] = list(group_out.chunk(self.groups, dim=1))
            out_list[i] = torch.stack(out_list[i], dim=0)
        self.arr_output = torch.cat(out_list, dim=2)

        if self.psumOpt:
            if self.psum_mode == 'Array':
                for i in range(self.groups):
                    output += self.ps_quan_fn[i](self.arr_output[i])
            elif self.psum_mode == 'Layer':
                for i in range(self.groups):
                    output += self.arr_output[i]
                output = self.psum_quantize_fn(output)
        else:
            for i in range(self.groups):
                output += self.arr_output[i]

        return output
