# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np

from lib.any_quant import qfn, activation_quantize_fn, weight_quantize_fn, psum_quantize_fn, Activate, Linear_Q#, BatchNorm2d_Q
from lib.LSQ import LsqWeight, LsqPsum
from lib.utils import split4d, im2col_weight, im2col_acti, weightTile, weightTile_new, weightTile_HxW
from lib.SplitConv4Pim_group import SplitConv4Pim_group

class BasicBlock_arr(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    expansion = 1
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, in_planes, planes, arr_size, stride=1, down_sample=None, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super(BasicBlock_arr, self).__init__()
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.ps_bit = ps_bit
        self.num_sigma = num_sigma

        # Conv2d = conv2d_quantize_fn(self.bit_list)
        # NormLayer = batchnorm_fn(self.bit_list)

        # self.bn0 =BatchNorm2d_Q(self.a_bit, self.w_bit, in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = Activate(self.a_bit)
        self.conv1 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, in_planes, planes, kernel_size=3, N=arr_size,
                              stride=stride, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        # self.bn2 = BatchNorm2d_Q(self.a_bit, self.w_bit, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = Activate(self.a_bit)
        self.conv2 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, planes, planes, kernel_size=3, N=arr_size,
                              stride=1, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)

        self.dropout = nn.Dropout(0.3)

        self.down_sample = down_sample
        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn1(x)
        out = self.act1(out)

        out = self.dropout(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out += shortcut
        return out


class ResNet20_arr(nn.Module):
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, block, arr_size, num_units=[3, 3, 3], num_classes=10, isRow=True, w_per_ch=False, ps_per_ch=False, psumOpt=False, expand=5):

    # def __init__(self, block, num_units, bit_list, num_classes, expand=5):
        super(ResNet20_arr, self).__init__()
        self.in_planes = 16 # Resnet

        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.w_mode = w_mode

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.expand = expand

        # NormLayer = batchnorm_fn(self.bit_list)

        ep = self.expand
        self.conv0 = nn.Conv2d(3, 16 * ep, kernel_size=3, stride=1, padding=1, bias=False)

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            self.layers.append(block(self.a_bit, self.w_bit, self.split_bit, self.w_mode, self.ps_bit, self.num_sigma, self.psum_mode, in_planes, channel, arr_size, stride, isRow=self.isRow, w_per_ch=self.w_per_ch, ps_per_ch=self.ps_per_ch, psumOpt=self.psumOpt))
            in_planes = channel

        # self.bn =  BatchNorm2d_Q(self.a_bit, self.w_bit, 64*ep)
        self.bn = nn.BatchNorm2d(64*ep)
        self.fc = nn.Linear(64 * ep, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out)
        return out



class PreActBottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, in_planes, planes, arr_size, stride=1, down_sample=None, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=False):
        super(PreActBottleneckQ, self).__init__()
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.ps_bit = ps_bit
        self.num_sigma = num_sigma

        # Conv2d = conv2d_quantize_fn(self.bit_list)
        # norm_layer = batchnorm_fn(self.bit_list)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = Activate(self.a_bit)
        self.conv1 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, in_planes, planes, kernel_size=1, N=arr_size,
                              stride=1, padding=0, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = Activate(self.a_bit)
        self.conv2 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, planes, planes, kernel_size=3, N=arr_size,
                              stride=stride, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        self.bn3 = nn.BatchNorm2d(planes)
        self.act3 = Activate(self.a_bit)
        self.conv3 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, planes, planes*self.expansion, kernel_size=1, N=arr_size,
                              stride=1, padding=0, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        self.down_sample = down_sample

    def forward(self, x):        
        shortcut = self.down_sample(x) if self.down_sample is not None else x
        print(shortcut.shape)
        out = self.conv0(self.act0(self.bn0(x)))
        print(out.shape)
        out = self.conv1(self.act1(self.bn1(out)))
        print(out.shape)
        out = self.conv2(self.act2(self.bn2(out)))
        print(out.shape)
        out += shortcut
        return out


class BasicBlock_arr_v2(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    expansion = 1
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, in_planes, planes, arr_size, stride=1, down_sample=None, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super(BasicBlock_arr_v2, self).__init__()
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.ps_bit = ps_bit
        self.num_sigma = num_sigma

        # Conv2d = conv2d_quantize_fn(self.bit_list)
        # NormLayer = batchnorm_fn(self.bit_list)

        self.conv1 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, in_planes, planes, kernel_size=3, N=arr_size,
                              stride=stride, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        # self.bn0 =BatchNorm2d_Q(self.a_bit, self.w_bit, in_planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = Activate(self.a_bit)
        
        # self.bn2 = BatchNorm2d_Q(self.a_bit, self.w_bit, planes)
        self.conv2 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, planes, planes, kernel_size=3, N=arr_size,
                              stride=1, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.act2 = Activate(self.a_bit)
        

        # dropout: p=0.2
        self.dropout = nn.Dropout(0.2)

        self.down_sample = down_sample
        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.act2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        return out

class ResNetBottleneck_arr(nn.Module):
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, block, arr_size, num_units=[3, 4, 6, 3], num_classes=200, isRow=True, w_per_ch=False, ps_per_ch=False, psumOpt=False):
        super(ResNetBottleneck_arr, self).__init__()
        # self.bit_list = bit_list
        # self.wbit = self.bit_list[-1]
        # self.abit = self.bit_list[-1]

        # self.norm_layer = batchnorm_fn(self.bit_list)

        self.in_planes = 64 # Resnet

        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.w_mode = w_mode

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.N = arr_size

        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = Activate(self.a_bit)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_units[0])
        self.layer2 = self._make_layer(block, 128, num_units[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_units[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_units[3], stride=2)

        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        # self.act = Activate(self.a_bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.a_bit, self.w_bit, self.split_bit, self.w_mode, self.ps_bit, self.num_sigma, self.psum_mode, self.in_planes, planes, self.N,
                            stride, down_sample, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.a_bit, self.w_bit, self.split_bit, self.w_mode, self.ps_bit, self.num_sigma, self.psum_mode, self.in_planes, planes, self.N, isRow=self.isRow, w_per_ch=self.w_per_ch, ps_per_ch=self.ps_per_ch, psumOpt=self.psumOpt))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class BasicBlock_FP(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    expansion = 1
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, in_planes, planes, arr_size, stride=1, down_sample=None, isRow=False, w_per_ch=False, ps_per_ch=False, psumOpt=True):
        super(BasicBlock_FP, self).__init__()
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.ps_bit = ps_bit
        self.num_sigma = num_sigma

        # Conv2d = conv2d_quantize_fn(self.bit_list)
        # NormLayer = batchnorm_fn(self.bit_list)

        # self.conv1 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, in_planes, planes, kernel_size=3, N=arr_size,
        #                       stride=stride, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.act1 = Activate(self.a_bit)
        self.act1 = nn.ReLU()
        
        # self.conv2 = SplitConv4Pim_group(self.w_bit, self.split_bit, w_mode, self.ps_bit, self.num_sigma, psum_mode, planes, planes, kernel_size=3, N=arr_size,
        #                       stride=1, padding=1, bias=False, isRow=isRow, w_per_ch=w_per_ch, ps_per_ch=ps_per_ch, psumOpt=psumOpt)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)        

        # dropout: p=0.2
        self.dropout = nn.Dropout(0.2)

        self.down_sample = down_sample
        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.act2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        return out

class ResNetBottleneck_FP(nn.Module):
    def __init__(self, a_bit, w_bit, split_bit, w_mode, ps_bit, num_sigma, psum_mode, block, arr_size, num_units=[3, 4, 6, 3], num_classes=200, isRow=True, w_per_ch=False, ps_per_ch=False, psumOpt=False):
        super(ResNetBottleneck_FP, self).__init__()
        # self.bit_list = bit_list
        # self.wbit = self.bit_list[-1]
        # self.abit = self.bit_list[-1]

        # self.norm_layer = batchnorm_fn(self.bit_list)

        self.in_planes = 64 # Resnet

        self.a_bit = a_bit
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.w_mode = w_mode

        self.ps_bit = ps_bit
        self.num_sigma = num_sigma
        self.psum_mode = psum_mode

        self.N = arr_size

        self.isRow = isRow
        self.w_per_ch = w_per_ch
        self.ps_per_ch = ps_per_ch
        self.psumOpt = psumOpt

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_units[0])
        self.layer2 = self._make_layer(block, 128, num_units[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_units[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_units[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        # self.act = Activate(self.a_bit)
        self.act = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.a_bit, self.w_bit, self.split_bit, self.w_mode, self.ps_bit, self.num_sigma, self.psum_mode, self.in_planes, planes, self.N,
                            stride, down_sample, self.isRow, self.w_per_ch, self.ps_per_ch, self.psumOpt))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.a_bit, self.w_bit, self.split_bit, self.w_mode, self.ps_bit, self.num_sigma, self.psum_mode, self.in_planes, planes, self.N, isRow=self.isRow, w_per_ch=self.w_per_ch, ps_per_ch=self.ps_per_ch, psumOpt=self.psumOpt))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



'''
# For CIFAR10
def resnet20q(bit_list, num_classes=10):
    return PreActResNet(PreActBasicBlockQ, [3, 3, 3], bit_list, num_classes=num_classes)


# For ImageNet
def resnet50q(bit_list, num_classes=1000):
    return PreActResNetBottleneck(PreActBottleneckQ, [3, 4, 6, 3], bit_list, num_classes=num_classes)
'''
