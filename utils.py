import torch
import math

# Bit split
class split4d(nn.Module):
    def __init__(self, w_bit, split_bit):
        super(split4d, self).__init__()
        self.w_bit = w_bit
        self.split_bit = split_bit
        self.num_splits = int(w_bit/split_bit)                                # e.g. 4bits weight -> 2 splits * 2bits

    def _floor_div(self, input, K):
        rem = torch.remainder(input, K)
        out = (input - rem) / K
        return out

    def forward(self, input):
        # create an empty arr for output: num_splits x original shape
        outshape = (self.num_splits,) + tuple(input.shape)                  # output shape: num_splits x shape
        output = torch.zeros(outshape)

        divisor = 2**self.split_bit
        count = 0
        for i in range(self.num_splits-1, 0, -1):
            """
                output idx  0      ---------->  num_splits
                split       higher ---------->  lower
            """
            output[i] = torch.remainder(input, divisor)
            input = self._floor_div(input, divisor)
        output[0] = input
        return output

# Im2col weight mapping
def im2col_weight(kernel):
    output_channel = kernel.shape[0]
    mapping = kernel.view(output_channel, -1)
    return mapping

# CIM array tiling
class weightTile_HxW():
    def __init__(self, h, w, ic, oc, ker, isRow):                                        # h x w array
        self.h = h
        self.w = w
        self.input_w = ic*(ker**2)
        self.input_h = oc                                                           # expanded number of OCs
        self.isRow = isRow                                                          # determine whether tile along OC direction

        self.arrays = 0                                                             # initialize the number of arrays
        self.col_slide = 0
        self.row_slide = 0
        
        if self.isRow:
            self.num_oc = self.h
            if self.h >= self.input_h:
                self.num_oc = self.input_h
                self.h = self.input_h
        else:
            self.num_oc = self.input_h
            self.h = self.input_h
        self.step_w = self.w - self.w % (ker**2)                                           # how many IC elements are tiled at once

        # padding
        pd_w = 0
        pd_h = 0
        self.num_step = math.ceil((self.input_w-self.w)/self.step_w)
        w_need = self.w + self.step_w * self.num_step
        if w_need > self.input_w:
            pd_w = w_need - self.input_w
        if self.isRow:
            if self.input_h % self.h:
                pd_h = self.h - self.input_h % self.h
        self.pd = (0, pd_w, 0, pd_h)

    def __call__(self, input):
        # input: 2dim tensor
        input = F.pad(input, self.pd, "constant", 0)
        self.col_slide = self.num_step+1

        if self.isRow:
            self.row_slide = math.ceil(input.shape[0] / self.h)
            result = input.unfold(1,self.w,self.step_w).unfold(0,self.h,self.h).transpose(2,3).reshape(-1, self.h, self.w)
        else:
            self.row_slide = 1
            result = input.unfold(1,self.w,self.step_w).transpose(0, 1).reshape(self.col_slide, -1, self.w)
        self.arrays = result.shape[0]

        return result
