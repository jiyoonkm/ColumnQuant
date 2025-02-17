# ColumnQuant

A convolutional framework for column-wise weight and partial-sum quantization.

The paper will be presented at DATE 2025.

**Column-wise Quantization of Weights and Partial Sums for Accurate and Efficient Compute-In-Memory Accelerators**

🔗 https://www.arxiv.org/abs/2502.07842

## Overview

`SplitConv4Pim_group.py` implements a convolution framework designed for compute-in-memory (CIM) accelerators. This framework supports:
- Weight quantization from layer-wise to column-wise level
- Partial-sum quantization from layer-wise to column-wise level

The primary functionality includes handling quantization using LSQ (Learned Step Size Quantization) for both weights and partial-sums, enabling precise control over various granularities and optimization for CIM architectures.

## Key Features

- **Weight Splitting and Quantization:** The framework splits weights based on the number of bits per cell and applies quantization through LSQ.
- **Partial-Sum Quantization:** Supports quantization of partial sums across various granularities.
- **Group Convolution Support:** Convolutions are performed across groups of arrays, enabling faster operations.
- **Row-wise Tiling:** Provides flexibility to tile mapped weights row-wise for efficient implementation of partial-sum quantization.

## Main Classes

### `SplitConv4Pim_group`
This class implements the core convolution framework for CIM architectures. 

#### Parameters:
- `w_bit`: Number of bits for weight quantization.
- `split_bit`: Number of bits per cell.
- `w_mode`: Mode for weight processing (`'Array'` or `'Layer'`).
- `ps_bit`: Number of bits for partial-sum quantization.
- `num_sigma`: Controls the clipping range for partial sums, restricting values to the range `[mu-num_sigma*sigma, mu+num_sigma*sigma]`.
- `psum_mode`: Mode for partial-sum quantization (`'Array'` or `'Layer'`).
- `in_planes`: Number of input channels.
- `planes`: Number of output channels.
- `kernel_size`: Kernel size of the convolution.
- `N`: CIM array size.
- `stride`: Convolution stride.
- `padding`: Convolution padding.
- `bias`: Whether to use a bias term.
- `isRow`: Tile weights row-wise if `True`.
- `w_per_ch`: Enables weight quantization on an output-channel basis if `True`.
- `ps_per_ch`: Enables partial-sum quantization on an output-channel basis if `True`.
- `psumOpt`: Enables partial-sum quantization if `True`.

#### Key Methods:
- `forward`: Executes the forward pass.

### `Conv4Pim_group_split`
Handles layer-wise or channel-wise weight quantization.

### `Conv4Pim_group_arr`
Handles array-wise or column-wise weight quantization.

## LSQ (Learned Step Size Quantization)

`LSQ.py` follows [LSQ (Learned Step Size Quantization)](https://arxiv.org/abs/1902.08153) proposed by Steven K. Esser et al. from IBM for both weight and partial-sum quantization. Our LSQ implementation is revised to support various granularities based on [LSQ-Net repository](https://github.com/zhutmost/lsq-net).

### Key LSQ Features:
- **Scale Factor Initialization**: The scale factors for weight quantization is initialized using the mean absolute value of the tensor. In the case of partial-sum quantization, the scale factors are initialized to constant values.
- **Learned Step Sizes**: The quantization scale factors are learnable parameters and are updated during training, improving accuracy and adaptability to different network architectures.
- **Support for Per-channel Quantization**: We provide options to enable per-channel quantization, aligning with the flexibility of LSQ.


## Usage

```python
import torch
from SplitConv4Pim_group import SplitConv4Pim_group

# Example initialization: Array-wise weight and column-wise partial-sum quantization
conv = SplitConv4Pim_group(
    w_bit=4,
    split_bit=2,
    w_mode='Array',
    ps_bit=3,
    num_sigma=6,
    psum_mode='Array',
    in_planes=64,
    planes=128,
    kernel_size=3,
    N=256,
    stride=1,
    padding=1,
    bias=False,
    isRow=True,
    w_per_ch=False,
    ps_per_ch=True,
    psumOpt=True
)

# Forward pass
input_tensor = torch.randn(1, 64, 32, 32)  # Example input tensor
output = conv(input_tensor)
