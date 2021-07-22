# DeepLabv3+(modified aligned xception as backbone) implementation for the use in MONAI
# Code adapted from https://github.com/MLearing/Pytorch-DeepLab-v3-plus/blob/master/networks/deeplab_xception.py

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool
from monai.utils import ensure_tuple_rep
import math
from typing import Optional, Sequence, Type, Union

__all__ = ["Deeplab"]



def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False,norm=None):
        super(SeparableConv2d_same, self).__init__()
        dim = 3
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = Convolution(dim, inplanes, inplanes, kernel_size=kernel_size,
                                 groups=inplanes, padding=0, dilation=dilation, bias=bias, strides=stride)
        if norm == None:
           self.pointwise = Convolution(dim, inplanes, planes, kernel_size=1, strides=1,
                                     padding=0, dilation=1, groups=1, bias=bias)
        else:
           self.pointwise = Convolution(dim, inplanes, planes, kernel_size=1, strides=1,
                                     padding=0, dilation=1, groups=1, bias=bias,norm=Norm.BATCH)

    def forward(self, x):
        x = fixed_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        dim = 3
        if planes != inplanes or stride != 1:
            self.skip = Convolution(dim, inplanes, planes, kernel_size=1, bias=False, strides=stride,norm=Norm.BATCH)
        else:
            self.skip = None

        self.relu = Act[Act.RELU](inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,norm=Norm.BATCH))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation,norm=Norm.BATCH))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,norm=Norm.BATCH))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(
        self,
        dim: int = 3,
        in_chns: int = 1,
        out_chns: int = 4,
    ):

        super(Xception, self).__init__()

        entry_block3_stride = 2
        middle_block_dilation = 1
        exit_block_dilations = (1, 2)

        # entry flow
        self.conv1 = Convolution(dim, in_chns, 32, kernel_size=3,bias=False, strides=2, padding=1,norm=Norm.BATCH)
        self.relu = Act[Act.RELU](inplace=True)

        self.conv2 = Convolution(dim, 32, 64, kernel_size=3,bias=False, strides=1, padding=1,norm=Norm.BATCH)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1],norm=Norm.BATCH)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1],norm=Norm.BATCH)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1],norm=Norm.BATCH)



    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        return x, low_level_feat


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        dim = 3
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = Convolution(dim, inplanes, planes, kernel_size=kernel_size,
                                              strides=1, padding=padding, dilation=dilation, bias=False,norm=Norm.BATCH)
        self.relu = Act[Act.RELU]()


    def forward(self, x):
        x = self.atrous_convolution(x)

        return self.relu(x)



class Deeplab(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
    ):

        super(Deeplab, self).__init__()
        self.xception_features = Xception(dimensions, in_channels, out_channels)

        # ASPP
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPP_module(2048, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(2048, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(2048, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(2048, 256, dilation=dilations[3])

        self.relu = Act[Act.RELU]()
        pool_type: Type[Union[nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.ADAPTIVEAVG, dimensions]
        self.global_avg_pool = nn.Sequential(pool_type(1),
                                             Convolution(dimensions, 2048, 256, kernel_size=1, strides=1, bias=False,norm=Norm.BATCH),
                                             Act[Act.RELU]())

        self.conv1 = Convolution(dimensions, 1280, 256, kernel_size=1, bias=False,norm=Norm.BATCH)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = Convolution(dimensions, 128, 48, kernel_size=1, bias=False,norm=Norm.BATCH)

        self.last_conv = nn.Sequential(Convolution(dimensions, 304, 256, kernel_size=3, strides=1, padding=1, bias=False,norm=Norm.BATCH),
                                       Act[Act.RELU](),
                                       Convolution(dimensions, 256, 256, kernel_size=3,
                                                   strides=1, padding=1, bias=False,norm=Norm.BATCH),
                                       Act[Act.RELU](),
                                       Convolution(dimensions, 256, out_channels, kernel_size=1, strides=1))

    def forward(self, input):

        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=(x4.size()[2],x4.size()[3],x4.size()[4]), mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-3] / 4)), int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='trilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=(input.size()[2],input.size()[3],input.size()[4]), mode='trilinear', align_corners=True)

        return x
