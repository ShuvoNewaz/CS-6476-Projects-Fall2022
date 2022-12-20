from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class PPM(nn.Module):
    """
    A hierarchical global prior, containing information with different scale
    and varying among different sub-regions.

    The pyramid pooling module fuses features under four different pyramid
    scales.

    The coarsest level highlighted in red is global pooling to generate a
    single bin output. The following pyramid level separates the feature map
    into different sub-regions and forms pooled representation for different
    locations. The output of different levels in the pyramid pooling module
    contains the feature map with varied sizes. To maintain the weight of
    global feature, we use 1×1 convolution layer after each pyramid level to
    reduce the dimension of context representation to 1/N of the original one
    if the level size of pyramid is N.
    """

    def __init__(self, in_dim: int, reduction_dim: int, bins: List[int]) -> None:
        """
        If bins=(1, 2, 3, 6), then the PPM will operate at four levels, with
        bin sizes of 1×1, 2×2, 3×3 and 6×6, respectively.

        The PPM utilizes nn.AdaptiveAvgPool2d(bin) to break an image into
        (bin x bin) subregions, and then pools all entries inside each
        subregion.

        When initializing the Conv2d layer, set the bias to False. Some students
        in the past also had to set affine to false in the BatchNorm layer.

        For each bin size, you should create a sequential module of
        (AdaptiveAvgPool2d, 2d conv w/ 1x1 kernel, 2d batch norm, and ReLU).
        These modules should be stored inside the self.features attribute.
        """
        super().__init__()
        self.features = []

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        for bin_size in bins:
            self.features.append(nn.Sequential(
                                                nn.AdaptiveAvgPool2d([bin_size, bin_size]),
                                                nn.Conv2d(in_channels=in_dim, out_channels=reduction_dim,
                                                            kernel_size=1, stride=1, bias=False),
                                                nn.BatchNorm2d(reduction_dim),
                                                nn.ReLU()
                                                 ))

        # raise NotImplementedError('`__init__()` function in ' +
        #     '`part1_ppm.py` needs to be implemented')

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        self.features = nn.ModuleList(self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the PPM module.

        Feed the input through each module in the module list, upsample to the
        desired output size, and append each output to a list of outputs.
        Finally, concatenate them along the channel dimension. The first item
        in the output list should be the input itself.

        For upsampling, use Pytorch's bilinear interpolation, and ensure that
        corners are aligned.

        Args:
            x: tensor of shape (N,in_dim,H,W)

        Returns:
            out: tensor of shape (N,C,H,W) where
                C = in_dim + len(bins)*reduction_dim
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        output = x
        H, W = x.shape[2:]
        for bin_features in self.features:
            out = bin_features(x)
            out = F.interpolate(input=out, size=[H, W], mode='bilinear', align_corners=True)
            output = torch.concat((output, out), dim=1)

        # raise NotImplementedError('`forward()` function in ' +
        #     '`part1_ppm.py` needs to be implemented')

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return output
