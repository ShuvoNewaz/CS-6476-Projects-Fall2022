
import copy
from typing import List

import torch
from torch import nn

from src.vision.part1_ppm import PPM


def test_PPM_6x6():
    """Ensure Pyramid Pooling Module returns the correct return shapes.

    Check values for a single, simple (6,6) feature map as input.
    """
    input = (
        torch.Tensor(
            [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6],
                [4, 4, 5, 5, 6, 6],
                [7, 7, 8, 8, 1, 2],
                [7, 7, 8, 8, 3, 4],
            ]
        )
        .reshape(1, 1, 6, 6)
        .type(torch.float32)
    )

    # should be 4.2778
    feature_map_mean = input.mean().item()

    ppm = PPM(in_dim=1, reduction_dim=1, bins=(1, 2, 3, 6))

    # fill all conv weights with 1s, to create identity op for the unit test
    for m in ppm.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 1)

    ppm.eval()  # needed for batch size of 1 through batch norm
    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    output = ppm(copy.deepcopy(input))

    assert output.shape == (1, 5, 6, 6)
    # zero'th channel is just the input
    assert torch.allclose(output[0, 0, :, :], input)
    # last channel has 36 bins (6x6), so also equivalent to input
    assert torch.allclose(output[0, 4, :, :], input)

    expected_3x3_upsampled = torch.Tensor(
        [
            [1.0, 1.4, 1.8, 2.20, 2.60, 3.0],
            [2.2, 2.6, 3.0, 3.40, 3.80, 4.2],
            [3.4, 3.8, 4.2, 4.60, 5.00, 5.4],
            [4.6, 5.0, 5.4, 5.54, 5.42, 5.3],
            [5.8, 6.2, 6.6, 6.22, 5.06, 3.9],
            [7.0, 7.4, 7.8, 6.90, 4.70, 2.5],
        ]
    )


def test_PPM_fullres():
    """ Check for correct output sizes with full-resolution input."""
    batch_size = 10
    H = 200
    W = 300
    input = torch.rand(batch_size,100,H,W).type(torch.float32)

    ppm = PPM(in_dim=100, reduction_dim=50, bins=(1, 2, 3, 6, 12))

    output = ppm(copy.deepcopy(input))
    
    # 5 pyramid scales w/ 50 channels, plus the original input w/ 100 channels
    assert output.shape == (batch_size, (50*5) + 100, H, W)
    # zero'th channel is just the input
    assert torch.allclose(output[:,:100,:,:], input)
