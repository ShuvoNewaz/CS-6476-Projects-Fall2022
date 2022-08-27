from pathlib import Path

import torch
from vision.part3 import my_conv2d_pytorch

ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_my_conv2d_pytorch():
    """Assert that convolution output is correct, and groups are handled correctly
    for a 2-channel image with 4 filters (yielding 2 groups).
    """
    image = torch.zeros((1, 2, 3, 3), dtype=torch.int)
    image[0, 0] = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int)
    image[0, 1] = torch.tensor(
        [[9, 10, 11], [12, 13, 14], [15, 16, 17]], dtype=torch.int
    )

    identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.int)
    double_filter = torch.tensor([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=torch.int)
    triple_filter = torch.tensor([[0, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=torch.int)
    ones_filter = torch.ones(3, 3, dtype=torch.int)
    filters = torch.stack(
        [identity_filter, double_filter, triple_filter, ones_filter], 0
    )

    filters = filters.reshape(4, 1, 3, 3)
    feature_maps = my_conv2d_pytorch(image.int(), filters)

    assert feature_maps.shape == torch.Size([1, 4, 3, 3])

    gt_feature_maps = torch.zeros((1, 4, 3, 3), dtype=torch.int)

    # identity filter on channel 1
    gt_feature_maps[0, 0] = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int
    )
    # doubling filter on channel 1
    gt_feature_maps[0, 1] = torch.tensor(
        [[0, 2, 4], [6, 8, 10], [12, 14, 16]], dtype=torch.int
    )
    # tripling filter on channel 2
    gt_feature_maps[0, 2] = torch.tensor(
        [[27, 30, 33], [36, 39, 42], [45, 48, 51]], dtype=torch.int
    )
    gt_feature_maps[0, 3] = torch.tensor(
        [[44, 69, 48], [75, 117, 81], [56, 87, 60]], dtype=torch.int
    )

    assert torch.allclose(gt_feature_maps.int(), feature_maps.int())


if __name__ == "__main__":
    test_my_conv2d_pytorch()
