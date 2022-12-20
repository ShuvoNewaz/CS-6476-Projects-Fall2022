from types import SimpleNamespace

import numpy as np
import torch

import src.vision.cv2_transforms as transform
from src.vision.part3_training_utils import get_train_transform, get_val_transform


def test_get_train_transform():
    """Ensure that the proper crop size and types are returned."""
    args = SimpleNamespace(
        **{
            "short_size": 240,
            "train_h": 201,
            "train_w": 201,
            "scale_min": 0.5,  # minimum random scale
            "scale_max": 2.0,  # maximum random scale
            "rotate_min": -10,  # minimum random rotate
            "rotate_max": 10,  # maximum random rotate
            "ignore_label": 255,
        }
    )
    train_transform = get_train_transform(args)

    assert isinstance(train_transform, transform.Compose)

    H = 720
    W = 960
    x = np.random.randint(low=0, high=256, size=(H, W, 3)).astype(np.float32)
    y = np.random.randint(low=0, high=12, size=(H, W)).astype(np.int64)

    # feed sample (x,y) pair through
    x, y = train_transform(x, y)

    assert x.shape == (3, args.train_h, args.train_w)
    assert y.shape == (args.train_h, args.train_w)

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_get_val_transform():
    """ Ensure that the proper crop size and types are returned."""
    args = SimpleNamespace(
        **{
            "short_size": 240,
            "train_h": 201,
            "train_w": 201,
            "scale_min": 0.5,  # minimum random scale
            "scale_max": 2.0,  # maximum random scale
            "rotate_min": -10,  # minimum random rotate
            "rotate_max": 10,  # maximum random rotate
            "ignore_label": 255,
        }
    )

    val_transform = get_val_transform(args)

    assert isinstance(val_transform, transform.Compose)

    # feed sample (x,y) pair through
    H = 720
    W = 960

    # with 4:3 aspect ratio

    # Generate toy data such that the center crop will all
    # different values than the rest of the image
    # for 201 x 201 crop size
    # corresponds to [19:220, 59:260] of 240 x 320
    x = np.random.randint(low=0, high=256, size=(H, W, 3)).astype(np.float32)
    x[19 * 3 : 220 * 3, 59 * 3 : 260 * 3] = 15.0

    y = np.random.randint(low=0, high=12, size=(H, W)).astype(np.int64)
    y[19 * 3 : 220 * 3, 59 * 3 : 260 * 3] = 5

    x, y = val_transform(x, y)

    assert x.shape == (3, args.train_h, args.train_w)
    assert y.shape == (args.train_h, args.train_w)

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    expected_y = torch.ones(201,201).type(torch.int64) * 5

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.120000000000005, 57.375]

    expected_r = (15.0 - mean[0]) / std[0]
    expected_g = (15.0 - mean[1]) / std[1]
    expected_b = (15.0 - mean[2]) / std[2]

    # should be center crop region, have all the same values
    assert torch.allclose(x[0,:,:], torch.Tensor([expected_r]), atol=1e-2)
    assert torch.allclose(x[1,:,:], torch.Tensor([expected_g]), atol=1e-2)
    assert torch.allclose(x[2,:,:], torch.Tensor([expected_b]), atol=1e-2)
    assert torch.allclose(y, expected_y)
