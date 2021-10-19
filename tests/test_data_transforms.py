from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from vision.data_transforms import (
    get_all_transforms, get_fundamental_augmentation_transforms,
    get_fundamental_transforms)

PROJ_ROOT = Path(__file__).resolve().parent.parent


def test_fundamental_transforms():
    """
    Checks whether expected transforms are present
    """

    tr = get_fundamental_transforms(inp_size=(100, 50))
    expected = [transforms.Resize, transforms.ToTensor]
    returned_types = [type(item) for item in tr.transforms]

    for t in expected:
        assert (
            t in returned_types
        ), "one of the expected transforms was missing: {}".format(t)


def test_data_augmentation_transforms():
    """
    Checks whether expected transforms are present
    """

    tr = get_fundamental_augmentation_transforms(inp_size=(100, 50))
    expected = [transforms.Resize, transforms.ToTensor, transforms.RandomHorizontalFlip]
    returned_types = [type(item) for item in tr.transforms]

    for t in expected:
        assert (
            t in returned_types
        ), "one of the expected transforms was missing: {}".format(t)


def test_data_augmentation_with_normalization_transforms():
    """
    Checks whether expected transforms are present
    """

    tr = get_all_transforms(inp_size=(100, 50), pixel_mean=[0.5], pixel_std=[0.3])
    expected = [
        transforms.Resize,
        transforms.ToTensor,
        transforms.RandomHorizontalFlip,
        transforms.Normalize,
        transforms.ColorJitter,
    ]
    returned_types = [type(item) for item in tr.transforms]

    for t in expected:
        assert (
            t in returned_types
        ), "one of the expected transforms was missing: {}".format(t)
