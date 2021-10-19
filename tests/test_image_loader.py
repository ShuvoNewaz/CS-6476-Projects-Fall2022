from pathlib import Path

import numpy as np
import torch
from vision.data_transforms import get_fundamental_transforms
from vision.image_loader import ImageLoader

PROJ_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_length():
    train_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="test",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    assert train_image_loader.__len__() == 2985
    assert test_image_loader.__len__() == 1500


def test_unique_vals():
    train_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    item1 = train_image_loader.__getitem__(10)
    item2 = train_image_loader.__getitem__(25)

    assert not torch.allclose(item1[0], item2[0])


def test_class_values():
    """ """
    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="test",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )

    class_labels = test_image_loader.class_dict
    class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}

    # should be 15 unique keys and 15 unique values in the dictionary
    assert len(set(class_labels.values())) == 15
    assert len(set(class_labels.keys())) == 15

    # indices must be ordered from [0,14] only
    assert set(list(range(15))) == set(class_labels.values())

    # must be ordered alphabetically
    assert class_labels["industrial"] == 4
    assert class_labels["suburb"] == 13


def test_load_img_from_path():
    test_image_loader = ImageLoader(
        root_dir=f"{PROJ_ROOT}/data/",
        split="train",
        transform=get_fundamental_transforms(inp_size=(64, 64)),
    )
    im_path = f"{PROJ_ROOT}/data/train/bedroom/image_0003.jpg"

    im_np = np.asarray(test_image_loader.load_img_from_path(im_path))

    expected_data = np.loadtxt(f"{PROJ_ROOT}/tests/data/sample_inp.txt")

    assert np.allclose(expected_data, im_np)


if __name__ == "__main__":
    test_load_img_from_path()
