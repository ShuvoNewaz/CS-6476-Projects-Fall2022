from pathlib import Path
import torch
from vision.part1_dataloader import Argoverse

PROJ_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_length():
    train_dataset = Argoverse(
        data_root=f'{PROJ_ROOT}/data/sweeps',
        split='train',
        pad_size=200
    )

    test_dataset = Argoverse(
        data_root=f'{PROJ_ROOT}/data/sweeps',
        split='test',
        pad_size=200
    )

    assert len(train_dataset) == 3400
    assert len(test_dataset) == 600


def test_unique_values():
    train_dataset = Argoverse(
        data_root=f'{PROJ_ROOT}/data/sweeps',
        split='train',
        pad_size=200
    )

    cloud1, _ = train_dataset[1318]
    cloud2, _ = train_dataset[2231]

    assert not torch.allclose(cloud1, cloud2)


def test_get_points_from_file():
    train_dataset = Argoverse(
        data_root=f'{PROJ_ROOT}/data/sweeps',
        split='train',
        pad_size=200
    )

    points = train_dataset.get_points_from_file(f'{PROJ_ROOT}/data/sweeps/STOP_SIGN/0.txt')
    expected = torch.tensor([
        [-0.375, -0.0625, 0.991943359375],
        [-0.34375, 0.109375, -1.204833984375],
        [-0.1875, -0.078125, 0.331787109375],
        [-0.21875, 0.03125, -1.209716796875],
        [-0.15625, 0.03125, 0.499267578125],
        [-0.0625, -0.09375, 0.279052734375],
        [-0.09375, 0.015625, 0.770263671875],
        [0.0, -0.109375, 0.327392578125],
        [-0.09375, 0.0625, 0.936279296875],
        [-0.03125, -0.015625, 0.989013671875],
        [0.0, -0.046875, 0.716552734375],
        [0.0, -0.03125, 0.496826171875],
        [-0.0625, 0.078125, 1.152099609375],
        [0.03125, -0.0625, 0.769287109375],
        [0.03125, -0.015625, 0.935302734375],
        [0.0625, 0.0, 1.151123046875],
        [0.0625, 0.03125, 1.209716796875],
        [0.1875, -0.109375, 0.273193359375],
        [0.1875, -0.03125, 0.986083984375],
        [0.28125, -0.046875, 0.712158203125],
        [0.375, -0.03125, 0.763916015625]
    ])

    assert torch.allclose(points, expected)


def test_pad_points():
    train_dataset = Argoverse(
        data_root=f'{PROJ_ROOT}/data/sweeps',
        split='train',
        pad_size=30
    )

    points = train_dataset.get_points_from_file(f'{PROJ_ROOT}/data/sweeps/STOP_SIGN/0.txt')
    padded_points = train_dataset.pad_points(points)

    expected = torch.tensor([
        [-0.375, -0.0625, 0.991943359375],
        [-0.34375, 0.109375, -1.204833984375],
        [-0.1875, -0.078125, 0.331787109375],
        [-0.21875, 0.03125, -1.209716796875],
        [-0.15625, 0.03125, 0.499267578125],
        [-0.0625, -0.09375, 0.279052734375],
        [-0.09375, 0.015625, 0.770263671875],
        [0.0, -0.109375, 0.327392578125],
        [-0.09375, 0.0625, 0.936279296875],
        [-0.03125, -0.015625, 0.989013671875],
        [0.0, -0.046875, 0.716552734375],
        [0.0, -0.03125, 0.496826171875],
        [-0.0625, 0.078125, 1.152099609375],
        [0.03125, -0.0625, 0.769287109375],
        [0.03125, -0.015625, 0.935302734375],
        [0.0625, 0.0, 1.151123046875],
        [0.0625, 0.03125, 1.209716796875],
        [0.1875, -0.109375, 0.273193359375],
        [0.1875, -0.03125, 0.986083984375],
        [0.28125, -0.046875, 0.712158203125],
        [0.375, -0.03125, 0.763916015625],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375],
        [-0.375, -0.0625, 0.991943359375]
    ])

    assert torch.allclose(padded_points, expected)


def test_class_values():
    """ """
    test_image_loader = Argoverse(
        data_root=f"{PROJ_ROOT}/data/sweeps",
        split="test",
        pad_size=200
    )

    class_labels = test_image_loader.class_dict
    class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}

    assert len(set(class_labels.values())) == 20
    assert len(set(class_labels.keys())) == 20

    assert set(list(range(20))) == set(class_labels.values())

    assert class_labels["bus"] == 4
    assert class_labels["stroller"] == 14