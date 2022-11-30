import torch
from torch import nn
from vision.part4_analysis import get_critical_indices, get_confusion_matrix
import numpy as np

class CritIndexTestNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        
        class_outputs = torch.tensor([[0.3, 0.2, 0.1, 0.1, 0.1]])
        encodings = torch.tensor([[
            [2, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1],
            [1, 1, 1, 1, 1, 2]
        ]])

        return class_outputs, encodings


def test_critical_indices():
    test_model = CritIndexTestNet()
    crit_indices = get_critical_indices(test_model, torch.arange(0, 18).view(1,6,3))
    expected = np.array([0, 1, 3, 5, 7, 8])

    assert np.allclose(crit_indices, expected)


class CritIndexDuplicateTestNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        
        class_outputs = torch.tensor([[0.3, 0.2, 0.1, 0.1, 0.1]])
        encodings = torch.tensor([[
            [3, 1, 3, 1, 1, 1],
            [1, 1, 2, 1, 1, 1],
            [1, 1, 2, 1, 3, 1],
            [1, 1, 1, 3, 1, 1],
            [1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 2, 1],
            [1, 3, 1, 1, 1, 3],
            [1, 1, 2, 1, 1, 1],
            [1, 2, 1, 1, 1, 2]
        ]])

        return class_outputs, encodings


def test_critical_indices_with_duplicates():
    test_model = CritIndexDuplicateTestNet()
    crit_indices = get_critical_indices(test_model, torch.arange(0, 18).view(1,6,3))
    expected = np.array([0, 2, 3, 6])

    assert np.allclose(crit_indices, expected)


class ConfusionMatrixTestNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max = torch.argmax(x.squeeze(0).squeeze(0))
        class_outputs = torch.ones((1, 3)) * 0.1
        class_outputs[0, max] = 1
        encodings = torch.ones((1, 1, 7))

        return class_outputs, encodings


def test_confusion_matrix():

    test_model = ConfusionMatrixTestNet()
    
    test_data_loader = [
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([0])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([0])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([0])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([1])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([1])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([1])),
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([2])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([2])),
        (torch.tensor([[[1, 1, 2]]]), torch.tensor([2])),
    ]

    confusion_matrix = get_confusion_matrix(test_model, test_data_loader, 3, False)

    expected = np.array([
        [1, 2, 0],
        [0, 3, 0],
        [1, 1, 1]
    ])

    assert np.allclose(confusion_matrix, expected)


def test_confusion_matrix_normalized():

    test_model = ConfusionMatrixTestNet()
    
    test_data_loader = [
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([0])),
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([0])),
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([0])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([1])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([1])),
        (torch.tensor([[[1, 0, 2]]]), torch.tensor([1])),
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([2])),
        (torch.tensor([[[2, 1, 1]]]), torch.tensor([2])),
        (torch.tensor([[[1, 2, 1]]]), torch.tensor([2])),
    ]

    confusion_matrix = get_confusion_matrix(test_model, test_data_loader, 3, True)

    expected = np.array([
        [1,   0,   0  ],
        [0,   2/3, 1/3],
        [2/3, 1/3, 0  ]
    ])

    assert np.allclose(confusion_matrix, expected)