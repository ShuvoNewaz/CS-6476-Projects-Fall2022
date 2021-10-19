import numpy as np
import torch
import torch.nn as nn
from vision.dl_utils import compute_accuracy, compute_loss


class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.net = nn.Linear(5, 5, bias=False)

        self.net.weight = nn.Parameter(
            torch.arange(25, dtype=torch.float32).reshape(5, 5) - 12
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x):
        return self.net(x)


def test_compute_accuracy():
    """
    Test the label prediction logic on a dummy net
    """

    test_net = SampleModel()

    x = torch.FloatTensor(
        [
            [1.4, -1.4, -0.7, 2.3, 0.3],
            [0.3, 100.4, -1.4, -3.7, 2.1],
            [2.3, 0.3, 1.4, -1.4, -5],
        ]
    )
    logits = test_net(x)
    accuracy = compute_accuracy(logits, torch.LongTensor([4, 4, 0]))
    assert np.isclose(accuracy, 1, atol=1e-2)


def test_compute_loss():
    """
    Test the loss computation on a dummy data
    """

    test_net = SampleModel()

    x = torch.FloatTensor([+1.4, -1.4, -0.7, 2.3, 0.3]).reshape(1, -1)

    assert torch.allclose(
        compute_loss(test_net, test_net(x), torch.LongTensor([4])),
        torch.FloatTensor([7.486063259420916e-05]),
        atol=5e-7,
    )
    assert torch.allclose(
        compute_loss(test_net, test_net(x), torch.LongTensor([3])),
        torch.FloatTensor([9.500075340270996]),
        atol=1e-3,
    )
