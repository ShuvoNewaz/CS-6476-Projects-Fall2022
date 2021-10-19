import numpy as np
import torch
from PIL import Image
from vision.simple_net_final import SimpleNetFinal
from tests.model_test_utils import extract_model_layers


def test_simple_net_final():
    """
    Tests the SimpleNet now contains Dropout, batchnorm, and more conv layers
    """
    this_simple_net = SimpleNetFinal()

    _, output_dim, counter, *_ = extract_model_layers(this_simple_net)

    assert counter["Dropout"] >= 1
    assert counter["Conv2d"] >= 3
    assert counter["BatchNorm2d"] >= 1
    assert counter["Linear"] >= 2
    assert counter["ReLU"] >= 2
    assert output_dim == 15
