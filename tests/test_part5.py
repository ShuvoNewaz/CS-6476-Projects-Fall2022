from vision.part5_positional_encoding import PointNetPosEncoding
from tests.model_test_utils import count_params, get_layers, get_model_layer_counts 
import torch

def test_positional_encoding():
    model = PointNetPosEncoding(20, encoding_dim=2, pts_per_obj=5)
    pts = torch.tensor([[
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 1], 
        [0, 1, 0]
    ]])

    positional_encoding = model.positional_encoding(pts)

    expected = torch.tensor([[
        [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00, 1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00, 0.0000e+00,  1.0000e+00],
        [ 1.0000e+00, -4.3711e-08, -8.7423e-08, -1.0000e+00,  1.0000e+00,-4.3711e-08, -8.7423e-08, -1.0000e+00,  1.0000e+00, -4.3711e-08,-8.7423e-08, -1.0000e+00],
        [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00, 1.0000e+00,  0.0000e+00,  1.0000e+00,  1.0000e+00, -4.3711e-08, -8.7423e-08, -1.0000e+00],
        [ 1.0000e+00, -4.3711e-08, -8.7423e-08, -1.0000e+00,  0.0000e+00, 1.0000e+00,  0.0000e+00,  1.0000e+00,  1.0000e+00, -4.3711e-08, -8.7423e-08, -1.0000e+00],
        [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  1.0000e+00, -4.3711e-08, -8.7423e-08, -1.0000e+00,  0.0000e+00,  1.0000e+00, 0.0000e+00,  1.0000e+00]
    ]])

    assert torch.allclose(positional_encoding, expected, atol=1e-5, rtol=1e-3)


def test_pointnet_with_positional_encoding():
    model = PointNetPosEncoding(20)
    params = count_params(model)
    layers = get_layers(model)
    layer_counts = get_model_layer_counts(layers)

    assert params > 8.1e5 and params < 8.2e5 
    assert layer_counts['Linear'] == 7
    assert layer_counts['BatchNorm1d'] == 3
    assert layers[-1].out_features == 20