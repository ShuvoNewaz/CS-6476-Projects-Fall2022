from vision.part3_pointnet import PointNet
from tests.model_test_utils import count_params, get_layers, get_model_layer_counts 

def test_pointnet():
    model = PointNet(20)
    params = count_params(model)
    layers = get_layers(model)
    layer_counts = get_model_layer_counts(layers)

    assert params > 8e5 and params < 8.1e5
    assert layer_counts['Linear'] == 7
    assert layer_counts['BatchNorm1d'] == 3
    assert layers[-1].out_features == 20