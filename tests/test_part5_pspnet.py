
from types import SimpleNamespace

import torch
from torch import nn
import math

from src.vision.part3_training_utils import get_model_and_optimizer
from src.vision.part1_ppm import PPM
from src.vision.part5_pspnet import PSPNet


def test_get_model_and_optimizer_pspnet():
    """ """
    args = SimpleNamespace(
        **{
            "classes": 11,
            "zoom_factor": 8,
            "layers": 50,
            "ignore_label": 255,
            "arch": "PSPNet",
            "base_lr": 1e-3,
            "momentum": 0.99,
            "weight_decay": 1e-5,
            "pretrained": False,
            "use_ppm": True
        }
    )
    model, optimizer = get_model_and_optimizer(args)
    assert isinstance(model, PSPNet)
    assert isinstance(optimizer, torch.optim.Optimizer)

    assert len(optimizer.param_groups) > 1
    param_learning_rates = [group["lr"] for group in optimizer.param_groups]
    # some modules should be trained at this rate
    assert args.base_lr in param_learning_rates
    # should be separate rates
    assert not all([param_learning_rates == 1e-3])


def test_pspnet_output_shapes() -> None:
    """"""
    args = SimpleNamespace(
        **{
            "classes": 11,
            "zoom_factor": 8,
            "layers": 50,
            "ignore_label": 255,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        use_ppm=True,
        pretrained=False
    )

    # check whether PPM exists
    assert isinstance(model.ppm, nn.Module)
    assert isinstance(model.ppm, PPM)

    batch_size = 5
    H = 201
    W = 201
    x = torch.rand(batch_size,3,H,W).type(torch.float32)
    y = torch.ones(batch_size,H,W).type(torch.int64) * 255

    logits, yhat, main_loss, aux_loss = model(x, y)

    assert logits.shape == (batch_size, args.classes, H, W)

    # make sure that the output shape is correct
    assert yhat.shape == (batch_size, H, W)

    assert isinstance(logits, torch.Tensor)
    assert isinstance(yhat, torch.Tensor)
    assert isinstance(main_loss, torch.Tensor)
    assert isinstance(aux_loss, torch.Tensor)

    # check loss with all ground truth set to ignore index
    # assert torch.allclose(main_loss, torch.Tensor([0.]))
    # assert torch.allclose(aux_loss, torch.Tensor([0.]))

    # check the dilation settings

def test_check_output_shapes_testtime_pspnet():
    """When y is not provided to the model, losses should be None"""
    args = SimpleNamespace(
        **{
            "classes": 11,
            "zoom_factor": 8,
            "layers": 50,
            "ignore_label": 255,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        use_ppm=True,
        pretrained=False
    )

    batch_size = 5
    H = 201
    W = 201
    x = torch.rand(batch_size,3,H,W).type(torch.float32)
    y = torch.ones(batch_size,H,W).type(torch.int64) * 255

    logits, yhat, main_loss, aux_loss = model(x)

    assert logits.shape == (batch_size, args.classes, H, W)

    # make sure that the output shape is correct
    assert yhat.shape == (batch_size, H, W)

    assert isinstance(logits, torch.Tensor)
    assert isinstance(yhat, torch.Tensor)

    assert main_loss == None
    assert aux_loss == None

def test_pspnet_output_with_zoom_factor():
    args = SimpleNamespace(
        **{
            "classes": 11,
            "layers": 50,
            "ignore_label": 255,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        criterion=criterion,
        use_ppm=True,
        pretrained=False
    )
    batch_size = 5
    H = 201
    W = 201

    for zoom_factor in [1, 2, 4, 8]:
        model.zoom_factor = zoom_factor
        h_scaled = int(math.ceil(H/8 * zoom_factor))
        w_scaled = int(math.ceil(W/8 * zoom_factor))
        x = torch.rand(batch_size,3,H,W).type(torch.float32)
        y = torch.ones(batch_size,h_scaled,w_scaled).type(torch.int64) * 255

        logits, yhat, main_loss, aux_loss = model(x)

        assert logits.shape == (batch_size, args.classes, h_scaled, w_scaled)
        assert yhat.shape == (batch_size, h_scaled, w_scaled)


