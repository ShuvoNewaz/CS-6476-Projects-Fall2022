import logging
import os
import pdb
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.vision.part5_pspnet import PSPNet
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = get_logger()


def load_pretrained_model(args, use_cuda: bool):
    """Load Pytorch pre-trained PSPNet model from disk of type torch.nn.DataParallel.

    Note that `args.num_model_classes` will be size of logits output.

    Args:
        args:
        use_cuda:

    Returns:
        model
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        pretrained=False
    )

    # logger.info(model)
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        if use_cuda:
            checkpoint = torch.load(args.model_path)
        else:
            checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"=> loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")

    return model



def model_and_optimizer(args, model) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    This function is similar to get_model_and_optimizer in Part 3.

    Use the model trained on Camvid as the pretrained PSPNet model, change the
    output classes number to 2 (the number of classes for Kitti).
    Refer to Part 3 for optimizer initialization.

    Args:
        args: object containing specified hyperparameters
        model: pre-trained model on Camvid

    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    lr = args.base_lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    args.classes = 2
    model.cls[4] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)
    model.aux[4] = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
    
    layer0_params = {'params': model.layer0.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer1_params = {'params': model.layer1.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer2_params = {'params': model.layer2.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer3_params = {'params': model.layer3.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer4_params = {'params': model.layer4.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        ppm_params = {'params': model.ppm.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    cls_params = {'params': model.cls.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    aux_params = {'params': model.aux.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        optimizer = torch.optim.SGD([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        ppm_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = torch.optim.SGD([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)

    # raise NotImplementedError('`model_and_optimizer()` function in ' +
    #     '`part6_transfer_learning.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer
