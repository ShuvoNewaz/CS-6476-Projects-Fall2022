from typing import Tuple

import torch
from torch import nn

import src.vision.cv2_transforms as transform
from src.vision.part5_pspnet import PSPNet
from src.vision.part4_segmentation_net import SimpleSegmentationNet


def get_model_and_optimizer(args) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create your model, optimizer and configure the initial learning rates.

    Use the SGD optimizer, use a parameters list, and set the momentum and
    weight decay for each parameter group according to the parameter values
    in `args`.

    Create 5 param groups for the 0th + 1st,2nd,3rd,4th ResNet layer modules,
    and then add separate groups afterwards for the classifier and/or PPM
    heads.

    You should set the learning rate for the resnet layers to the base learning
    rate (args.base_lr), and you should set the learning rate for the new
    PSPNet PPM and classifiers to be 10 times the base learning rate.

    Args:
        args: object containing specified hyperparameters, including the "arch"
           parameter that determines whether we should return PSPNet or the
           SimpleSegmentationNet
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    lr = args.base_lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    if args.arch == "SimpleSegmentationNet":
        model = SimpleSegmentationNet(num_classes=args.classes, pretrained=args.pretrained)
        layer0_params = {'params': model.layer0.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
        resnet_layer1_params = {'params': model.resnet.layer1.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
        resnet_layer2_params = {'params': model.resnet.layer2.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
        resnet_layer3_params = {'params': model.resnet.layer3.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
        resnet_layer4_params = {'params': model.resnet.layer4.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
        cls_params = {'params': model.cls.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}

        optimizer = torch.optim.SGD([
                                        layer0_params,
                                        resnet_layer1_params,
                                        resnet_layer2_params,
                                        resnet_layer3_params,
                                        resnet_layer4_params,
                                        cls_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif args.arch == "PSPNet":
        model = PSPNet(num_classes=args.classes, use_ppm=args.use_ppm, zoom_factor=args.zoom_factor, pretrained=args.pretrained)
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
    
    # raise NotImplementedError('`get_model_and_optimizer()` function in ' +
    #     '`part3_training_utils.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer


def update_learning_rate(current_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Given an updated current learning rate, set the ResNet modules to this
    current learning rate, and the classifiers/PPM module to 10x the current
    lr.

    Hint: You can loop over the dictionaries in the optimizer.param_groups
    list, and set a new "lr" entry for each one. They will be in the same order
    you added them above, so if the first N modules should have low learning
    rate, and the next M modules should have a higher learning rate, this
    should be easy modify in two loops.

    Note: this depends upon how you implemented the param groups above.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    for i in range(len(optimizer.param_groups)):
        if i < 5:
            optimizer.param_groups[i]['lr'] = current_lr
        else:
            optimizer.param_groups[i]['lr'] = current_lr * 10

    # raise NotImplementedError('`update_learning_rate()` function in ' +
    #     '`part3_training_utils.py` needs to be implemented')


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return optimizer


def get_train_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the training split, with data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    then random horizontal flipping, blurring, rotation, scaling (in any order),
    followed by taking a random crop of size (args.train_h, args.train_w), converting
    the Numpy array to a Pytorch tensor, and then normalizing by the
    Imagenet mean and std (provided here).

    Note that your scaling should be confined to the [scale_min,scale_max] params in the
    args. Also, your rotation should be confined to the [rotate_min,rotate_max] params.

    To prevent black artifacts after a rotation or a random crop, specify the paddings
    to be equal to the Imagenet mean to pad any black regions.

    You should set such artifact regions of the ground truth to be ignored.

    Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    Args:
        args: object containing specified hyperparameters

    Returns:
        train_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    train_transform = transform.Compose([
                                        transform.ResizeShort(size=args.short_size),
                                        transform.RandomHorizontalFlip(),
                                        transform.RandomGaussianBlur(),
                                        transform.RandRotate(rotate=(args.rotate_min, args.rotate_max), padding=mean),
                                        transform.RandScale(scale=(args.scale_min, args.scale_max)),
                                        transform.Crop(size=(args.train_h, args.train_w), crop_type='rand', padding=mean),
                                        transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)
                                        ])

    # raise NotImplementedError('`get_train_transform()` function in ' +
    #     '`part3_training_utils.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return train_transform


def get_val_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the val split, with no data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    taking a *center* crop of size (args.train_h, args.train_w) with a padding equal
    to the Imagenet mean, converting the Numpy array to a Pytorch tensor, and then
    normalizing by the Imagenet mean and std (provided here).

    Args:
        args: object containing specified hyperparameters

    Returns:
        val_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    val_transform = transform.Compose([
                                        transform.ResizeShort(size=args.short_size),
                                        transform.Crop(size=(args.train_h, args.train_w), crop_type='center', padding=mean),
                                        transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)
                                        ])

    # raise NotImplementedError('`get_val_transform()` function in ' +
    #     '`part3_training_utils.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return val_transform
