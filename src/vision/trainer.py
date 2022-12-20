import argparse
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.distributed as dist

from src.vision.utils import get_logger, save_json_dict, load_class_names
from src.vision.iou import intersectionAndUnionGPU
from src.vision.avg_meter import AverageMeter, SegmentationAverageMeter

from src.vision.part2_dataset import SemData, KittiData
from src.vision.part3_training_utils import (
    get_model_and_optimizer,
    get_train_transform,
    get_val_transform,
    update_learning_rate,
)
from src.vision.part5_pspnet import PSPNet
from src.vision.part6_transfer_learning import load_pretrained_model, model_and_optimizer

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logger = get_logger()


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == "psp":
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    else:
        raise Exception("architecture not supported yet".format(args.arch))


def poly_learning_rate(base_lr: float, curr_iter, max_iter, power: float = 0.9) -> float:
    """Compute the learning rate at a specific iteration, given a polynomial learning rate policy."""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def main_worker(args, use_cuda: bool):
    """ """
    model, optimizer = get_model_and_optimizer(args)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    # logger.info(model)

    if use_cuda:
        model = model.cuda()

    # if args.weight:
    #     if os.path.isfile(args.weight):

    #         logger.info("=> loading weight '{}'".format(args.weight))
    #         checkpoint = torch.load(args.weight)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         logger.info("=> loaded weight '{}'".format(args.weight))
    #     else:
    #         logger.info("=> no weight found at '{}'".format(args.weight))

    # data_aug hyperparameter
    if args.data_aug:
        train_transform = get_train_transform(args)
    else:
        train_transform = get_val_transform(args)
    train_data = SemData(
        split="train", data_root=args.data_root, data_list_fpath=args.train_list, transform=train_transform
    )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_transform = get_val_transform(args)
    val_data = SemData(split="val", data_root=args.data_root, data_list_fpath=args.val_list, transform=val_transform)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    results_dict = defaultdict(list)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = run_epoch(
            args,
            use_cuda,
            train_loader,
            model,
            optimizer,
            epoch,
            split="train",
        )
        results_dict["loss_train"] += [round(float(loss_train), 3)]
        results_dict["mIoU_train"] += [round(float(mIoU_train), 3)]
        results_dict["mAcc_train"] += [round(float(mAcc_train), 3)]
        results_dict["allAcc_train"] += [round(float(allAcc_train), 3)]

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + "/train_epoch_" + str(epoch_log) + ".pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {"epoch": epoch_log, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename
            )
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + "/train_epoch_" + str(epoch_log - args.save_freq * 2) + ".pth"
                os.remove(deletename)
        if args.evaluate:
            with torch.no_grad():
                loss_val, mIoU_val, mAcc_val, allAcc_val = run_epoch(
                    args, use_cuda, val_loader, model, optimizer=None, epoch=epoch, split="val"
                )
            results_dict["loss_val"] += [round(float(loss_val), 3)]
            results_dict["mIoU_val"] += [round(float(mIoU_val), 3)]
            results_dict["mAcc_val"] += [round(float(mAcc_val), 3)]
            results_dict["allAcc_val"] += [round(float(allAcc_val), 3)]

    logger.info("======> Training complete ======>")
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    with torch.no_grad():
        loss_val, mIoU_val, mAcc_val, allAcc_val = run_epoch(
            args, use_cuda, val_loader, model, optimizer=None, epoch=epoch, split="val"
        )
    logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    print("Results Dict: ", results_dict)
    save_json_dict(os.path.join(args.save_path, "training_results_dict.json"), results_dict)


def transfer_train(args, use_cuda: bool):
    """ """
    # model, optimizer = get_model_and_optimizer(args)
    args.classes = 11
    pre_model = load_pretrained_model(args, use_cuda)
    args.classes = 2
    model, optimizer = model_and_optimizer(args, pre_model)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)

    if use_cuda:
        model = model.cuda()

    # if args.weight:
    #     if os.path.isfile(args.weight):

    #         logger.info("=> loading weight '{}'".format(args.weight))
    #         checkpoint = torch.load(args.weight)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         logger.info("=> loaded weight '{}'".format(args.weight))
    #     else:
    #         logger.info("=> no weight found at '{}'".format(args.weight))

    # data_aug hyperparameter
    if args.data_aug:
        train_transform = get_train_transform(args)
    else:
        train_transform = get_val_transform(args)
    train_data = KittiData(split="train", data_root=args.data_root, transform=train_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_transform = get_val_transform(args)
    val_data = KittiData(split="test", data_root=args.data_root, transform=val_transform)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    results_dict = defaultdict(list)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = run_epoch(
            args,
            use_cuda,
            train_loader,
            model,
            optimizer,
            epoch,
            split="train",
        )
        results_dict["loss_train"] += [round(float(loss_train), 3)]
        results_dict["mIoU_train"] += [round(float(mIoU_train), 3)]
        results_dict["mAcc_train"] += [round(float(mAcc_train), 3)]
        results_dict["allAcc_train"] += [round(float(allAcc_train), 3)]

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + "/train_epoch_" + str(epoch_log) + ".pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {"epoch": epoch_log, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename
            )
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + "/train_epoch_" + str(epoch_log - args.save_freq * 2) + ".pth"
                os.remove(deletename)
        if args.evaluate:
            with torch.no_grad():
                loss_val, mIoU_val, mAcc_val, allAcc_val = run_epoch(
                    args, use_cuda, val_loader, model, optimizer=None, epoch=epoch, split="val"
                )
            results_dict["loss_val"] += [round(float(loss_val), 3)]
            results_dict["mIoU_val"] += [round(float(mIoU_val), 3)]
            results_dict["mAcc_val"] += [round(float(mAcc_val), 3)]
            results_dict["allAcc_val"] += [round(float(allAcc_val), 3)]

    logger.info("======> Training complete ======>")
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    with torch.no_grad():
        loss_val, mIoU_val, mAcc_val, allAcc_val = run_epoch(
            args, use_cuda, val_loader, model, optimizer=None, epoch=epoch, split="val"
        )
    logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    print("Results Dict: ", results_dict)
    save_json_dict(os.path.join(args.save_path, "training_results_dict.json"), results_dict)


def run_epoch(
    args,
    use_cuda: bool,
    data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    split: str,
) -> Tuple[float, float, float, float]:
    """
    Run the network over all examples within a dataset split. If this split is the train split, also run backprop.
    """
    class_names = load_class_names(dataset_name=args.dataset)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()

    sam = SegmentationAverageMeter()

    if split == "train":
        model.train()
    elif split in ["val", "test"]:
        model.eval()

    end = time.time()
    max_iter = args.epochs * len(data_loader)
    for i, (input, target) in enumerate(data_loader):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = (
                F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=True)
                .squeeze(1)
                .long()
            )
            # output = F.interpolate(output, size=target.size()[1:], mode="bilinear", align_corners=True)

        if use_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        _, preds, main_loss, aux_loss = model(input, target)

        # adding aux_loss hyperparameter
        if not args.aux_loss:
            aux_loss = torch.Tensor([0])

        main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        if split == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n = input.size(0)

        sam.update_metrics_gpu(preds, target, args.classes, args.ignore_label, args.multiprocessing_distributed)

        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if split == "train":
            current_iter = epoch * len(data_loader) + i + 1
            current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)

            optimizer = update_learning_rate(current_lr, optimizer)

            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        else:
            remain_time = 0  # dummy value

        if (i + 1) % args.print_freq == 0:

            iou_class, accuracy_class, mIoU, mAcc, allAcc = sam.get_metrics()

            logger_message = f"{split} Epoch: [{epoch + 1}/{args.epochs}][{i+1}/{len(data_loader)}] "
            logger_message += f"mIoU {mIoU} "
            logger_message += f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
            logger_message += f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
            logger_message += f"Remain {remain_time} "
            logger_message += f"MainLoss {main_loss_meter.val:.4f} "
            logger_message += f"AuxLoss {aux_loss_meter.val:.4f} "
            logger_message += f"Loss {loss_meter.val:.4f} "
            logger.info(logger_message)

    iou_class, accuracy_class, mIoU, mAcc, allAcc = sam.get_metrics()

    if split == "train":
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
    else:
        logger.info(f"Val result: mIoU/mAcc/allAcc {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}.")
        for i in range(args.classes):
            logger.info(
                f"Class_{i} - {class_names[i]} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}."
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return main_loss_meter.avg, mIoU, mAcc, allAcc


def main(opts):
    """ """
    # check(args)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    use_cuda = torch.cuda.is_available()
    main_worker(args, use_cuda)


def check_mkdir(dirpath: str) -> None:
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)


DEFAULT_ARGS = SimpleNamespace(
    **{
        # DATA
        "names_path": "../dataset_lists/camvid-11/camvid-11_names.txt",
        "data_root": "../Camvid/",
        "train_list": "../dataset_lists/camvid-11/list/train.txt",  # 'mseg-api/mseg/dataset_lists/camvid-11/list/train.txt',
        "val_list": "../dataset_lists/camvid-11/list/val.txt",  # 'mseg-api/mseg/dataset_lists/camvid-11/list/val.txt',
        "classes": 11,
        # TRAIN
        "arch": "PSPNet",  #  "SimpleSegmentationNet", #
        "epochs": 100,
        "zoom_factor": 8,
        "use_ppm": True,
        "aux_weight": 0.4,
        "aux_loss": True,
        "save_path": "exp/camvid/pspnet50/model",
        "layers": 50,
        "workers": 2,
        "batch_size": 32,
        "batch_size_val": 32,
        "short_size": 240,
        "data_aug": True,
        "train_h": 201,
        "train_w": 201,
        "init_weight": "../initmodel/resnet50_v2.pth",
        "scale_min": 0.5,  # minimum random scale
        "scale_max": 2.0,  # maximum random scale
        "rotate_min": -10,  # minimum random rotate
        "rotate_max": 10,  # maximum random rotate
        "ignore_label": 255,
        "base_lr": 0.01,
        "start_epoch": 0,
        "power": 0.9,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "manual_seed": 0,
        "print_freq": 10,
        "save_freq": 1,
        "evaluate": True,  # evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend
        "multiprocessing_distributed": False,
        "pretrained": True,
        # INFERENCE
        "dataset": "camvid-11",
        "base_size": 720,
        "test_h": 201,
        "test_w": 201,
        "scales": [1.0],  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        "test_list": "../dataset_lists/camvid-11/list/val.txt",
        "vis_freq": 10,
    }
)


if __name__ == "__main__":
    check_mkdir(args.save_path)
    print(DEFAULT_ARGS)
    main(DEFAULT_ARGS)
