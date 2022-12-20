
import json
import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_logger() -> Logger:
    """Getter for the main logger."""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def save_json_dict(
    json_fpath: Union[str, "os.PathLike[str]"],
    dictionary: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary to a JSON file.
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
        function: Python function object
    Returns:
        string that is colored red or green when printed, indicating success
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'



def read_str_list(fpath: str) -> List[str]:
    """ Obtain carriage-return separated lines of a file as a list of strings. """
    return list(np.genfromtxt(fpath, delimiter='\n', dtype=str))


def load_class_names(dataset_name: str) -> List[str]:
    """
    Args:
        dataset_name: str
    Returns: 
        list of strings, representing class names
    """
    return read_str_list(f'{REPO_ROOT}/dataset_lists/{dataset_name}/{dataset_name}_names.txt')


def get_imagenet_mean_std() -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """ See use here in Pytorch ImageNet script: 
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197
    
    Returns:
        mean: r,g,b pixel means in [0,255]
        std: rgb pixel standard deviations for [0,255] data
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std


def normalize_img(
    input: torch.Tensor, 
    mean: Tuple[float,float,float], 
    std: Optional[Tuple[float,float,float]] = None
) -> None:
    """ Pass in by reference Torch tensor, and normalize its values.
    Args:
        input: Torch tensor of shape (3,M,N), must be in this order, and
            of type float (necessary).
        mean: mean values for each RGB channel
        std: standard deviation values for each RGB channel
    """
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)


def cv2_imread_rgb(fpath: str) -> np.ndarray:
    """
    Args:
    fpath:  string representing absolute path where image should be loaded from
    """
    if not Path(fpath).exists():
        print(f'{fpath} does not exist.')
        raise RuntimeError
        exit()
    return cv2.imread(fpath).copy()[:,:,::-1]


def get_dataloader_id_to_classname_map(
    dataset_name: str, 
    class_names: List[str] = None,
    include_ignore_idx_cls: bool = True, 
    ignore_index: int = 255
) -> Dict[int,str]:
    """ Get the 1:1 mapping stored in our `names.txt` file that maps a class name to a 
    data loader class index.
    Returns:
    dataloader_id_to_classname_map: dictionary mapping integers to strings
    """
    if class_names is None:
        class_names = load_class_names(dataset_name)

    dataloader_id_to_classname_map = {dataloader_id:classname for dataloader_id, classname in enumerate(class_names)}

    if include_ignore_idx_cls:
        dataloader_id_to_classname_map[ignore_index] = 'unlabeled'
    return dataloader_id_to_classname_map
