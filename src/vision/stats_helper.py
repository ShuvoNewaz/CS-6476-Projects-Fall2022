import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    count = 0

    for data_split_name in os.listdir(dir_name):
        data_split_dir = os.path.join(dir_name, data_split_name)
        for class_name in os.listdir(data_split_dir):
            class_dir = os.path.join(data_split_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_dir = os.path.join(class_dir, image_name)
                image = Image.open(image_dir).convert(mode='L')
                image = np.array(image) / 255
                image = image.ravel()[:, np.newaxis]
                if count == 0:
                    all_image = image
                else:
                    all_image = np.concatenate((all_image, image), axis=0)
                count += 1
    all_image = all_image.ravel()
    mean = all_image.mean()
    var = all_image.var()
    std = np.sqrt(var)

    # raise NotImplementedError(
    #         "`compute_mean_and_std` function in "
    #         + "`stats_helper.py` needs to be implemented"
    #     )

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
