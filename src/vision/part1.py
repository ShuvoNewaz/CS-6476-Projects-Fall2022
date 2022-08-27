#!/usr/bin/python3

from typing import Tuple

import numpy as np


def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.

    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1

    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution

    Returns:
        kernel: 1d column vector of shape (k,1)

    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """

    raise NotImplementedError(
        "`create_Gaussian_kernel_1D` function in `part1.py` needs to be implemented"
    )

    return kernel


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`create_Gaussian_kernel_2D` function in `part1.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return kernel


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.

    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`my_conv2d_numpy` function in `part1.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`create_hybrid_image` function in `part1.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
