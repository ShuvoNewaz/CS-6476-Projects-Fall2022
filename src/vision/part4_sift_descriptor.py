#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from vision.part1_harris_corner import compute_image_gradients
from torch import nn, sin
from typing import Tuple


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """
    magnitudes = []  # placeholder
    orientations = []  # placeholder

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    magnitudes = np.sqrt(Ix ** 2 + Iy ** 2)
    orientations = np.arctan2(Iy, Ix)
    
    # raise NotImplementedError('`get_magnitudes_and_orientations()` function ' +
    #     'in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively. 
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    h, w = window_magnitudes.shape
    bins = np.arange(-8, 9, 2) * np.pi / 8 - 1e-6
    wgh = np.zeros(128)
    counter = 0
    for i in range(h // 4):
        for j in range(w // 4):
            hist, _ = np.histogram(window_orientations[4*i:4*(i+1), 4*j:4*(j+1)],\
                 bins=bins, weights=window_magnitudes[4*i:4*(i+1), 4*j:4*(j+1)])
            wgh[counter*8:(counter+1)*8] = hist
            counter += 1
    wgh = wgh[:, np.newaxis]

    # raise NotImplementedError('`get_gradient_histogram_vec_from_patch` ' +
    #     'function in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
    c: float,
    r: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    padded_magnitudes = np.pad(magnitudes, feature_width // 2, 'constant', constant_values=(0,))
    padded_orientations = np.pad(orientations, feature_width // 2, 'constant', constant_values=(0,))
    new_c, new_r = c + feature_width // 2, r + feature_width // 2
    start_i, start_j = new_r - (feature_width // 2 - 1), new_c - (feature_width // 2 - 1)
    magnitudes_patch = padded_magnitudes[start_i:start_i+feature_width, start_j:start_j+feature_width]
    orientations_patch = padded_orientations[start_i:start_i+feature_width, start_j:start_j+feature_width]

    fv = get_gradient_histogram_vec_from_patch(magnitudes_patch, orientations_patch)
    fv = fv / np.linalg.norm(fv.ravel())
    fv = np.sqrt(fv)

    # raise NotImplementedError('`get_feat_vec` function in ' +
    #     '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    k = len(X)
    fvs = np.zeros((k, 128))
    Ix, Iy = compute_image_gradients(image_bw=image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)
    for i in range(k):
        fvs[i] = get_feat_vec(X[i], Y[i], magnitudes=magnitudes, orientations=orientations, feature_width=feature_width).ravel()

    # raise NotImplementedError('`get_SIFT_descriptors` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


### ----------------- OPTIONAL (below) ------------------------------------

## Implementation of the function below is  optional (extra credit)

sobel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
sobel_y = np.array(
[
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]).astype(np.float32)


def ImageGradientsLayer(image_bw: np.array) -> torch.Tensor:
    sobel = np.concatenate([sobel_x.reshape(1,1,3,3), sobel_y.reshape(1,1,3,3)], axis=0)
    sobel = torch.from_numpy(sobel)

    conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, bias=False, padding=1, stride=1)
    conv2d.weight.requires_grad = False
    conv2d.weight = nn.Parameter(sobel)


    return conv2d(torch.from_numpy(image_bw).unsqueeze(0).unsqueeze(0))


def AngleBasis(angles: np.array) -> np.array:
    basis = np.zeros((len(angles), 2))
    basis[:, 0] = np.cos(angles)
    basis[:, 1] = np.sin(angles)

    return basis


def AngleGradLayer(gradients: np.array) -> torch.Tensor:
    angles = torch.Tensor([np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8,\
         9*np.pi/8, 11*np.pi/8, 13*np.pi/8, 15*np.pi/8])
    angle_vectors = torch.from_numpy(AngleBasis(angles)).to(torch.float32)
    dx = torch.Tensor([1, 0]).view(1,2)
    dy = torch.Tensor([0, 1]).view(1,2)

    conv2d = nn.Conv2d(in_channels=2, out_channels=10, kernel_size=1, bias=False)
    conv2d.weight.requires_grad = False
    weight_param = torch.cat((angle_vectors, dx,dy), dim=0).view(10,2,1,1)
    conv2d.weight = nn.Parameter(weight_param)

    return conv2d(gradients)


def HistogramLayer(angle_grad):
    M, N = angle_grad.shape[2], angle_grad.shape[3]
    
    angle_grad = angle_grad.squeeze(0).view(10, M*N)
    angles = angle_grad[:8, :]
    im_grads = angle_grad[8:, :]
    orientations = torch.zeros_like(angles)
    magnitude = torch.sqrt(im_grads[0] ** 2 + im_grads[1] ** 2)
    ind = torch.argmax(angles, dim=0).ravel().detach().numpy()

    orientations[ind, np.arange(M*N)] = magnitude[ind]
    orientations = orientations.view(8, M, N)
    orientations = orientations.unsqueeze(0)
    
    return orientations


def AccumulationLayer(histogram):
    conv2d = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, padding=2, groups=8, bias=False, stride=1)
    conv2d.weight.requires_grad = False
    conv2d.weight = nn.Parameter(torch.ones(8, 1, 4, 4))

    return conv2d(histogram)


def PatchFinder(r: int, c: int):
    x = np.linspace(r - 7, r + 8, 16)
    y = np.linspace(c - 7, c + 8, 16)
    x_grid, y_grid = np.meshgrid(x, y)

    x_grid = x_grid.ravel().astype(np.int)
    y_grid = y_grid.ravel().astype(np.int)

    return x_grid, y_grid


def get_sift_features_vectorized(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    gradients = ImageGradientsLayer(image_bw=image_bw)
    orientation = AngleGradLayer(gradients)
    histogram = HistogramLayer(orientation)
    features = AccumulationLayer(histogram)
    
    total_features=[]
    for i in range(len(X)):
        x_grid, y_grid = PatchFinder(X[i], Y[i])
        feat = features[:, :, y_grid, x_grid].view(1, 8 * len(x_grid))
        feat = nn.functional.normalize(feat, dim=1)
        feat = feat ** 0.65
        total_features.append(feat)
    fvs = torch.cat(total_features, dim=0)
    fvs= fvs.detach().numpy()


    # raise NotImplementedError('`get_SIFT_features_vectorized` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
