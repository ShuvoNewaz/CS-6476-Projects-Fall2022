import math

import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    num_samples = np.log(1 - prob_success) \
        / np.log(1 - ind_prob_correct**sample_size)

    # raise NotImplementedError(
    #     "`calculate_num_ransac_iterations` function "
    #     + "in `ransac.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    iterations = calculate_num_ransac_iterations(.99, 9, .9)
    threshold = 1e-5
    inlier_count = 0

    N = len(matches_a)
    matches_a = np.concatenate((matches_a, np.zeros((N, 1))), axis=1)
    matches_b = np.concatenate((matches_b, np.zeros((N, 1))), axis=1)
    for i in range(iterations):
        idx = np.random.choice(len(matches_a), 9, replace=False)
        temp_a = matches_a[idx]
        temp_b = matches_b[idx]
        temp_F = estimate_fundamental_matrix(temp_a, temp_b)
        
        a_F_b = ((matches_b @ temp_F) * matches_a).sum(axis=1)
        inlier_ind = np.where(a_F_b <= threshold)
        if len(inlier_ind) > inlier_count:
            inlier_count = len(inlier_ind)
            best_F = temp_F
            inliers_a = matches_a[inlier_ind][:, :2]
            inliers_b = matches_b[inlier_ind][:, :2]

    # raise NotImplementedError(
    #     "`ransac_fundamental_matrix` function in "
    #     + "`ransac.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
