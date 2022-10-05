"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    N = len(points)
    mean = np.mean(points, axis=0)[np.newaxis, :]
    std = np.std(points, axis=0)[np.newaxis, :]

    points_normalized = (points - mean) / std

    A = np.concatenate((points, np.ones((N, 1))), axis=1)
    B = np.concatenate((points_normalized, np.ones((N, 1))), axis=1)
    T = np.linalg.inv((A.T @ A) + 1e-9) @ (A.T @ B)
    T = T.T

    # raise NotImplementedError(
    #     "`normalize_points` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = (T_b.T @ F_norm) @ T_a

    # raise NotImplementedError(
    #     "`unnormalize_F` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    N = len(points_a)
    A = np.zeros((N, 8))
    A[:, 0] = points_a[:, 0] * points_b[:, 0]
    A[:, 1] = points_a[:, 1] * points_b[:, 0]
    A[:, 2] = points_b[:, 0]
    A[:, 3] = points_a[:, 0] * points_b[:, 1]
    A[:, 4] = points_a[:, 1] * points_b[:, 1]
    A[:, 5] = points_b[:, 1]
    A[:, 6] = points_a[:, 0]
    A[:, 7] = points_a[:, 1]

    B = -np.ones((N, 1))

    F = np.linalg.inv((A.T @ A) + 1e-9) @ (A.T @ B)
    F = np.concatenate((F.ravel(), [1]), axis=0).reshape(3, 3)
    U, S, VT = np.linalg.svd(F)
    S[len(S) - 1] = 0
    F = (U @ np.diag(S)) @ VT

    # raise NotImplementedError(
    #     "`estimate_fundamental_matrix` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
