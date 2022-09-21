#!/usr/bin/python3

import dis
import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    features1_euclidean = np.sum(features1 ** 2, axis=1, keepdims=True)
    features2t = features2.T
    features2_euclidean = np.sum(features2t ** 2, axis=0, keepdims=True)
    dist2 = np.abs(features1_euclidean + features2_euclidean - 2.0 * features1 @ features2t)
    dists = np.sqrt(dist2)

    # raise NotImplementedError('`compute_feature_distances` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    threshold = 0.8275
    dists = compute_feature_distances(features1, features2)
    sorted_dists = np.sort(dists, axis=1)
    sorted_dist_ind = np.argsort(dists, axis=1)
    NNDR = sorted_dists[:, 0] / sorted_dists[:, 1]
    allowed_matches_count = (NNDR <= threshold).sum()
    confidence_ind = np.argsort(NNDR)[:allowed_matches_count]
    if len(confidence_ind) > 0:
        confidences = NNDR[confidence_ind]
        matches = np.zeros((len(confidences), 2), dtype=np.int)
        matches[:, 0] = confidence_ind
        matches[:, 1] = sorted_dist_ind[confidence_ind, 0]
    else:
        confidences, matches = [], []

    # raise NotImplementedError('`match_features_ratio_test` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
