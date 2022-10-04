import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental_matrix

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!
    
    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    panorama = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError("`panorama_stitch` function in "
        + "`part6_panorama_stitching.py` needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama
