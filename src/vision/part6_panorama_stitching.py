import numpy as np
import cv2
from vision.part3_ransac import ransac_fundamental_matrix


def HomographyMatrix(imageA, imageB):
    if len(imageA.shape) == 3:
        image1_bw = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        image1_bw = imageA
    if len(imageB.shape) == 3:
        image2_bw = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        image2_bw = imageB

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_bw, None)
    kp2, des2 = sift.detectAndCompute(image2_bw, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if (m.distance / n.distance) < 0.55:
            good.append(m)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    hA, wA = image1_bw.shape
    hB, wB = image2_bw.shape
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    src_pts = np.squeeze(src_pts, axis=1).T.astype(int)
    dst_pts = np.squeeze(dst_pts, axis=1).T.astype(int)

    src_pts[0] = np.minimum(src_pts[0], hA - 1)
    src_pts[1] = np.minimum(src_pts[1], wA - 1)
    dst_pts[0] = np.minimum(dst_pts[0], hB - 1)
    dst_pts[1] = np.minimum(dst_pts[1], wB - 1)

    src_pts[[0, 1]] = src_pts[[1, 0]]
    dst_pts[[0, 1]] = dst_pts[[1, 0]]
    homography[:, [0, 1]] = homography[:, [1, 0]]

    return homography, src_pts, dst_pts


def FlattenImageCoordinates(image):
    if len(image.shape) == 3:
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_bw = image
    M, N = image_bw.shape
    X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    image_coordinates = np.concatenate((X.reshape(1, M*N), Y.reshape(1, M*N)), axis=0)

    return image_coordinates


def WarpCoordinates(coordinates, homography):
    point_coordinates = np.concatenate((coordinates, np.ones((1, coordinates.shape[1]))), axis=0)
    warp = homography @ point_coordinates
    warp = warp / (warp[2] + 1e-9)
    projected_coordinates_X = np.round(warp[1]).astype(int)
    projected_coordinates_Y = np.round(warp[0]).astype(int)
    projected_coordinates = np.concatenate((projected_coordinates_X[np.newaxis, :], projected_coordinates_Y[np.newaxis, :]), axis=0)

    return projected_coordinates


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

    if len(imageA.shape) == 3:
        image1_bw = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        image1_bw = imageA
    if len(imageB.shape) == 3:
        image2_bw = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        image2_bw = imageB

    hA, wA = image1_bw.shape
    hB, wB = image2_bw.shape

    homography, src_points, dst_points = HomographyMatrix(imageA, imageB)
    flattened_image_coordinates = FlattenImageCoordinates(imageB)
    projected_image_coordinates = WarpCoordinates(flattened_image_coordinates, homography)
    projected_image_x = projected_image_coordinates[0].reshape(wB, hB)
    projected_image_y = projected_image_coordinates[1].reshape(wB, hB)
    projected_image_x_min = projected_image_coordinates[0].min()
    projected_image_y_min = projected_image_coordinates[1].min()
    projection_height = projected_image_coordinates[1].max()-projected_image_y_min
    projection_width = projected_image_coordinates[0].max()-projected_image_x_min
    panorama = np.zeros((max(hA, projection_height), wA+projection_width, 3), dtype=np.int)
    panorama[:hA, :wB, :] = imageA
    panorama[projected_image_x, projected_image_y, :] = 0
    remove_pixel_ind = np.sort(projected_image_coordinates[1][projected_image_coordinates[1] > 0])[hB*20]
    panorama[:, remove_pixel_ind:, :] = 0
    for i in range(3):
        panorama[projected_image_coordinates[0], projected_image_coordinates[1], i] = imageB[:, :, i].ravel()
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama#cv2.cvtColor(np.float32(panorama), cv2.COLOR_BGR2RGB)
