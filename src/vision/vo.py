import glob
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour import Color
from vision.part3_ransac import ransac_fundamental_matrix
from vision.utils import get_matches, load_image
from scipy.spatial.transform import Rotation

# DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = "data"


def get_emat_from_fmat(
    i2_F_i1: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """Create essential matrix from camera instrinsics and fundamental matrix"""
    i2_E_i1 = K2.T @ i2_F_i1 @ K1
    return i2_E_i1


def load_log_front_center_intrinsics() -> np.array:
    """Provide camera parameters for front-center camera for Argoverse vehicle log ID:
    273c1883-673a-36bf-b124-88311b1a80be
    """
    fx = 1392.1069298937407  # also fy
    px = 980.1759848618066
    py = 604.3534182680304

    K = np.array([[fx, 0, px], [0, fx, py], [0, 0, 1]])
    return K


def get_visual_odometry():
    """ """
    img_wildcard = f"{DATA_ROOT}/vo_seq_argoverse_273c1883/ring_front_center/*.jpg"
    img_fpaths = glob.glob(img_wildcard)
    img_fpaths.sort()
    num_imgs = len(img_fpaths)

    K = load_log_front_center_intrinsics()

    poses_wTi = []

    poses_wTi += [np.eye(4)]

    for i in range(num_imgs - 1):
        img_i1 = load_image(img_fpaths[i])
        img_i2 = load_image(img_fpaths[i + 1])
        pts_a, pts_b = get_matches(img_i1, img_i2, n_feat=int(4e3))

        # between camera at t=i and t=i+1
        i2_F_i1, inliers_a, inliers_b = ransac_fundamental_matrix(pts_a, pts_b)
        i2_E_i1 = get_emat_from_fmat(i2_F_i1, K1=K, K2=K)
        _num_inlier, i2Ri1, i2ti1, _ = cv2.recoverPose(i2_E_i1, inliers_a, inliers_b)

        # form SE(3) transformation
        i2Ti1 = np.eye(4)
        i2Ti1[:3, :3] = i2Ri1
        i2Ti1[:3, 3] = i2ti1.squeeze()

        # use previous world frame pose, to place this camera in world frame
        # assume 1 meter translation for unknown scale (gauge ambiguity)
        wTi1 = poses_wTi[-1]
        i1Ti2 = np.linalg.inv(i2Ti1)
        wTi2 = wTi1 @ i1Ti2
        poses_wTi += [wTi2]

        r = Rotation.from_matrix(i2Ri1.T)
        rz, ry, rx = r.as_euler("zyx", degrees=True)
        print(f"Rotation about y-axis from frame {i} -> {i+1}: {ry:.2f} degrees")

    return poses_wTi


def plot_poses(poses_wTi: List[np.ndarray], figsize=(7, 8)) -> None:
    """
    Poses are wTi (in world frame, which is defined as 0th camera frame)
    """
    axis_length = 0.5

    num_poses = len(poses_wTi)
    colors_arr = np.array(
        [
            [color_obj.rgb]
            for color_obj in Color("red").range_to(Color("green"), num_poses)
        ]
    ).squeeze()

    _, ax = plt.subplots(figsize=figsize)

    for i, wTi in enumerate(poses_wTi):
        wti = wTi[:3, 3]

        # assume ground plane is xz plane in camera coordinate frame
        # 3d points in +x and +z axis directions, in homogeneous coordinates
        posx = wTi @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
        posz = wTi @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

        ax.plot([wti[0], posx[0]], [wti[2], posx[2]], "b", zorder=1)
        ax.plot([wti[0], posz[0]], [wti[2], posz[2]], "k", zorder=1)

        ax.scatter(wti[0], wti[2], 40, marker=".", color=colors_arr[i], zorder=2)

    plt.axis("equal")
    plt.title("Egovehicle trajectory")
    plt.xlabel("x camera coordinate (of camera frame 0)")
    plt.ylabel("z camera coordinate (of camera frame 0)")


# if __name__ == '__main__':
# 	get_visual_odometry()
