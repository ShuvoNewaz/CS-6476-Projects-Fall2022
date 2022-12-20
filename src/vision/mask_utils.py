import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def write_six_img_grid_w_embedded_names(
    rgb_img: np.ndarray,
    pred: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Dict[int, str],
    save_fpath: str,
) -> None:
    """
    Create a 6-image tile grid with the following structure:
    ------------------------------------------------------------
    RGB Image | Blended RGB+GT Label Map   | GT Label Map
    ------------------------------------------------------------
    RGB Image | Blended RGB+Pred Label Map | Predicted Label Map
    ------------------------------------------------------------
    We embed classnames directly into the predicted and ground
    truth label maps, instead of using a colorbar.
    Args:
        rgb_img:
        pred: predicted label map
        label_img
        id_to_class_name_map
        save_fpath
    """
    os.makedirs(Path(save_fpath).parent, exist_ok=True)
    assert label_img.ndim == 2
    assert pred.ndim == 2
    assert rgb_img.ndim == 3
    label_hgrid = form_mask_triple_embedded_classnames(
        rgb_img, label_img, id_to_class_name_map, save_fpath="dummy.jpg", save_to_disk=False
    )
    pred_hgrid = form_mask_triple_embedded_classnames(
        rgb_img, pred, id_to_class_name_map, save_fpath="dummy.jpg", save_to_disk=False
    )
    vstack_img = form_vstacked_imgs(img_list=[label_hgrid, pred_hgrid], vstack_save_fpath=save_fpath, save_to_disk=True)


def form_mask_triple_embedded_classnames(
    rgb_img: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Dict[int, str],
    save_fpath: str,
    save_to_disk: bool = False,
) -> np.ndarray:
    """
    Args:
    -   rgb_img:
    -   label_img:
    -   id_to_class_name_map
    -   save_fpath
    -   save_to_disk
    Returns:
    -   Array, representing 3 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks, Semantic Masks
    """
    rgb_with_mask = convert_instance_img_to_mask_img(label_img, rgb_img.copy())

    # or can do max cardinality conn comp of each class
    rgb2 = save_classnames_in_image_sufficientpx(rgb_with_mask, label_img, id_to_class_name_map)
    mask_img = convert_instance_img_to_mask_img(label_img, img_rgb=None)
    rgb3 = save_classnames_in_image_sufficientpx(mask_img, label_img, id_to_class_name_map)
    return form_hstacked_imgs([rgb_img, rgb2, rgb3], save_fpath, save_to_disk)


def convert_instance_img_to_mask_img(instance_img: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given a grayscale image where intensities denote instance IDs (same intensity denotes
    belonging to same instance), convert this to an RGB image where all pixels corresponding
    to the same instance get the same color. Note that two instances may not have unique colors,
    do to a finite-length colormap.
        Args:
        -   instance_img: Numpy array of shape (M,N), representing grayscale image, in [0,255]
        -   img_rgb: Numpy array representing RGB image, possibly blank, in [0,255]
        Returns:
        -   img_rgb:
    """
    img_h, img_w = instance_img.shape
    if img_rgb is None:
        img_rgb = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    assert instance_img.dtype in [
        np.uint8,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
    ], "Label map is not composed of integers."
    assert img_rgb.dtype in [np.uint8, np.uint16]
    our_colormap = colormap(rgb=True)
    num_unique_colors = our_colormap.shape[0]
    # np.unique will always sort the values
    if np.unique(instance_img).size > 0:
        for i, instance_id in enumerate(np.unique(instance_img)):
            col = our_colormap[(instance_id) % num_unique_colors]
            mask = instance_img == instance_id
            img_rgb = vis_mask(img_rgb, mask, col, alpha=0.4)
    return img_rgb


def save_classnames_in_image_sufficientpx(
    rgb_img: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Dict[int, str],
    font_color=(0, 0, 0),
    save_to_disk: bool = False,
    save_fpath: str = "",
    min_conncomp_px: int = 4000,
    font_scale: int = 1,
):
    """
    Write a classname over each connected component of a label
    map as long as the connected component has a sufficiently
    large number of pixels (specified as argument).
    Args:
        rgb_img: Numpy array (H,W,3) representing RGB image
        label_img: Numpy array (H,W) representing label map
        id_to_class_name_map: mapping from class ID to classname
        font_color: 3-tuple representing RGB font color
        save_to_disk: whether to save image to disk
        save_fpath: absolute file path
        min_conncomp_px: minimum number of pixels to justify
            placing a text label over connected component
        font_scale: scale of font text
    Returns:
        rgb_img: Numpy array (H,W,3) with embedded classanmes
    """
    H, W, C = rgb_img.shape
    class_to_conncomps_dict = scipy_conn_comp(label_img)

    for class_idx, conncomps_list in class_to_conncomps_dict.items():
        for conncomp in conncomps_list:
            if conncomp.sum() < min_conncomp_px:
                continue
            text = id_to_class_name_map[class_idx]

            y, x = get_mean_mask_location(conncomp)
            x -= 55  # move the text so approx. centered over mask.
            x = max(0, x)
            x = min(W - 1, x)

            # jitter location if nonconvex object mean not within its mask
            if conncomp[y, x] != 1:
                x, y = search_jittered_location_in_mask(x, y, conncomp)

            # print(f'Class idx: {class_idx}: (x,y)=({x},{y})')
            rgb_img = add_text_cv2(
                rgb_img, text, coords_to_plot_at=(x, y), font_color=font_color, font_scale=font_scale, thickness=2
            )

    if save_to_disk:
        cv2_write_rgb(save_fpath, rgb_img)

    return rgb_img


def scipy_conn_comp(img: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """
        labelsndarray of dtype int
        Labeled array, where all connected regions
        are assigned the same integer value.
        numint, optional
        Number of labels, which equals the maximum label index
        and is only returned if return_num is True.
    Args:
       img:
    Returns:
       class_to_conncomps_dict:
    """
    class_to_conncomps_dict = defaultdict(list)
    present_class_idxs = np.unique(img)

    for class_idx in present_class_idxs:
        structure = np.ones((3, 3), dtype=np.int32)  # this defines the connection filter
        instance_img, nr_objects = scipy.ndimage.label(input=img == class_idx, structure=structure)

        for i in np.unique(instance_img):
            if i == 0:  # 0 doesn't count as an instance ID
                continue
            bin_arr = (instance_img == i).astype(np.uint8)
            class_to_conncomps_dict[class_idx] += [bin_arr]

    return class_to_conncomps_dict


def search_jittered_location_in_mask(mean_x: float, mean_y: float, conncomp: np.ndarray) -> Tuple[int, int]:
    """
    For visualizing classnames in an image.
    When we wish to place text over a mask, for nonconvex regions, we cannot
    use mask pixel mean location (may not fall within mask), so we will
    jitter the location until we find a valid location within mask.
    """
    H, W = conncomp.shape
    num_attempts = 100
    for i in range(num_attempts):
        # grow the jitter up to half width of image at end
        SCALE = ((i + 1) / num_attempts) * (W / 2)
        dx, dy = np.random.randn(2) * SCALE
        # print(f'On iter {i}, mul noise w/ {SCALE} to get dx,dy={dx},{dy}')
        x = int(mean_x + dx)
        y = int(mean_y + dy)

        # Enforce validity
        x = max(0, x)
        x = min(W - 1, x)
        y = max(0, y)
        y = min(H - 1, y)

        if conncomp[y, x] != 1:
            continue
        else:
            return x, y

    return mean_x, mean_y


def form_hstacked_imgs(img_list: List[np.ndarray], hstack_save_fpath: str, save_to_disk: bool = True) -> np.ndarray:
    """
    Concatenate images along a horizontal axis and save them.
    Accept RGB images, and convert to BGR for OpenCV to save them.

    Args:
        img_list: list of Numpy arrays e.g. representing different RGB visualizations of same image,
            must all be of same height
        hstack_save_fpath: string, representing file path

    Returns:
        hstack_img: Numpy array representing RGB image, containing horizontally stacked images as tiles.
    """
    img_file_type = Path(hstack_save_fpath).suffix
    assert img_file_type in [".jpg", ".png"]
    # create_leading_fpath_dirs(hstack_save_fpath)

    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # height and number of channels must match
    assert all(img.shape[0] == img_h for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)

    all_widths = [img.shape[1] for img in img_list]
    hstack_img = np.zeros((img_h, sum(all_widths), 3), dtype=np.uint8)

    running_w = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_w
        end = start + w
        hstack_img[:, start:end, :] = img
        running_w += w

    if save_to_disk:
        cv2.imwrite(hstack_save_fpath, hstack_img[:, :, ::-1])
    return hstack_img


def form_vstacked_imgs(img_list: List[np.ndarray], vstack_save_fpath: str, save_to_disk: bool = True) -> np.ndarray:
    """
    Concatenate images along a vertical axis and save them.
    Accept RGB images, and convert to BGR for OpenCV to save them.

    Args:
        img_list: list of Numpy arrays representing different RGB visualizations of same image,
            must all be of same shape
        hstack_save_fpath: string, representing file path

    Returns:
        hstack_img: Numpy array representing RGB image, containing vertically stacked images as tiles.
    """
    img_file_type = Path(vstack_save_fpath).suffix
    assert img_file_type in [".jpg", ".png"]

    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # width and number of channels must match
    assert all(img.shape[1] == img_w for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)
    all_heights = [img.shape[0] for img in img_list]
    vstack_img = np.zeros((sum(all_heights), img_w, 3), dtype=np.uint8)

    running_h = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_h
        end = start + h
        vstack_img[start:end, :, :] = img
        running_h += h

    if save_to_disk:
        cv2.imwrite(vstack_save_fpath, vstack_img[:, :, ::-1])
    return vstack_img


def add_text_cv2(
    img: np.ndarray, text: str, coords_to_plot_at=None, font_color=(0, 0, 0), font_scale=1, thickness=2
) -> np.ndarray:
    """
    font_color = (0,0,0)
    x: x-coordinate from image origin to plot text at
    y: y-coordinate from image origin to plot text at
    """
    corner_offset = 5
    font = cv2.FONT_HERSHEY_TRIPLEX  # cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w, _ = img.shape

    if img_h < (corner_offset + 1) or img_w < (corner_offset + 1):
        return

    if coords_to_plot_at is None:
        coords_to_plot_at = (corner_offset, img_h - 1 - corner_offset)

    line_type = 2
    img = cv2.putText(
        img=img,
        text=text,
        org=coords_to_plot_at,
        fontFace=font,
        fontScale=font_scale,
        color=font_color,
        thickness=thickness,
        lineType=line_type,
    )
    return img


def colormap(rgb: bool = False):
    """
    Create an array of visually distinctive RGB values.
    Args:
        rgb: boolean, whether to return in RGB or BGR order. BGR corresponds to OpenCV default.
    Returns:
        color_list: Numpy array of dtype uin8 representing RGB color palette.
    """
    color_list = np.array(
        [
            [252, 233, 79],
            # [237, 212, 0],
            [196, 160, 0],
            [252, 175, 62],
            # [245, 121, 0],
            [206, 92, 0],
            [233, 185, 110],
            [193, 125, 17],
            [143, 89, 2],
            [138, 226, 52],
            # [115, 210, 22],
            [78, 154, 6],
            [114, 159, 207],
            # [52, 101, 164],
            [32, 74, 135],
            [173, 127, 168],
            # [117, 80, 123],
            [92, 53, 102],
            [239, 41, 41],
            # [204, 0, 0],
            [164, 0, 0],
            [238, 238, 236],
            # [211, 215, 207],
            # [186, 189, 182],
            [136, 138, 133],
            # [85, 87, 83],
            [46, 52, 54],
        ]
    ).astype(np.uint8)
    assert color_list.shape[1] == 3
    assert color_list.ndim == 2

    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def vis_mask(img: np.ndarray, mask: np.ndarray, col: Tuple[int,int,int], alpha: float=0.4) -> np.ndarray:
    """
    Visualizes a single binary mask by coloring the region inside a binary mask
    as a specific color, and then blending it with an RGB image.
    
    Args:
        img: Numpy array, representing RGB image with values in the [0,255] range
        mask: Numpy integer array, with values in [0,1] representing mask region
        col: color, tuple of integers in [0,255] representing RGB values
        alpha: blending coefficient (higher alpha shows more of mask, 
           lower alpha preserves original image)
    Returns:
        image: Numpy array, representing an RGB image, representing a blended image
            of original RGB image and specified colors in mask region.
    """
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    return img.astype(np.uint8)

def get_mean_mask_location(mask: np.ndarray):
    """ Given a binary mask, find the mean location for entries equal to 1.

    Args:
        mask
    Returns:
        coordinate of mean pixel location as (x,y)
    """
    coords = np.vstack(np.where(mask == 1)).T
    return np.mean(coords, axis=0).astype(np.int32)

