import pytest
import numpy as np
import logging
import cv2
from pathlib import Path
from vision.part3_ransac import (
    calculate_num_ransac_iterations,
    ransac_fundamental_matrix,
)
from vision.utils import load_image, get_matches

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_calculate_num_ransac_iterations():
    data_set = [
        (0.99, 1, 0.99, 1),
        (0.99, 10, 0.9, 11),
        (0.9, 15, 0.5, 75450),
        (0.95, 5, 0.66, 22),
    ]

    for prob_success, sample_size, ind_prob, num_samples in data_set:
        S = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob)
        assert pytest.approx(num_samples, abs=1.0) == S


def test_ransac_fundamental_matrix():
    np.random.seed(0)
    pic_a = load_image(f"{DATA_ROOT}/argoverse_log_273c1883/ring_front_center_315975640448534784.jpg")
    scale_a = 0.5
    pic_b = load_image(f"{DATA_ROOT}/argoverse_log_273c1883/ring_front_center_315975643412234000.jpg")
    scale_b = 0.5
    n_feat = 4e3
    pic_a = cv2.resize(pic_a, None, fx=scale_a, fy=scale_a)
    pic_b = cv2.resize(pic_b, None, fx=scale_b, fy=scale_b)
    points_2d_pic_a, points_2d_pic_b = get_matches(pic_a, pic_b, n_feat)
    F, _, _ = ransac_fundamental_matrix(points_2d_pic_a, points_2d_pic_b)
    expected_F = np.array(
        [
            [1.85972504e-06, 1.13259293e-04, -3.48078668e-02],
            [-1.18639791e-04, -4.47303499e-06, 7.50543631e-02],
            [3.35992820e-02, -2.12300033e-02, -1.45713450e01],
        ]
    )

    # compare up to scale ambiguity
    F /= F[2, 2]
    expected_F /= expected_F[2, 2]

    assert np.allclose(F, expected_F, atol=1e-2)
