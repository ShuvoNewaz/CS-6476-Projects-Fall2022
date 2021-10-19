import os
from pathlib import Path

import numpy as np
from vision.stats_helper import compute_mean_and_std

PROJ_ROOT = Path(__file__).resolve().parent


def test_mean_and_variance():
    mean, std = compute_mean_and_std(f"{PROJ_ROOT}/small_data/")

    assert np.allclose(mean, 0.46178912, atol=1e-3)
    assert np.allclose(std, 0.25604102, atol=1e-3)
