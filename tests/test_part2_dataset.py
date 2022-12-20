
import pdb
from pathlib import Path

import numpy as np
import torch

from src.vision.part2_dataset import SemData, make_dataset
from src.vision.cv2_transforms import ToTensor


TEST_DATA_ROOT = Path(__file__).resolve().parent / "test_data"

def test_make_dataset() -> None:
	"""Ensure make_dataset() returns the proper outputs"""
	split = "train"
	data_root = "/home/dummy_data_root"
	data_list_fpath = str(TEST_DATA_ROOT / "dummy_camvid_train.txt")
	image_label_list = make_dataset(split, data_root, data_list_fpath)

	expected_image_label_list = [
		(f"{data_root}/701_StillsRaw_full/0001TP_006690.png", f"{data_root}/semseg11/0001TP_006690_L.png"),
		(f"{data_root}/701_StillsRaw_full/0001TP_006720.png", f"{data_root}/semseg11/0001TP_006720_L.png"),
		(f"{data_root}/701_StillsRaw_full/0001TP_006750.png", f"{data_root}/semseg11/0001TP_006750_L.png")
	]
	assert image_label_list == expected_image_label_list


def test_getitem_no_data_aug() -> None:
	"""Ensure SemData __getitem__() works properly, when transform is only ToTensor (no data augmentation)."""
	split = "train"
	data_root = str(TEST_DATA_ROOT / "CamvidSubsampled")
	data_list_fpath = str(TEST_DATA_ROOT / "dummy_camvid_train.txt")
	dataset = SemData(split, data_root, data_list_fpath, transform=ToTensor())
	image, label = dataset[2]

	assert isinstance(image, torch.Tensor)
	assert isinstance(label, torch.Tensor)

	# should be in CHW order now
	assert image.shape == (3,720,960)
	assert image.dtype == torch.float32
	assert np.isclose(image.mean().item(), 47.6283, atol=1e-2)

	assert label.shape == (720, 960)
	assert label.dtype == torch.int64
	assert label.sum().item() == 32121290


# def test_getitem_transform() -> None:
# 	"""Ensure SemData __getitem__() works properly, when transform is not None.
# 	"""
# 	data_list_fpath
# 	dataset = SemData(split: str, data_root: str, data_list_fpath: str, transform=None)
# 	image, label = dataset[2]
# 	assert isinstance(image, torch.Tensor)
# 	assert isinstance(label, torch.Tensor)
# 	pdb.set_trace()



def test_SemData_len() -> None:
	""" Ensure length of dataset is properly generated. This essentially tests make_dataset() """
	split = "train"
	data_root = str(TEST_DATA_ROOT / "CamvidSubsampled")
	data_list_fpath = str(TEST_DATA_ROOT / "dummy_camvid_train.txt")
	dataset = SemData(split, data_root, data_list_fpath, transform=ToTensor())

	assert len(dataset) == 3
