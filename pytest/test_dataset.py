from data import LogoDetectionDataset
import os
import numpy as np
from glob import glob


def test_dataset(dataset):
    assert dataset is not None


def test_data_dir(data_dir):
    assert os.path.exists(data_dir), f"Data directory does not exist {data_dir}"


def test_data_dir_contains_images(data_dir):
    assert (
        len(glob(os.path.join(data_dir, "*.jpg"))) > 0
    ), f"Data directory is empty {data_dir}"


def test_labels_file_exists(data_dir):
    assert os.path.exists(
        os.path.join(data_dir, "_annotations.coco.json")
    ), "Labels file does not exist"
