from data import DFLogoDetectionDataset
import os
import numpy as np
from glob import glob


def test_dataset(dataset):
    assert dataset is not None


def test_data_dir(data_dir):
    assert os.path.exists(data_dir), "Data directory does not exist"


def test_data_dir_contains_images(data_dir):
    assert len(glob(os.path.join(
        data_dir, "**/*.jpg"
    ))) > 0, "Data directory is empty"


def test_labels_file_exists(labels_file):
    assert os.path.exists(labels_file), "Labels file does not exist"


def test_init(data_dir, labels_file):
    # Test that the dataset is initialized correctly
    dataset = DFLogoDetectionDataset(data_dir, labels_file)
    assert dataset.data_dir == data_dir, "Unexpected data_dir value"
    assert dataset.labels.columns == ["Brand Name", "x", "y", "width", "height",
                                      "filename", "image_width", "image_height", 'Brand Name (Encoded)'], "Unexpected columns in labels DataFrame"
    assert len(dataset) == len(
        dataset.labels), "Unexpected number of samples in the dataset"


def test_getitem(data_dir, labels_file):
    # Test that the __getitem__ method returns a sample with the correct image and label data
    dataset = DFLogoDetectionDataset(data_dir, labels_file)
    sample = dataset[0]
    assert isinstance(
        sample["image"], np.ndarray), "Unexpected type for image data"
    assert sample["brand_name"] == dataset.labels.iloc[0]["Brand Name"], "Unexpected brand_name value"


def test_transform(data_dir, labels_file):
    from tf import get_transforms
    transform, _ = get_transforms()
    # Test that the __getitem__ method applies the transform correctly
    dataset = DFLogoDetectionDataset(
        data_dir, labels_file, transform=transform)
    sample = dataset[0]
    assert sample["image"] == transform(
        dataset.labels.iloc[0]["image"]), "Unexpected image data after applying transform"
