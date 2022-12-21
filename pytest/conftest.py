try:
    from model import LogoDetectionModel
    from data import DFLogoDetectionDataset
except ModuleNotFoundError:
    import sys
    import os

    # Add the parent directory to the path
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from model import LogoDetectionModel
    from data import DFLogoDetectionDataset
import pytest


@pytest.fixture
def model():
    return LogoDetectionModel


@pytest.fixture
def dataset():
    return DFLogoDetectionDataset


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")


@pytest.fixture
def labels_file(data_dir):
    return os.path.join(data_dir, "labels_logo-detection.csv")
