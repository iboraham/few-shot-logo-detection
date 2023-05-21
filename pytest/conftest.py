try:
    from model import LogoDetectionModel
    from data import LogoDetectionDataset
except ModuleNotFoundError:
    import sys
    import os

    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from model import LogoDetectionModel
    from data import LogoDetectionDataset
import pytest


@pytest.fixture
def model():
    return LogoDetectionModel


@pytest.fixture
def dataset():
    return LogoDetectionDataset


@pytest.fixture
def data_dir():
    return os.path.join("data", "game_stats-1", "train")
