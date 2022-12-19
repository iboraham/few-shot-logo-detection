import pytest
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LogoDetectionModel
from data import LogoDetectionDataset


@pytest.fixture
def model():
    return LogoDetectionModel


def test_model(model):
    assert model is not None


@pytest.fixture
def dataset():
    return LogoDetectionDataset


def test_dataset(dataset):
    assert dataset is not None
