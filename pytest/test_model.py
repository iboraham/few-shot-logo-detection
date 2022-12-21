import pytest
import torch


def test_model(model):
    assert model is not None


@pytest.mark.parametrize("num_classes", [10, 100, 1000])
def test_logo_detection_model(num_classes, model):
    # Create a logo detection model instance
    model = model(num_classes)

    # Create random input data
    batch_size = 16
    images = torch.randn(batch_size, 3, 256, 256)

    # Perform a forward pass through the model
    predictions = model(images)

    # Verify that the output of the model has the expected shape
    assert predictions.shape == (batch_size, num_classes)
