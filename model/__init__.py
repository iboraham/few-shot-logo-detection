import torch
from .helpers import set_requires_grad, create_model, save_model, load_model


class LogoDetectionModel(torch.nn.Module):
    """
    A model for logo detection. This model is a wrapper around a pretrained
    ResNet18 model. The last fully connected layer is replaced with a new one
    that predicts the number of classes.

    @TODO: Add support for other pretrained models.

    Args:
        num_classes (int): Number of classes to predict.
    """

    def __init__(self, num_classes: int):

        super(LogoDetectionModel, self).__init__()
        self.model: torch.nn.Module = create_model(num_classes)
        set_requires_grad(self.model, requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Perform classification on the extracted features
        return self.model(images)

    def save(self, path: str) -> None:
        save_model(self, path)

    @staticmethod
    def load(path: str, num_classes: int) -> "LogoDetectionModel":
        return load_model(path, num_classes)
