import torch
import torchvision
from torchvision.models import detection
from torchvision.models.detection import RetinaNet, RetinaNet_ResNet50_FPN_Weights


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad: bool = requires_grad


def create_model(num_classes: int) -> torch.nn.Module:
    # Use a pretrained backbone such as ResNet or VGG
    model: torchvision.models.detection.RetinaNet = detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
        progress=False,
    )
    print(model)

    return model


def save_model(model: torch.nn.Module, path: str) -> None:
    # Save the model parameters
    torch.save(model.state_dict(), path)


def load_model(path: str, num_classes: int) -> torch.nn.Module:
    # Create a new model and load the parameters
    model: torch.nn.Module = create_model(num_classes)
    model.load_state_dict(torch.load(path))

    return model
