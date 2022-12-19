import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary


class LogoDetectionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(LogoDetectionModel, self).__init__()

        # Use a pretrained backbone such as ResNet or VGG
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the weights of the backbone so that they are not updated
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new one
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, images):
        # Perform classification on the extracted features
        return self.model(images)


if __name__ == "__main__":
    # Create an instance of the model with the desired number of classes
    model = LogoDetectionModel(num_classes=10)
    print(summary(model, (3, 256, 256)))
