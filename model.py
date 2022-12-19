import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary


class LogoDetectionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(LogoDetectionModel, self).__init__()

        # Use a pretrained backbone such as ResNet or VGG
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the weights of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Drop the last classification layer of the backbone

        # Replace the classification layer of the backbone with
        # a new layer that has the desired number of classes
        self.backbone.fc = torch.nn.Linear(
            in_features=self.backbone.fc.in_features, out_features=num_classes
        )

    def forward(self, images):
        # Use the backbone to extract features from the input images
        features = self.backbone(images)

        # Perform classification on the extracted features
        return self.backbone.fc(features)


if __name__ == "__main__":
    # Create an instance of the model with the desired number of classes
    model = LogoDetectionModel(num_classes=10)
    print(model)
    print(summary(model, (3, 256, 256)))
