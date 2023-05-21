from torchsummary import summary

from model import LogoDetectionModel

# Create an instance of the model with the desired number of classes
model = LogoDetectionModel(num_classes=10)
print(summary(model, (3, 256, 256)))
