import torch
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


def create_model(name, num_classes, pretrained=False):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported CNN: {name}")
    return model
