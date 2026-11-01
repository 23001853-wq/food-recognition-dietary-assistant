from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def preprocessing(name):
    return {
        "resize": 256,
        "crop": 224,
        "interpolation": "bicubic" if name == "efficientnet_b0" else "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def build_transform(config, train=False):
    interpolation = InterpolationMode(config["interpolation"])
    spatial = (
        [T.RandomResizedCrop(config["crop"], interpolation=interpolation), T.RandomHorizontalFlip()]
        if train
        else [T.Resize(config["resize"], interpolation=interpolation), T.CenterCrop(config["crop"])]
    )
    return T.Compose([*spatial, T.ToTensor(), T.Normalize(config["mean"], config["std"])])
