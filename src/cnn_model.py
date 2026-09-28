"""
cnn_model.py - the deep-learning model, shared by training AND the app.

Why one shared file? The image must be converted to numbers in EXACTLY the
same way during training and when the app makes a prediction. If the two
drift apart ("training/serving skew"), accuracy silently drops in production.

MODEL: ResNet-18 with TRANSFER LEARNING
  - ResNet-18 is a Convolutional Neural Network (CNN) with 18 layers.
    Early layers learn simple patterns (edges, curves); deeper layers combine
    them into shapes (loops, letters, letter-pairs).
  - "Residual" = skip connections: each block adds its input to its output.
    This lets gradients flow through deep networks, so they train well.
  - It was PRE-TRAINED on ImageNet (1.2 million photos). Its early layers
    already know how to detect edges and strokes. We keep that knowledge and
    only re-train it on our 3,120 handwriting images. That is transfer
    learning, and it is why a deep model works with such a small dataset.
  - We replace the last layer (1000 ImageNet classes) with a new one that
    outputs our 78 medicine classes.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src import config
from src.preprocessing import preprocess

# ImageNet statistics: the pre-trained network expects inputs normalised
# with the same mean/std it saw during its original training.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_model(num_classes, pretrained=True):
    """ResNet-18 with a new final layer for our classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def image_to_tensor(clean_img):
    """Preprocessed uint8 image (H, W) -> tensor (1, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(clean_img)).float().div(255.0).unsqueeze(0)


def normalise(x):
    """(1, H, W) grey tensor -> (3, H, W) normalised tensor.

    ResNet expects 3 colour channels, so we copy the grey channel 3 times.
    """
    return (x.repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD


def gray_to_input(gray):
    """Raw grey image -> model-ready tensor (3, H, W). Used by the app."""
    clean = preprocess(gray, config.CNN_IMG_SIZE)
    return normalise(image_to_tensor(clean))


def get_device():
    """Use a GPU if there is one (NVIDIA 'cuda' or Apple 'mps'), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cnn(path=config.CNN_MODEL_PATH):
    """Load a trained model saved by train_cnn.py. Returns (model, classes)."""
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # switch off training-only behaviour (batch-norm updates)
    return model, checkpoint["classes"]
