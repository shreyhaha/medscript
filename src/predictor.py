"""
predictor.py - one simple interface for BOTH models.

    p = Predictor()                  # CNN if trained, otherwise the baseline
    probs = p.predict_proba([gray])  # (n_images, 78) probabilities

The app and evaluate.py only talk to this class, so they don't care which
model is underneath. (Software-engineering idea: a common interface.)
"""
import joblib
import numpy as np

from src import config
from src.features import hog_matrix
from src.preprocessing import preprocess


def torch_available():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


class BaselinePredictor:
    name = "HOG + Logistic Regression (baseline)"

    def __init__(self, path=config.BASELINE_MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.hog_cell = bundle["hog_cell"]
        self.img_size = tuple(bundle["img_size"])

    def predict_proba(self, grays):
        clean = [preprocess(g, self.img_size) for g in grays]
        return self.model.predict_proba(hog_matrix(clean, self.hog_cell))


class CNNPredictor:
    name = "ResNet-18 CNN (transfer learning)"

    def __init__(self, path=config.CNN_MODEL_PATH):
        import torch

        from src.cnn_model import load_cnn
        self.torch = torch
        self.model, self.classes = load_cnn(path)
        self.classes = list(self.classes)

    def predict_proba(self, grays, batch_size=64):
        from src.cnn_model import gray_to_input
        out = []
        with self.torch.no_grad():
            for i in range(0, len(grays), batch_size):
                x = self.torch.stack([gray_to_input(g) for g in grays[i:i + batch_size]])
                out.append(self.torch.softmax(self.model(x), dim=1).numpy())
        return np.concatenate(out)


def Predictor(prefer="best"):
    """Factory: prefer='best' -> CNN if available, else baseline.
    prefer='baseline' or 'cnn' forces one."""
    cnn_ready = config.CNN_MODEL_PATH.exists() and torch_available()
    if prefer == "cnn" or (prefer == "best" and cnn_ready):
        if not cnn_ready:
            raise FileNotFoundError("No trained CNN found (run python -m src.train_cnn) "
                                    "or PyTorch is not installed.")
        return CNNPredictor()
    if not config.BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python -m src.train_baseline")
    return BaselinePredictor()


def available_models():
    """Names of the models that are trained and loadable on this computer."""
    names = []
    if config.BASELINE_MODEL_PATH.exists():
        names.append("baseline")
    if config.CNN_MODEL_PATH.exists() and torch_available():
        names.append("cnn")
    return names
