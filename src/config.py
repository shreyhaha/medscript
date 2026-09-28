"""
config.py - every path and setting in ONE place.

Why: if you want to change the image size or a folder name, you change it here
once, instead of hunting through every file. (Good software-engineering habit
to mention in the interview.)
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # the medscript/ folder

DATA_DIR = ROOT / "data" / "raw"
HANDWRITING_DIR = DATA_DIR / "handwriting"             # Training/, Validation/, Testing/
MEDICINE_DB_PATH = DATA_DIR / "A_Z_medicines_dataset_of_India.csv"

MODELS_DIR = ROOT / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.joblib"
CNN_MODEL_PATH = MODELS_DIR / "cnn.pt"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# The three splits exactly as the dataset ships them.
#   split name -> (sub-folder, labels csv, images folder)
SPLITS = {
    "train": ("Training", "training_labels.csv", "training_words"),
    "val": ("Validation", "validation_labels.csv", "validation_words"),
    "test": ("Testing", "testing_labels.csv", "testing_words"),
}

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
# Handwritten words are wide and short (median ~54 x 157 px, ratio ~1:3),
# so we use a wide canvas instead of a square one. This avoids squashing
# the letters.
BASELINE_IMG_SIZE = (64, 192)   # (height, width) for the HOG baseline
CNN_IMG_SIZE = (96, 288)        # a bit bigger for the CNN (same 1:3 ratio)

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 3e-4
CNN_PATIENCE = 5                # early stopping: stop if val accuracy
                                # does not improve for this many epochs

# ---------------------------------------------------------------------------
# Human-in-the-loop setting
# ---------------------------------------------------------------------------
# If the model's confidence is below this, the app says
# "Uncertain - please ask a pharmacist to verify".
# evaluate.py prints a table to help you choose this value.
CONFIDENCE_THRESHOLD = 0.60
TOP_K = 3                       # how many suggestions to show
