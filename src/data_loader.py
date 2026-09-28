"""
data_loader.py - reads the dataset's CSV files and finds the images.

The dataset ships as three splits:
    Training   (3,120 images) -> the model LEARNS from these
    Validation (  780 images) -> we use these to CHOOSE settings
                                 (e.g. which C value, which epoch was best)
    Testing    (  780 images) -> touched ONCE at the very end to report
                                 the final, honest score

Each split has a CSV with 3 columns: IMAGE, MEDICINE_NAME, GENERIC_NAME.
"""
import hashlib

import pandas as pd

from src import config


def find_handwriting_dir():
    """Return the folder that contains Training/, Validation/, Testing/.

    The zip from Kaggle has a strange folder name
    ("Doctor#U2019s Handwritten Prescription BD dataset"), so if you did not
    rename it to `handwriting`, we search data/raw/ for it automatically.
    """
    if (config.HANDWRITING_DIR / "Training").exists():
        return config.HANDWRITING_DIR
    for csv in config.DATA_DIR.rglob("training_labels.csv"):
        return csv.parent.parent
    raise FileNotFoundError(
        f"Could not find the handwriting dataset inside {config.DATA_DIR}.\n"
        "Unzip the Kaggle dataset there (see README, step 2)."
    )


def load_split(split):
    """Load one split ('train', 'val' or 'test') as a DataFrame.

    Columns: image_path, medicine, generic
    """
    folder, csv_name, img_folder = config.SPLITS[split]
    base = find_handwriting_dir() / folder
    df = pd.read_csv(base / csv_name)
    df = df.rename(columns={"MEDICINE_NAME": "medicine", "GENERIC_NAME": "generic"})
    df["image_path"] = df["IMAGE"].apply(lambda name: str(base / img_folder / name))
    return df[["image_path", "medicine", "generic"]]


def get_class_names(train_df):
    """Sorted list of the 78 medicine names. Index in this list = class id."""
    return sorted(str(m) for m in train_df["medicine"].unique())


def get_brand_to_generic(train_df):
    """Dictionary: brand name -> generic (active ingredient) name."""
    return dict(zip(train_df["medicine"], train_df["generic"]))


def file_hash(path):
    """MD5 fingerprint of a file. Two identical files -> identical hash."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_leaked_duplicates(eval_df, train_df, verbose=True):
    """Drop images from val/test that are EXACT copies of a training image.

    Why this matters (DATA LEAKAGE): if the exact same picture is in both
    training and test, the model has literally seen the answer. Its test
    score would be slightly too optimistic. We found 22 such copies in the
    test set (and 27 in validation), so we remove them for an honest evaluation.
    """
    train_hashes = {file_hash(p) for p in train_df["image_path"]}
    is_dup = eval_df["image_path"].apply(lambda p: file_hash(p) in train_hashes)
    if verbose and is_dup.any():
        print(f"  Removed {is_dup.sum()} images that also appear in the training set (leakage).")
    return eval_df[~is_dup].reset_index(drop=True)


def load_all(remove_duplicates=True):
    """Load train/val/test in one call. Returns (train, val, test)."""
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    if remove_duplicates:
        val = remove_leaked_duplicates(val, train)
        test = remove_leaked_duplicates(test, train)
    return train, val, test
