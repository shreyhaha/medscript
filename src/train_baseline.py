"""
train_baseline.py - the classic machine-learning baseline.

    python -m src.train_baseline        (about 3-5 minutes on a laptop CPU)

Model:  cleaned image -> HOG features -> StandardScaler -> Logistic Regression

Why start with a baseline?
  - It tells us how hard the problem is. If a simple model already gets 95%,
    a deep network is overkill. If it gets 75%, there is room to improve.
  - It gives the CNN something to BEAT. "My CNN got 90%" means nothing on its
    own; "the baseline got 80%, the CNN got 90%" is a real result.

Why Logistic Regression?
  - Fast, well understood, gives PROBABILITIES (needed for our confidence
    check), and C (regularisation strength) is its only important setting.
  - We also tried an RBF-kernel SVM: same accuracy, 4x slower, so we kept LR.

What this script does:
  1. Ablation study: how much does each preprocessing step help?
  2. Grid search on the VALIDATION set for HOG cell size and C.
  3. Saves the best model to models/baseline.joblib.
The TEST set is NOT used here - that happens once, in evaluate.py.
"""
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import get_class_names, load_all
from src.features import hog_matrix
from src.preprocessing import load_gray, preprocess


def make_model(C):
    # StandardScaler: puts every feature on the same scale (mean 0, std 1).
    # Logistic regression is trained with gradient-based optimisation and
    # its regularisation treats all features equally, so scale matters.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def main():
    size = config.BASELINE_IMG_SIZE
    train, val, _ = load_all()
    classes = get_class_names(train)
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_train = train["medicine"].map(class_to_id).values
    y_val = val["medicine"].map(class_to_id).values

    raw_train = [load_gray(p) for p in train["image_path"]]
    raw_val = [load_gray(p) for p in val["image_path"]]

    # ---------------------------------------------------------------------
    # 1. Ablation study: switch preprocessing steps on/off, same model.
    # ---------------------------------------------------------------------
    print("1) Ablation study - does the image cleaning help? (C=0.03, cell=8)")
    variants = {
        "only resize (no cleaning)": dict(clean=False),
        "textbook: median + hard Otsu": dict(clean=True, use_median=True, dilate=False),
        "no median + hard Otsu": dict(clean=True, use_median=False, dilate=False),
        "FINAL: no median + dilated Otsu": dict(clean=True, use_median=False, dilate=True),
    }
    ablation = []
    for name, kwargs in variants.items():
        X_tr = hog_matrix([preprocess(g, size, **kwargs) for g in raw_train])
        X_va = hog_matrix([preprocess(g, size, **kwargs) for g in raw_val])
        acc = accuracy(make_model(0.03).fit(X_tr, y_train), X_va, y_val)
        ablation.append({"preprocessing": name, "val_accuracy": round(acc, 4)})
        print(f"   {name:35s} val accuracy = {acc:.3f}")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation).to_csv(config.REPORTS_DIR / "ablation_preprocessing.csv", index=False)

    # ---------------------------------------------------------------------
    # 2. Grid search on the validation set (final preprocessing).
    # ---------------------------------------------------------------------
    print("\n2) Grid search (choose settings using the VALIDATION set)")
    clean_train = [preprocess(g, size) for g in raw_train]
    clean_val = [preprocess(g, size) for g in raw_val]
    results, best = [], None
    for cell in [8, 12]:
        X_tr, X_va = hog_matrix(clean_train, cell), hog_matrix(clean_val, cell)
        for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
            start = time.time()
            model = make_model(C).fit(X_tr, y_train)
            acc = accuracy(model, X_va, y_val)
            results.append({"hog_cell": cell, "C": C, "n_features": X_tr.shape[1],
                            "val_accuracy": round(acc, 4)})
            print(f"   cell={cell:2d}  C={C:<6}  features={X_tr.shape[1]:5d}  "
                  f"val acc={acc:.3f}  ({time.time() - start:.0f}s)")
            if best is None or acc > best["acc"]:
                best = {"acc": acc, "cell": cell, "C": C, "model": model}
    pd.DataFrame(results).to_csv(config.REPORTS_DIR / "baseline_grid_search.csv", index=False)

    # ---------------------------------------------------------------------
    # 3. Save the best model (+ everything needed to use it later).
    # ---------------------------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best["model"],
        "classes": classes,
        "hog_cell": best["cell"],
        "img_size": size,
        "val_accuracy": best["acc"],
    }, config.BASELINE_MODEL_PATH)
    print(f"\nBest: HOG cell={best['cell']}, C={best['C']} -> val accuracy {best['acc']:.3f}")
    print(f"Saved to {config.BASELINE_MODEL_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
