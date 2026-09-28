"""
eda.py - Exploratory Data Analysis. Run this FIRST.

    python -m src.eda

What it answers (and saves as figures in reports/figures/):
  1. How many images per split and per medicine?   -> is the data balanced?
  2. How big are the images?                         -> what size to resize to?
  3. What do they look like?                         -> sample grid
  4. What does my cleaning pipeline do?              -> before/after figure
  5. Are there duplicate images across splits?       -> data leakage check
  6. How many brands share one active ingredient?    -> why we also report
                                                        "generic-level" accuracy
"""
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loader import file_hash, load_all
from src.preprocessing import load_gray, preprocess, preprocess_steps


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(config.ROOT)}")


def main():
    print("Loading data (duplicates NOT removed yet, so we can count them)...")
    train, val, test = load_all(remove_duplicates=False)

    # 1. Split sizes and class balance --------------------------------------
    print("\n1) Split sizes")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(df):5d} images, {df['medicine'].nunique()} medicines")
    counts = train["medicine"].value_counts()
    print(f"  Images per medicine in train: min={counts.min()}, max={counts.max()}")
    print("  -> Perfectly BALANCED, so plain accuracy is a fair metric here.")

    # 6. Brands per generic --------------------------------------------------
    brands_per_generic = train.groupby("generic")["medicine"].nunique().sort_values()
    print(f"\n6) {train['medicine'].nunique()} brands map to only "
          f"{train['generic'].nunique()} active ingredients:")
    print(brands_per_generic.to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    brands_per_generic.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of brand names in dataset")
    ax.set_title("Brands per active ingredient")
    save(fig, "eda_brands_per_generic.png")

    # 2. Image sizes ---------------------------------------------------------
    shapes = np.array([load_gray(p).shape for p in train["image_path"]])
    h, w = shapes[:, 0], shapes[:, 1]
    print(f"\n2) Image size (train): height median={np.median(h):.0f}px, "
          f"width median={np.median(w):.0f}px, width/height median={np.median(w / h):.1f}")
    print(f"  -> Wide images, so we resize onto a {config.BASELINE_IMG_SIZE[0]}x"
          f"{config.BASELINE_IMG_SIZE[1]} canvas (same ~1:3 shape).")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(h, bins=40, color="#4C72B0")
    axes[0].set_title("Image height (px)")
    axes[1].hist(w / h, bins=40, color="#DD8452")
    axes[1].set_title("Aspect ratio (width / height)")
    save(fig, "eda_image_sizes.png")

    # 3. Sample grid ---------------------------------------------------------
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_meds = rng.choice(train["medicine"].unique(), 6, replace=False)
    fig, axes = plt.subplots(6, 4, figsize=(10, 7))
    for r, med in enumerate(sample_meds):
        paths = train[train["medicine"] == med]["image_path"].sample(4, random_state=r)
        for c, p in enumerate(paths):
            axes[r, c].imshow(load_gray(p), cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].set_title(med, loc="left", fontsize=9)
    fig.suptitle("Same medicine, different handwriting (4 samples each)")
    save(fig, "eda_samples.png")

    # 4. Preprocessing before/after -----------------------------------------
    # pick the image with the darkest background to show the effect clearly
    bg = [np.percentile(load_gray(p), 90) for p in train["image_path"][:400]]
    demo_path = train["image_path"][int(np.argmin(bg))]
    steps = preprocess_steps(load_gray(demo_path), config.BASELINE_IMG_SIZE)
    fig, axes = plt.subplots(len(steps), 1, figsize=(6, 9))
    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, loc="left")
        ax.axis("off")
    save(fig, "eda_preprocessing_steps.png")

    fig, axes = plt.subplots(4, 2, figsize=(8, 5))
    for r, p in enumerate(train["image_path"].sample(4, random_state=1)):
        axes[r, 0].imshow(preprocess(load_gray(p), config.BASELINE_IMG_SIZE, clean=False), cmap="gray")
        axes[r, 1].imshow(preprocess(load_gray(p), config.BASELINE_IMG_SIZE, clean=True), cmap="gray")
        axes[r, 0].axis("off")
        axes[r, 1].axis("off")
    axes[0, 0].set_title("Only resized")
    axes[0, 1].set_title("Cleaned (our pipeline)")
    save(fig, "eda_before_after.png")

    # 5. Leakage check -------------------------------------------------------
    print("\n5) Duplicate check (exact same image file in two splits):")
    hashes = {name: [file_hash(p) for p in df["image_path"]]
              for name, df in [("train", train), ("val", val), ("test", test)]}
    train_set = set(hashes["train"])
    for name in ["val", "test"]:
        n = sum(hsh in train_set for hsh in hashes[name])
        print(f"  {name} images that are copies of a train image: {n}")
    print("  -> These are removed before evaluation (see data_loader.py).")

    # 7. Near-duplicates: how SIMILAR is each val/test image to its closest
    #    training image? (catches re-saved copies that exact hashing misses)
    print("\n7) Similarity to the closest training image (smaller = more similar):")
    small = (32, 96)
    vec = lambda df: np.array([preprocess(load_gray(p), small).ravel() / 255.0 for p in df["image_path"]])
    X_train = vec(train)
    sq_train = (X_train ** 2).sum(1)
    distances = {}
    for name, df in [("val", val), ("test", test)]:
        X = vec(df)
        d2 = (X ** 2).sum(1)[:, None] + sq_train[None, :] - 2 * X @ X_train.T
        distances[name] = np.sqrt(np.maximum(d2, 0)).min(axis=1)
        print(f"  {name:5s} median distance = {np.median(distances[name]):.1f}")
    print("  -> Validation images are CLOSER to training images than test images are")
    print("     (likely the same writers). So validation accuracy is optimistic;")
    print("     the TEST score is the honest number to report.")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, d in distances.items():
        ax.hist(d, bins=40, alpha=0.6, label=name)
    ax.set_xlabel("pixel distance to closest training image")
    ax.set_title("How similar are val/test images to training images?")
    ax.legend()
    save(fig, "eda_similarity_to_train.png")

    summary = pd.DataFrame({
        "split": ["train", "val", "test"],
        "images": [len(train), len(val), len(test)],
        "duplicates_of_train": [0] + [sum(hsh in train_set for hsh in hashes[n]) for n in ["val", "test"]],
    })
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "eda_summary.csv", index=False)
    print("\nDone. Look at the figures in reports/figures/.")


if __name__ == "__main__":
    main()
