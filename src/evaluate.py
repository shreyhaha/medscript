"""
evaluate.py - the FINAL, honest evaluation on the untouched TEST set.

    python -m src.evaluate

Evaluates every trained model (baseline, and the CNN if you trained it) and
saves tables + figures in reports/.

Metrics and why:
  accuracy            classes are perfectly balanced, so accuracy is fair
  macro F1            average F1 over the 78 classes; would reveal a class
                      the model is bad at, even if overall accuracy looks good
  top-3 accuracy      is the right answer among the 3 suggestions we show?
                      (matters because a pharmacist picks from our list)
  ingredient accuracy is the ACTIVE INGREDIENT right? (brand can be wrong but
                      the drug still right, e.g. Napa vs Ace = Paracetamol).
                      This is the patient-safety metric.

Confidence analysis (human-in-the-loop):
  If we auto-accept only predictions with confidence >= t and send the rest
  to a pharmacist, how accurate are the accepted ones, and how much work is
  left for the pharmacist? The threshold is CHOSEN ON VALIDATION data and
  then REPORTED ON TEST data, so the test set never influences a decision.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from src import config
from src.correction import brand_to_generic
from src.data_loader import load_all
from src.predictor import Predictor, available_models
from src.preprocessing import load_gray

TARGET_ACCEPTED_ACCURACY = 0.95   # we want >=95% accuracy on auto-accepted words
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_metrics(probs, y_true, classes):
    mapping = brand_to_generic()
    pred = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    # ingredient-level prediction: add up probabilities per ingredient
    generics = sorted(set(mapping.values()))
    g_index = {g: i for i, g in enumerate(generics)}
    brand_to_g = np.array([g_index[mapping[c]] for c in classes])
    g_probs = np.zeros((len(probs), len(generics)))
    for j in range(len(classes)):
        g_probs[:, brand_to_g[j]] += probs[:, j]
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "top3_accuracy": float(np.mean([y in row for y, row in zip(y_true, top3)])),
        "ingredient_accuracy": float((g_probs.argmax(axis=1) == brand_to_g[y_true]).mean()),
    }


def threshold_table(probs, y_true):
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y_true
    rows = []
    for t in THRESHOLDS:
        accepted = conf >= t
        rows.append({
            "threshold": t,
            "auto_accepted_%": 100 * accepted.mean(),
            "accuracy_on_accepted_%": 100 * correct[accepted].mean() if accepted.any() else np.nan,
            "sent_to_pharmacist_%": 100 * (~accepted).mean(),
            "errors_caught_%": 100 * (~accepted & ~correct).sum() / max((~correct).sum(), 1),
        })
    return pd.DataFrame(rows).round(1)


def choose_threshold(val_table):
    """Lowest threshold whose accepted predictions reach the target accuracy
    (lowest = least work for the pharmacist)."""
    ok = val_table[val_table["accuracy_on_accepted_%"] >= 100 * TARGET_ACCEPTED_ACCURACY]
    return float(ok["threshold"].min()) if len(ok) else float(val_table["threshold"].max())


def save_fig(fig, name):
    fig.savefig(config.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def detailed_report(key, model, probs, y_true, test, val_probs, y_val):
    classes = model.classes
    pred = probs.argmax(axis=1)

    # --- most confused pairs -------------------------------------------------
    cm = confusion_matrix(y_true, pred, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    pairs = [(classes[i], classes[j], int(cm[i, j])) for i, j in zip(*np.nonzero(cm))]
    pairs = pd.DataFrame(pairs, columns=["true", "predicted_as", "count"])
    pairs = pairs.sort_values("count", ascending=False).head(15)
    mapping = brand_to_generic()
    pairs["same_ingredient"] = [mapping[a] == mapping[b] for a, b in zip(pairs["true"], pairs["predicted_as"])]
    pairs.to_csv(config.REPORTS_DIR / f"{key}_confused_pairs.csv", index=False)
    print("\n  Most confused pairs (true -> predicted):")
    print(pairs.head(10).to_string(index=False))

    # --- full confusion matrix heatmap --------------------------------------
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(confusion_matrix(y_true, pred, labels=range(len(classes))), cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {model.name} (diagonal = correct)")
    ax.tick_params(labelsize=6)
    save_fig(fig, f"{key}_confusion_matrix.png")

    # --- per-class F1: which medicines are hardest? --------------------------
    f1 = f1_score(y_true, pred, average=None, labels=range(len(classes)))
    per_class = pd.DataFrame({"medicine": classes, "f1": f1}).sort_values("f1")
    per_class.to_csv(config.REPORTS_DIR / f"{key}_per_class_f1.csv", index=False)
    print("\n  Hardest 5 medicines (lowest F1):")
    print(per_class.head(5).round(2).to_string(index=False))

    # --- failure examples ----------------------------------------------------
    wrong = np.where(pred != y_true)[0]
    if len(wrong):
        rng = np.random.default_rng(config.RANDOM_SEED)
        pick = rng.choice(wrong, min(12, len(wrong)), replace=False)
        fig, axes = plt.subplots(4, 3, figsize=(12, 6))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), pick):
            ax.imshow(load_gray(test["image_path"].iloc[i]), cmap="gray")
            ax.set_title(f"true: {classes[y_true[i]]}\npred: {classes[pred[i]]} ({probs[i].max():.0%})",
                         fontsize=8)
        fig.suptitle(f"Failure examples - {model.name}")
        save_fig(fig, f"{key}_failures.png")

    # --- confidence / human-in-the-loop analysis ------------------------------
    val_table = threshold_table(val_probs, y_val)
    chosen = choose_threshold(val_table)
    test_table = threshold_table(probs, y_true)
    test_table.to_csv(config.REPORTS_DIR / f"{key}_confidence_thresholds.csv", index=False)
    print(f"\n  Confidence thresholds on TEST (threshold chosen on validation = {chosen}):")
    print(test_table.to_string(index=False))
    row = test_table[test_table["threshold"] == chosen].iloc[0]
    print(f"\n  -> At threshold {chosen}: {row['auto_accepted_%']:.0f}% of words are auto-accepted "
          f"with {row['accuracy_on_accepted_%']:.1f}% accuracy; {row['sent_to_pharmacist_%']:.0f}% "
          f"go to a pharmacist, catching {row['errors_caught_%']:.0f}% of all mistakes.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(test_table["threshold"], test_table["accuracy_on_accepted_%"], "o-", label="accuracy on auto-accepted")
    ax.plot(test_table["threshold"], test_table["auto_accepted_%"], "s-", label="% auto-accepted")
    ax.axvline(chosen, color="grey", ls="--", label=f"chosen threshold ({chosen})")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("%")
    ax.set_title("Trade-off: automation vs. accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"{key}_confidence_tradeoff.png")
    return chosen


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = available_models()
    if not models:
        raise SystemExit("No trained model. Run python -m src.train_baseline first.")
    _, val, test = load_all()
    test_imgs = [load_gray(p) for p in test["image_path"]]
    val_imgs = [load_gray(p) for p in val["image_path"]]

    rows, results = [], {}
    for key in models:
        model = Predictor(prefer=key)
        class_to_id = {c: i for i, c in enumerate(model.classes)}
        y_test = test["medicine"].map(class_to_id).values
        y_val = val["medicine"].map(class_to_id).values
        print(f"\n=== {model.name} ===")
        probs = model.predict_proba(test_imgs)
        val_probs = model.predict_proba(val_imgs)
        metrics = compute_metrics(probs, y_test, model.classes)
        for m, v in metrics.items():
            print(f"  {m:20s} {v:.3f}")
        rows.append({"model": model.name, **{m: round(v, 4) for m, v in metrics.items()}})
        results[key] = (model, probs, y_test, val_probs, y_val)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    # bar chart comparing models
    fig, ax = plt.subplots(figsize=(8, 4))
    comparison.set_index("model")[["accuracy", "top3_accuracy", "ingredient_accuracy", "macro_f1"]].T.plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set performance")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "model_comparison.png")

    # detailed report for the best model (CNN if present)
    best_key = "cnn" if "cnn" in results else "baseline"
    model, probs, y_test, val_probs, y_val = results[best_key]
    print(f"\n=== Detailed report for the best model: {model.name} ===")
    chosen = detailed_report(best_key, model, probs, y_test, test, val_probs, y_val)
    if abs(chosen - config.CONFIDENCE_THRESHOLD) > 1e-9:
        print(f"\n  TIP: set CONFIDENCE_THRESHOLD = {chosen} in src/config.py so the app uses it.")
    print("\nSaved: reports/model_comparison.csv and figures in reports/figures/")


if __name__ == "__main__":
    main()
