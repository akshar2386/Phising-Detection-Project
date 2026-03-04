"""
Evaluation metrics and visualizations for phishing detection models.

Primary concern: high recall + low false positive rate (FPR).
  - FPR = FP / (FP + TN)  — legitimate site wrongly blocked
  - Recall = TP / (TP + FN) — phishing site caught

All plots save to results/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# phishing = -1, legitimate = 1
POS_LABEL = -1   # "positive" class is phishing


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute all relevant metrics for phishing detection.

    Parameters
    ----------
    y_true  : array-like, true labels (-1 or 1)
    y_pred  : array-like, predicted labels (-1 or 1)
    y_prob  : array-like or None, predicted probability of phishing class

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, fpr, roc_auc
    """
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()  # tn=legit correct, fp=legit wrong, etc.

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, -y_prob)  # prob of phishing
    else:
        metrics["roc_auc"] = None

    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print the metrics dictionary."""
    print(f"\n{'═' * 50}")
    print(f"  {model_name}")
    print(f"{'═' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← catch phishing")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  FPR       : {metrics['fpr']:.4f}  ← false alarm rate")
    if metrics.get("roc_auc"):
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']}")
    print(f"{'─' * 50}")


def evaluate_all_models(models: dict, data: dict, split: str = "test") -> pd.DataFrame:
    """
    Evaluate multiple models on the given split and return a summary DataFrame.

    Parameters
    ----------
    models : dict of name -> fitted estimator
    data   : dict from preprocessing.prepare_all()
    split  : 'val' or 'test'

    Returns
    -------
    pd.DataFrame with one row per model
    """
    if split == "test":
        X_raw = data["X_test"]
        X_sc = data["X_test_sc"]
        y_true = data["y_test"]
    else:
        X_raw = data["X_val"]
        X_sc = data["X_val_sc"]
        y_true = data["y_val"]

    rows = []
    for name, model in models.items():
        # Use scaled features for LR, raw for trees
        X = X_sc if "Logistic" in name else X_raw
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 0]  # prob of class -1 (phishing)
        else:
            y_prob = None

        m = compute_metrics(y_true, y_pred, y_prob)
        m["model"] = name
        print_metrics(m, name)
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrices(models: dict, data: dict, split: str = "test"):
    """Plot confusion matrices for all models side-by-side."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    y_true = data[f"y_{split}"]

    for ax, (name, model) in zip(axes, models.items()):
        X_sc = data[f"X_{split}_sc"]
        X_raw = data[f"X_{split}"]
        X = X_sc if "Logistic" in name else X_raw
        y_pred = model.predict(X)

        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Phishing", "Legit"],
            yticklabels=["Phishing", "Legit"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle(f"Confusion Matrices ({split} set)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_roc_curves(models: dict, data: dict, split: str = "test"):
    """Plot ROC curves for all models on one axes."""
    y_true = data[f"y_{split}"]
    # convert to binary: phishing=1, legitimate=0
    y_binary = (y_true == -1).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")

    for name, model in models.items():
        X_sc = data[f"X_{split}_sc"]
        X_raw = data[f"X_{split}"]
        X = X_sc if "Logistic" in name else X_raw

        if hasattr(model, "predict_proba"):
            prob_phishing = model.predict_proba(X)[:, list(model.classes_).index(-1)]
        else:
            prob_phishing = model.decision_function(X)

        fpr, tpr, _ = roc_curve(y_binary, prob_phishing)
        auc = roc_auc_score(y_binary, prob_phishing)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curves ({split} set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_precision_recall_curves(models: dict, data: dict, split: str = "test"):
    """Plot Precision-Recall curves."""
    y_true = data[f"y_{split}"]
    y_binary = (y_true == -1).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_binary.mean()
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline (P={baseline:.2f})")

    for name, model in models.items():
        X_sc = data[f"X_{split}_sc"]
        X_raw = data[f"X_{split}"]
        X = X_sc if "Logistic" in name else X_raw

        if hasattr(model, "predict_proba"):
            prob_phishing = model.predict_proba(X)[:, list(model.classes_).index(-1)]
        else:
            prob_phishing = model.decision_function(X)

        precision, recall, _ = precision_recall_curve(y_binary, prob_phishing)
        ax.plot(recall, precision, label=name)

    ax.set_xlabel("Recall (phishing sites caught)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves ({split} set)")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(RESULTS_DIR, "pr_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_metric_comparison(summary_df: pd.DataFrame):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available = [m for m in metrics_to_plot if m in summary_df.columns]

    plot_df = summary_df[available].dropna(axis=1)

    ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0, width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Key Metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_feature_importances(importance_df: pd.DataFrame, model_name: str, top_n: int = 15):
    """Horizontal bar chart of top-N feature importances."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(RESULTS_DIR, f"feature_importances_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {path}")
