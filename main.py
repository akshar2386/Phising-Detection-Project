"""
EE 467 – Phishing Website Detection
Main pipeline script.

Usage:
    python main.py                  # run full pipeline with defaults
    python main.py --tune           # run with grid-search hyperparameter tuning
    python main.py --split val      # evaluate on validation set instead of test
    python main.py --model lr       # run only Logistic Regression
    python main.py --no-plots       # skip plotting (useful in headless envs)
"""

import argparse
import os
import sys
import joblib
import pandas as pd

from src.data_loader import load_dataset, FEATURE_NAMES
from src.preprocessing import prepare_all
from src.models import (
    get_default_models,
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    tune_model,
    get_feature_importances,
    LR_PARAM_GRID,
    RF_PARAM_GRID,
    GB_PARAM_GRID,
)
from src.evaluation import (
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_metric_comparison,
    plot_feature_importances,
    RESULTS_DIR,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")


def parse_args():
    p = argparse.ArgumentParser(description="Phishing detection ML pipeline")
    p.add_argument("--tune", action="store_true",
                   help="Run grid-search hyperparameter tuning")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--model", choices=["lr", "rf", "gb", "all"], default="all",
                   help="Which model(s) to run")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating plots")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def train_models(args, data: dict) -> dict:
    """Train (and optionally tune) models. Return dict of fitted estimators."""
    X_train = data["X_train"]
    X_train_sc = data["X_train_sc"]
    y_train = data["y_train"]

    if args.tune:
        print("\n[Hyperparameter Tuning – this may take a few minutes]\n")

    trained = {}

    # ── Logistic Regression ─────────────────────────────────────────────────
    if args.model in ("lr", "all"):
        print("\n── Logistic Regression ─────────────────────────────────")
        if args.tune:
            lr_base = build_logistic_regression(random_state=args.seed)
            lr, _, _ = tune_model(lr_base, LR_PARAM_GRID, X_train_sc, y_train,
                                   scoring="f1")
        else:
            lr = build_logistic_regression(C=1.0, penalty="l2",
                                           random_state=args.seed)
            lr.fit(X_train_sc, y_train)
            print("Trained with default params.")
        trained["Logistic Regression"] = lr

    # ── Random Forest ────────────────────────────────────────────────────────
    if args.model in ("rf", "all"):
        print("\n── Random Forest ────────────────────────────────────────")
        if args.tune:
            rf_base = build_random_forest(random_state=args.seed)
            rf, _, _ = tune_model(rf_base, RF_PARAM_GRID, X_train, y_train,
                                   scoring="f1")
        else:
            rf = build_random_forest(n_estimators=200, random_state=args.seed)
            rf.fit(X_train, y_train)
            print("Trained with default params.")
        trained["Random Forest"] = rf

    # ── Gradient Boosting ────────────────────────────────────────────────────
    if args.model in ("gb", "all"):
        print("\n── Gradient Boosting ────────────────────────────────────")
        if args.tune:
            gb_base = build_gradient_boosting(random_state=args.seed)
            gb, _, _ = tune_model(gb_base, GB_PARAM_GRID, X_train, y_train,
                                   scoring="f1")
        else:
            gb = build_gradient_boosting(n_estimators=200, learning_rate=0.1,
                                          max_depth=5, random_state=args.seed)
            gb.fit(X_train, y_train)
            print("Trained with default params.")
        trained["Gradient Boosting"] = gb

    return trained


def save_models(trained: dict):
    """Persist fitted models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in trained.items():
        path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, path)
        print(f"Saved model: {path}")


def main():
    args = parse_args()
    print("=" * 60)
    print("  EE 467 – Phishing Website Detection")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading dataset...")
    df = load_dataset()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Class distribution:\n{df['Result'].value_counts().rename({-1: 'Phishing', 1: 'Legitimate'})}")

    # 2. Preprocess
    print("\n[2/4] Preprocessing...")
    data = prepare_all(df, random_state=args.seed)

    # 3. Train
    print("\n[3/4] Training models...")
    trained = train_models(args, data)
    save_models(trained)

    # 4. Evaluate
    print(f"\n[4/4] Evaluating on {args.split} set...")
    summary = evaluate_all_models(trained, data, split=args.split)

    print("\n── Summary Table ────────────────────────────────────────")
    display_cols = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc"]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(float_format="{:.4f}".format))

    # Save summary CSV
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nMetrics saved to: {summary_path}")

    # Feature importances
    for name, model in trained.items():
        try:
            imp_df = get_feature_importances(model, FEATURE_NAMES)
            imp_path = os.path.join(
                RESULTS_DIR, f"importances_{name.replace(' ', '_')}.csv"
            )
            imp_df.to_csv(imp_path, index=False)
        except ValueError:
            pass

    # Plots
    if not args.no_plots:
        print("\n── Generating plots ─────────────────────────────────────")
        plot_confusion_matrices(trained, data, split=args.split)
        plot_roc_curves(trained, data, split=args.split)
        plot_precision_recall_curves(trained, data, split=args.split)
        plot_metric_comparison(summary)
        for name, model in trained.items():
            try:
                imp_df = get_feature_importances(model, FEATURE_NAMES)
                plot_feature_importances(imp_df, name)
            except ValueError:
                pass

    print("\nDone. All results saved to results/")


if __name__ == "__main__":
    main()
