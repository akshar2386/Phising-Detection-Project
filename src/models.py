"""
Model definitions and hyperparameter tuning for phishing detection.

Approach 1  – Logistic Regression (baseline, interpretable)
Approach 2a – Random Forest
Approach 2b – Gradient Boosting (XGBoost-style via sklearn)

All models use scikit-learn APIs so they share a common interface.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# ── Default hyperparameter grids ─────────────────────────────────────────────

LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [1000],
    "class_weight": [None, "balanced"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"],
}

GB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5],
}


def build_logistic_regression(
    C: float = 1.0,
    penalty: str = "l2",
    class_weight=None,
    random_state: int = 42,
) -> LogisticRegression:
    """Logistic Regression baseline model."""
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver="liblinear",
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_random_forest(
    n_estimators: int = 200,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight=None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_gradient_boosting(
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    subsample: float = 0.8,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """Gradient Boosting classifier (sklearn implementation)."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )


def tune_model(
    model,
    param_grid: dict,
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple:
    """
    Grid search with stratified k-fold CV.

    Parameters
    ----------
    model       : unfitted sklearn estimator
    param_grid  : hyperparameter grid
    X_train     : training features (array or DataFrame)
    y_train     : training labels
    scoring     : metric to optimise (default 'f1')
    cv          : number of CV folds

    Returns
    -------
    best_model  : fitted estimator with best params
    cv_results  : dict of GridSearchCV cv_results_
    best_params : best hyperparameter dict
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"Best params : {gs.best_params_}")
    print(f"Best CV {scoring}: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.cv_results_, gs.best_params_


def get_default_models(random_state: int = 42) -> dict:
    """
    Return a dict of name -> unfitted model with sensible defaults.
    Use these for quick baseline runs without grid search.
    """
    return {
        "Logistic Regression": build_logistic_regression(random_state=random_state),
        "Random Forest": build_random_forest(random_state=random_state),
        "Gradient Boosting": build_gradient_boosting(random_state=random_state),
    }


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model.

    Returns a DataFrame sorted by importance descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
