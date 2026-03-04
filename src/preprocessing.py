"""
Data preprocessing for the phishing dataset.

The UCI dataset features are already encoded as ordinal integers {-1, 0, 1}
(or sometimes {0, 1}). This module handles:
  - Train/validation/test splitting (stratified)
  - Feature scaling (StandardScaler for linear models)
  - Handling class imbalance information
  - Feature group definitions for analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_NAMES, TARGET_NAME


# ── Feature groupings (for analysis and reporting) ──────────────────────────

URL_FEATURES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "HTTPS_token",
]

SECURITY_FEATURES = [
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "Abnormal_URL",
    "DNSRecord",
    "age_of_domain",
    "Statistical_report",
]

BEHAVIOR_FEATURES = [
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
]

FEATURE_GROUPS = {
    "URL-Based": URL_FEATURES,
    "Domain/Security": SECURITY_FEATURES,
    "Behavior/Structure": BEHAVIOR_FEATURES,
}


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified train / validation / test split.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series  (values: -1 or 1)
    """
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split the remaining into train / val
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data and transform all splits.
    Needed for Logistic Regression; tree models can use raw data.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_class_distribution(y: pd.Series) -> dict:
    """Return counts and proportions for each class."""
    counts = y.value_counts().to_dict()
    total = len(y)
    return {
        "phishing_count": counts.get(-1, 0),
        "legitimate_count": counts.get(1, 0),
        "phishing_pct": counts.get(-1, 0) / total * 100,
        "legitimate_pct": counts.get(1, 0) / total * 100,
        "total": total,
    }


def prepare_all(
    df: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
        X_train, X_val, X_test          (DataFrames, unscaled)
        X_train_sc, X_val_sc, X_test_sc (np.ndarray, scaled)
        y_train, y_val, y_test          (Series)
        scaler                          (fitted StandardScaler)
        feature_names                   (list of str)
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, random_state=random_state
    )
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(
        X_train, X_val, X_test
    )

    print("── Data splits ──────────────────────────")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = get_class_distribution(y)
        print(
            f"  {name:6s}: {dist['total']:5d} samples | "
            f"Phishing {dist['phishing_pct']:.1f}% | "
            f"Legit {dist['legitimate_pct']:.1f}%"
        )
    print("─────────────────────────────────────────")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_sc": X_train_sc,
        "X_val_sc": X_val_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
    }
