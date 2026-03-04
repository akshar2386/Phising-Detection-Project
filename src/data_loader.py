"""
Data loading and downloading for the UCI Phishing Websites Dataset.
Dataset: https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
11,055 instances, 30 features + 1 label (phishing=-1, legitimate=1)
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np


# All 30 feature names from the UCI dataset
FEATURE_NAMES = [
    "having_IP_Address",       # URL has IP address instead of domain
    "URL_Length",              # URL length (-1=long, 0=medium, 1=short)
    "Shortining_Service",      # Uses URL shortening service
    "having_At_Symbol",        # URL has '@' symbol
    "double_slash_redirecting",# URL has '//' after the path
    "Prefix_Suffix",           # Domain has '-' in name
    "having_Sub_Domain",       # Number of subdomains
    "SSLfinal_State",          # SSL certificate state
    "Domain_registeration_length", # Domain registration length
    "Favicon",                 # Favicon loaded from external domain
    "port",                    # Uses non-standard port
    "HTTPS_token",             # 'https' token in domain
    "Request_URL",             # Ratio of external requests
    "URL_of_Anchor",           # Ratio of anchor URLs pointing outside
    "Links_in_tags",           # Links in meta/script/link tags
    "SFH",                     # Server Form Handler
    "Submitting_to_email",     # Form submits to email
    "Abnormal_URL",            # URL doesn't match WHOIS domain
    "Redirect",                # Number of redirects
    "on_mouseover",            # Status bar customized via mouseover
    "RightClick",              # Right-click disabled
    "popUpWidnow",             # Pop-up with text fields
    "Iframe",                  # Uses IFrame
    "age_of_domain",           # Domain age < 6 months
    "DNSRecord",               # DNS record present
    "web_traffic",             # Alexa rank
    "Page_Rank",               # Google PageRank
    "Google_Index",            # Indexed by Google
    "Links_pointing_to_page",  # Number of inbound links
    "Statistical_report",      # Host/IP flagged in reports
]

TARGET_NAME = "Result"

UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00327/Training%20Dataset.arff"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_ARFF_PATH = os.path.join(DATA_DIR, "phishing_dataset.arff")
PROCESSED_CSV_PATH = os.path.join(DATA_DIR, "phishing_dataset.csv")


def download_dataset(force: bool = False) -> str:
    """Download the raw ARFF file from UCI if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_ARFF_PATH) and not force:
        print(f"Dataset already exists at {RAW_ARFF_PATH}")
        return RAW_ARFF_PATH

    print(f"Downloading dataset from UCI repository...")
    try:
        urllib.request.urlretrieve(UCI_DATA_URL, RAW_ARFF_PATH)
        print(f"Downloaded to {RAW_ARFF_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n"
            f"Manually download from:\n  {UCI_DATA_URL}\n"
            f"and save to: {RAW_ARFF_PATH}"
        )
    return RAW_ARFF_PATH


def parse_arff(filepath: str) -> pd.DataFrame:
    """Parse an ARFF file into a DataFrame."""
    data_section = False
    rows = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                data_section = True
                continue
            if data_section:
                values = [int(v) for v in line.split(",")]
                rows.append(values)

    columns = FEATURE_NAMES + [TARGET_NAME]
    df = pd.DataFrame(rows, columns=columns)
    return df


def load_dataset(use_cache: bool = True) -> pd.DataFrame:
    """
    Load the phishing dataset as a DataFrame.
    Downloads from UCI and caches as CSV if not already done.

    Returns
    -------
    df : pd.DataFrame
        Shape (11055, 31). Features are in {-1, 0, 1},
        target 'Result' is -1 (phishing) or 1 (legitimate).
    """
    if use_cache and os.path.exists(PROCESSED_CSV_PATH):
        print(f"Loading cached dataset from {PROCESSED_CSV_PATH}")
        return pd.read_csv(PROCESSED_CSV_PATH)

    download_dataset()
    df = parse_arff(RAW_ARFF_PATH)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Dataset saved to {PROCESSED_CSV_PATH}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_descriptions() -> dict:
    """Return a dict mapping feature name -> description string."""
    descriptions = {
        "having_IP_Address": "URL contains an IP address instead of a domain name",
        "URL_Length": "Length of the URL (long URLs more likely phishing)",
        "Shortining_Service": "URL uses a shortening service (e.g., bit.ly)",
        "having_At_Symbol": "URL contains '@' symbol",
        "double_slash_redirecting": "URL has '//' after the path portion",
        "Prefix_Suffix": "Domain name contains a dash '-'",
        "having_Sub_Domain": "Number of subdomains in the URL",
        "SSLfinal_State": "HTTPS used and certificate is trusted",
        "Domain_registeration_length": "Domain registration length (short = phishing)",
        "Favicon": "Favicon loaded from external domain",
        "port": "URL uses a non-standard port",
        "HTTPS_token": "Token 'https' appears in the domain name",
        "Request_URL": "Percentage of external object requests",
        "URL_of_Anchor": "Percentage of anchor tags pointing outside domain",
        "Links_in_tags": "Percentage of links in meta/script/link tags",
        "SFH": "Server Form Handler is empty or 'about:blank'",
        "Submitting_to_email": "Form data submitted to an email address",
        "Abnormal_URL": "URL does not match registered WHOIS domain",
        "Redirect": "Number of redirects in the URL path",
        "on_mouseover": "Status bar is customized via mouseover events",
        "RightClick": "Right-click functionality is disabled",
        "popUpWidnow": "Pop-up window contains text fields",
        "Iframe": "Page uses invisible or frameless IFrame",
        "age_of_domain": "Domain age is less than 6 months",
        "DNSRecord": "No DNS record found for domain",
        "web_traffic": "Website traffic rank (low rank = suspicious)",
        "Page_Rank": "Google PageRank (low rank = suspicious)",
        "Google_Index": "Website is indexed by Google",
        "Links_pointing_to_page": "Number of external links pointing to page",
        "Statistical_report": "Host/IP flagged in phishing statistical reports",
    }
    return descriptions
