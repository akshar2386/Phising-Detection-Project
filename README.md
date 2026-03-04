<<<<<<< HEAD
# Phising-Detection-Project
EE 467 Project and Presentation
=======
# EE 467 – Phishing Website Detection

**Team:** Pravir Goosari, Akshath Rao, Rohan Sriram

Binary classification of websites as phishing (−1) or legitimate (1) using the
[UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/Phishing+Websites)
(11,055 instances, 30 features).

---

## Project Structure

```
ee467-phishing-detection/
├── main.py                  # Full pipeline: load → preprocess → train → evaluate
├── requirements.txt
├── data/                    # Auto-populated on first run
│   ├── phishing_dataset.arff
│   └── phishing_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   └── 02_models.ipynb      # Model training, tuning, evaluation
├── src/
│   ├── data_loader.py       # UCI download + ARFF parsing
│   ├── preprocessing.py     # Train/val/test split + scaling
│   ├── models.py            # LR, RF, GB + hyperparameter grids
│   └── evaluation.py        # Metrics + all plots
└── results/                 # Saved metrics, plots, models
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full pipeline (downloads data automatically)
python main.py

# With hyperparameter tuning (slower, better results)
python main.py --tune

# Evaluate on validation set instead of test
python main.py --split val

# Run only one model
python main.py --model rf      # rf | lr | gb | all

# Skip plots (e.g. in headless environment)
python main.py --no-plots
```

## Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb      # EDA
jupyter notebook notebooks/02_models.ipynb   # Models
```

## Models

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline; interpretable coefficients; uses scaled features |
| **Random Forest** | Ensemble of decision trees; handles non-linearity |
| **Gradient Boosting** | Boosted trees; typically best F1 & AUC |

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| **Recall** | Fraction of phishing sites caught (primary goal) |
| **FPR** | Fraction of legitimate sites falsely blocked (minimize) |
| **F1** | Harmonic mean of precision and recall |
| **ROC-AUC** | Overall discriminative power |

## Dataset Features

The 30 features fall into three groups:

- **URL-Based** (8): IP address in URL, URL length, shortening services, `@` symbol, etc.
- **Domain/Security** (8): SSL state, domain age, DNS records, WHOIS data
- **Behavior/Structure** (14): Anchor links, form handlers, redirects, pop-ups, Alexa rank, PageRank

All features are encoded as integers in {−1, 0, 1}.
Labels: phishing = −1, legitimate = 1.
>>>>>>> f78f12b (Initial commit: push local workspace to GitHub)
