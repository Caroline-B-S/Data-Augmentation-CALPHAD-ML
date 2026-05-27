"""
Machine Learning Training Pipeline — TC Dataset
=================================================
Trains and evaluates classifiers to predict phase score from composition
descriptors computed from the Thermo-Calc grid dataset.

For each (threshold, augmentation distance, model) combination:
  1. Hyperparameters are tuned with Optuna (StratifiedKFold, balanced_accuracy)
  2. The best model is trained on the full training set
  3. Evaluation is performed on the held-out test set

Models available: svm, rf, gb, knn
A DummyClassifier (most_frequent) is always run as baseline.

Author: Caroline Binde Stoco
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)


# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_TO_RUN    = ["knn", "rf", "gb", "svm"]
THRESHOLDS       = ["ter0", "ter40"]
DISTANCES        = ["dist0", "dist10", "dist30", "dist50", "dist70", "dist90"]

N_TRIALS         = 100
N_BOOTSTRAP      = 1000      # iterations for confidence interval estimation
RANDOM_STATE     = 42

DATA_DIR         = Path("tc_descriptors")
OUTPUT_DIR       = Path("tc_ml_results")

TARGET           = "phase_score"
DROP_COLS        = ["task_id", "phase_name", "sample_origin", "interp_percent"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def matches_any(name: str, patterns: list[str]) -> bool:
    """
    Exact token match — prevents 'dist10' from matching 'dist100'.
    Tokens are delimited by underscores, hyphens, or end of string.
    """
    for pat in patterns:
        if re.search(rf"(?<![a-zA-Z0-9]){re.escape(pat)}(?![a-zA-Z0-9])", name):
            return True
    return False


def load_dataset(train_file: Path, test_file: Path):
    """Load train and test CSVs and return feature matrices and targets."""
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)

    X_train = train_df.drop(columns=DROP_COLS + [TARGET], errors="ignore")
    y_train = train_df[TARGET]

    X_test  = test_df.drop(columns=DROP_COLS + [TARGET], errors="ignore")
    y_test  = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> dict:
    """
    Bootstrap 95% confidence intervals for balanced_accuracy and cohen_kappa.
    Vectorized: all n bootstrap samples are drawn at once as a (n, len(y))
    index matrix, avoiding a Python loop over iterations.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.integers(0, len(y_true), size=(n, len(y_true)))

    y_true_boot = y_true[indices]
    y_pred_boot = y_pred[indices]

    all_labels = np.unique(y_true)   # fixed label set — prevents missing-class errors

    ba_scores = np.array([
        balanced_accuracy_score(y_true_boot[i], y_pred_boot[i])
        for i in range(n)
    ])

    kappa_scores = np.array([
        cohen_kappa_score(y_true_boot[i], y_pred_boot[i], labels=all_labels)
        for i in range(n)
    ])

    return {
        "balanced_accuracy_ci_low":  float(np.percentile(ba_scores,    2.5)),
        "balanced_accuracy_ci_high": float(np.percentile(ba_scores,   97.5)),
        "cohen_kappa_ci_low":        float(np.percentile(kappa_scores,  2.5)),
        "cohen_kappa_ci_high":       float(np.percentile(kappa_scores, 97.5)),
    }


def class_distribution(y: pd.Series) -> dict:
    """Fraction of each class — reported for transparency on class imbalance."""
    dist = y.value_counts(normalize=True).sort_index()
    return {f"class_{cls}_fraction": round(frac, 4) for cls, frac in dist.items()}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Full evaluation suite:
      - balanced_accuracy  : primary metric (robust to class imbalance)
      - cohen_kappa        : agreement beyond chance (valued by reviewers)
      - f1_weighted        : harmonic mean weighted by support
      - f1_macro           : unweighted harmonic mean across classes
      - roc_auc_ovr        : one-vs-rest AUC (multi-class)
      - bootstrap 95% CIs  : statistical grounding for comparisons
      - class distribution : transparency on test set composition
    """
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_ovr":       float(roc_auc_score(y_true, y_proba, multi_class="ovr")),
    }
    metrics.update(bootstrap_ci(np.array(y_true), np.array(y_pred)))
    metrics.update(class_distribution(y_true))
    return metrics


def save_curves(y_true, y_proba, classes: list, out_dir: Path):
    """Save ROC and precision-recall curves using real class labels."""
    roc_data, pr_data = [], []

    for cls in classes:
        binary_true = (y_true == cls).astype(int)
        scores      = y_proba[:, list(classes).index(cls)]

        fpr, tpr, _         = roc_curve(binary_true, scores)
        precision, recall, _ = precision_recall_curve(binary_true, scores)

        roc_data.append(pd.DataFrame({"class": cls, "fpr": fpr, "tpr": tpr}))
        pr_data.append(pd.DataFrame({"class": cls, "precision": precision, "recall": recall}))

    pd.concat(roc_data).to_csv(out_dir / "roc_curves.csv", index=False)
    pd.concat(pr_data).to_csv(out_dir / "pr_curves.csv", index=False)


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(model_type: str, params: dict):
    if model_type == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params.get("gamma", "scale"),
                degree=params.get("degree", 3),
                class_weight=params["class_weight"],
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_type == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                metric=params["metric"],
                p=params.get("p", 2),   # only relevant for minkowski; default 2
                leaf_size=params["leaf_size"],
                n_jobs=1,
            )),
        ])

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=1,               # parallelism handled by cross_val_score
        )

    if model_type == "gb":
        return HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=RANDOM_STATE,
            # HistGradientBoosting is histogram-based (LightGBM-style):
            # much faster than GradientBoostingClassifier on large datasets
        )

    raise ValueError(f"Unknown model type: {model_type}")


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial, model_type: str, X, y) -> float:
    if model_type == "svm":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C":            trial.suggest_float("C", 1e-2, 1e2, log=True),
            "kernel":       kernel,
            "gamma":        trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf", "poly"] else "scale",
            "degree":       trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    elif model_type == "knn":
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 60),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":      metric,
            "p":           trial.suggest_int("p", 1, 5) if metric == "minkowski" else (1 if metric == "manhattan" else 2),
            "leaf_size":   trial.suggest_int("leaf_size", 15, 60),
        }

    elif model_type == "rf":
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    elif model_type == "gb":
        params = {
            "max_iter":          trial.suggest_int("max_iter", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(
        make_model(model_type, params), X, y,
        cv=cv, scoring="balanced_accuracy", n_jobs=-1,
    ).mean()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_dummy(X_train, X_test, y_train, y_test, out_dir: Path):
    """Train and evaluate the DummyClassifier baseline."""
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(y_proba, columns=[f"class_{c}" for c in classes]).to_csv(out_dir / "probabilities.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    json.dump(compute_metrics(y_test, y_pred, y_proba), open(out_dir / "metrics.json", "w"), indent=4)
    save_curves(y_test.values, y_proba, classes, out_dir)
    print(f"  Baseline saved → {out_dir}")


def run_model(model_type: str, X_train, X_test, y_train, y_test,
              suffix: str, out_dir: Path):
    """Tune, train, evaluate, and save a single model."""
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, model_type, X_train, y_train),
        n_trials=N_TRIALS,
    )

    best_params = study.best_params
    model = make_model(model_type, best_params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_

    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / f"{model_type}_{suffix}_model.pkl")
    json.dump(best_params,   open(OUTPUT_DIR / f"{model_type}_{suffix}_params.json",  "w"), indent=4)
    study.trials_dataframe().to_csv(OUTPUT_DIR / f"{model_type}_{suffix}_optuna.csv", index=False)

    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(y_proba, columns=[f"class_{c}" for c in classes]).to_csv(out_dir / "probabilities.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    json.dump(compute_metrics(y_test, y_pred, y_proba), open(out_dir / "metrics.json", "w"), indent=4)
    save_curves(y_test.values, y_proba, classes, out_dir)

    print(f"  Saved → {out_dir}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_files = sorted(DATA_DIR.glob("tc_augmented_*_descriptors.csv"))

    for train_file in train_files:
        name = train_file.name

        if not matches_any(name, THRESHOLDS):
            continue
        if not matches_any(name, DISTANCES):
            continue

        suffix = name.replace("tc_augmented_", "").replace("_descriptors.csv", "")

        # Extract the ternary threshold (e.g. "ter50") from the filename
        threshold_match = re.search(r"(ter\d+)", name)
        if not threshold_match:
            print(f"  Could not extract threshold from filename, skipping: {name}")
            continue
        threshold = threshold_match.group(1)
        test_file = DATA_DIR / f"test_original_{threshold}_descriptors.csv"

        if not test_file.exists():
            print(f"  Test file not found, skipping: {test_file}")
            continue

        print(f"\n{'='*50}")
        print(f"Dataset : {suffix}")
        print(f"{'='*50}")

        X_train, X_test, y_train, y_test = load_dataset(train_file, test_file)

        # Always run dummy baseline
        print("\n  [baseline] DummyClassifier")
        run_dummy(X_train, X_test, y_train, y_test,
                  out_dir=OUTPUT_DIR / f"dummy_{suffix}")

        # Run selected models
        for model_type in MODELS_TO_RUN:
            print(f"\n  [{model_type}] Optimizing ({N_TRIALS} trials)...")
            run_model(model_type, X_train, X_test, y_train, y_test,
                      suffix=suffix,
                      out_dir=OUTPUT_DIR / f"{model_type}_{suffix}")


if __name__ == "__main__":
    main()