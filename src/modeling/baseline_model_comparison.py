"""
baseline_model_comparison.py
============================
Compare additional machine learning baselines against the current best
tree-based models for the Kadugannawa accident severity dataset.

Why this script exists
----------------------
Random Forest and XGBoost are strong choices, but a research report is more
convincing when it also shows that simpler/common alternatives were tested.
This script evaluates:
  - Dummy majority baseline
  - Logistic Regression
  - Support Vector Machine
  - K-Nearest Neighbours
  - Naive Bayes
  - Decision Tree
  - Extra Trees
  - Gradient Boosting
  - Random Forest
  - XGBoost, if installed

SMOTE is applied to the training data only. Cross-validation uses an imblearn
Pipeline so SMOTE is applied inside each fold and never leaks into validation.

Outputs:
  outputs/model_baselines/baseline_model_report.txt
  outputs/model_baselines/baseline_model_results.csv
  outputs/model_baselines/baseline_model_comparison.png
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "model_baselines"
RANDOM_STATE = 42
CLASS_NAMES = ["Low", "Medium", "High"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(DATA_DIR / "engineered_features.csv")
    y = pd.read_csv(DATA_DIR / "target_severity.csv").squeeze("columns")
    if X.isnull().sum().sum() != 0:
        raise ValueError("Feature matrix contains missing values. Run preprocessing first.")
    return X, y


def get_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Dummy majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Random Forest tuned": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        import xgboost as xgb

        models["XGBoost tuned"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        print("XGBoost is not installed; skipping XGBoost tuned baseline.")

    return models


def needs_scaling(model_name: str) -> bool:
    return model_name in {
        "Logistic Regression",
        "SVM RBF",
        "KNN",
        "Gaussian Naive Bayes",
    }


def model_pipeline(model_name: str, model: object) -> ImbPipeline:
    steps = []
    if needs_scaling(model_name):
        steps.append(("scaler", StandardScaler()))
    steps.extend([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", model),
    ])
    return ImbPipeline(steps)


def safe_auc(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    try:
        proba = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        return round(
            roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"),
            4,
        )
    except Exception:
        return np.nan


def evaluate_model(
    name: str,
    fitted_model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_mean: float,
    cv_std: float,
    fit_seconds: float,
) -> dict[str, object]:
    pred = fitted_model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "model": name,
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "kappa": round(cohen_kappa_score(y_test, pred), 4),
        "auc_roc": safe_auc(fitted_model, X_test, y_test),
        "cv_f1_macro_mean": round(cv_mean, 4),
        "cv_f1_macro_std": round(cv_std, 4),
        "f1_low": round(f1[0], 4),
        "f1_medium": round(f1[1], 4),
        "f1_high": round(f1[2], 4),
        "high_precision": round(precision[2], 4),
        "high_recall": round(recall[2], 4),
        "fit_seconds": round(fit_seconds, 2),
    }


def compare_models() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    models = get_models()

    for name, base_model in models.items():
        print(f"Training {name}...")
        pipe = model_pipeline(name, clone(base_model))

        cv_scores = cross_val_score(
            model_pipeline(name, clone(base_model)),
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        start = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - start

        rows.append(
            evaluate_model(
                name=name,
                fitted_model=pipe,
                X_test=X_test,
                y_test=y_test,
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                fit_seconds=fit_seconds,
            )
        )

    results = pd.DataFrame(rows).sort_values(
        ["f1_macro", "f1_high", "high_recall"],
        ascending=False,
    )
    results.to_csv(OUT_DIR / "baseline_model_results.csv", index=False)
    write_report(results)
    plot_results(results)
    return results


def write_report(results: pd.DataFrame) -> None:
    best = results.iloc[0]

    lines = [
        "BASELINE MODEL COMPARISON REPORT",
        "=" * 76,
        "Kadugannawa vehicle-type accident severity prediction",
        "",
        "Purpose",
        "-" * 76,
        "This experiment compares additional ML methods against the current",
        "Random Forest and XGBoost models. SMOTE is applied to training data",
        "only, and cross-validation applies SMOTE inside each fold.",
        "",
        "Overall Results",
        "-" * 76,
        (
            f"{'Model':<24} {'F1-Macro':>9} {'High-F1':>8} {'High-Rec':>9} "
            f"{'Kappa':>8} {'AUC':>8} {'CV Mean':>9}"
        ),
        "-" * 76,
    ]

    for _, row in results.iterrows():
        auc_value = row["auc_roc"]
        auc_text = "NA" if pd.isna(auc_value) else f"{auc_value:.4f}"
        lines.append(
            f"{row['model']:<24} {row['f1_macro']:>9.4f} "
            f"{row['f1_high']:>8.4f} {row['high_recall']:>9.4f} "
            f"{row['kappa']:>8.4f} {auc_text:>8} "
            f"{row['cv_f1_macro_mean']:>9.4f}"
        )

    lines += [
        "",
        "Recommended Interpretation",
        "-" * 76,
        f"Best model in this run: {best['model']} with F1-macro={best['f1_macro']:.4f}.",
        "For the dissertation, report these baselines to show that the final",
        "tree-based model was selected after comparison, not arbitrarily.",
        "",
        "Suggested model grouping for Chapter 5:",
        "1. Simple baseline: Dummy majority, Logistic Regression, Naive Bayes.",
        "2. Distance/margin methods: KNN and SVM.",
        "3. Tree methods: Decision Tree, Extra Trees, Random Forest.",
        "4. Boosting methods: AdaBoost, Gradient Boosting, XGBoost.",
        "",
        "Important note:",
        "Current results are based on mock/prototype accident records. Once real",
        "police records are available, rerun this script and update the final tables.",
    ]

    (OUT_DIR / "baseline_model_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def plot_results(results: pd.DataFrame) -> None:
    plot_df = results.sort_values("f1_macro", ascending=True)
    colors = [
        "#0f766e" if "XGBoost" in model else
        "#2563eb" if "Random Forest" in model else
        "#64748b"
        for model in plot_df["model"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(plot_df["model"], plot_df["f1_macro"], color=colors)
    for bar, value in zip(bars, plot_df["f1_macro"]):
        ax.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(1.0, plot_df["f1_macro"].max() + 0.08))
    ax.set_xlabel("F1-Macro on Held-Out Test Set")
    ax.set_title("Additional Model Baseline Comparison", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_model_comparison.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    results_df = compare_models()
    print("\nDone. Results saved to:")
    print(f"  {OUT_DIR / 'baseline_model_report.txt'}")
    print(f"  {OUT_DIR / 'baseline_model_results.csv'}")
    print(f"  {OUT_DIR / 'baseline_model_comparison.png'}")
    print("\nTop results:")
    print(
        results_df[
            ["model", "f1_macro", "f1_high", "high_recall", "cv_f1_macro_mean"]
        ].head(5).to_string(index=False)
    )
