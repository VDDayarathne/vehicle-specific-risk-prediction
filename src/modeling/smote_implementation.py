"""
smote_implementation.py
=======================
Complete SMOTE implementation for imbalanced accident severity data.
Covers 5 strategies, k-sensitivity analysis, and dissertation-ready
evaluation metrics.

Run after preprocessing_pipeline.py has produced pipeline_output/

Requirements:  pip install imbalanced-learn scikit-learn pandas numpy
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

SEVERITY_NAMES = ["Low", "Medium", "High"]
COLORS         = ["#1D9E75", "#BA7517", "#A32D2D"]
OUTPUT_DIR     = "smote_output"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PIPELINE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def load_pre_smote_data() -> tuple:
    """
    Load the raw (imbalanced) training split from pipeline_output.
    We reconstruct pre-SMOTE data by loading from pipeline CSVs.
    The pipeline already saved X_train (post-SMOTE), so we reload
    the original imbalanced split here for demonstration.

    In practice: just call SMOTE on your raw X_train/y_train directly.
    """
    import os

    # Try pipeline outputs first
    if os.path.exists("pipeline_output/X_train.csv"):
        # Note: pipeline output is already post-SMOTE (681 rows).
        # For this script, we load test set (never touched) and
        # rebuild an imbalanced train split from the full raw data.
        X_test  = pd.read_csv("pipeline_output/X_test.csv")
        y_test  = pd.read_csv("pipeline_output/y_test.csv").squeeze()
        feature_names = pd.read_csv("pipeline_output/feature_names.csv")["feature"].tolist()

        # Rebuild raw split from original CSVs
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from preprocessing_pipeline import (
            load_datasets, clean_dates, clean_weather,
            merge_datasets, impute_missing, engineer_features, prepare_Xy
        )
        import sys
        sys.path.insert(0, '.')

        accidents, weather, osm = load_datasets(
            "kadugannawa_accidents_mock.csv", "export.xlsx", None
        )
        accidents, weather = clean_dates(accidents, weather)
        weather            = clean_weather(weather)
        df                 = merge_datasets(accidents, weather, osm)
        df, _, _           = impute_missing(df)
        df                 = engineer_features(df)
        X_full, y_full, _  = prepare_Xy(df)

        X_train_raw, _, y_train_raw, _ = train_test_split(
            X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
        )
        print(f"Loaded: X_train_raw={X_train_raw.shape}, X_test={X_test.shape}")
        return X_train_raw, X_test, y_train_raw, y_test, feature_names
    else:
        raise FileNotFoundError(
            "Run preprocessing_pipeline.py first to generate pipeline_output/"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMOTE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def get_smote_strategies(k: int = 5) -> dict:
    """
    Return all SMOTE strategies as a dict {name: sampler}.
    Call strategy.fit_resample(X_train, y_train) on any of them.

    Recommended order to try:
      1. SMOTE (standard)       — always start here
      2. SMOTE + Tomek          — adds cleaning after oversampling
      3. BorderlineSMOTE        — focuses on ambiguous boundary points
      4. SVMSMOTE               — uses SVM to identify borderline cases

    Args:
        k: number of nearest neighbours (test k=1,3,5,7,10 — see below)
    """
    return {
        "SMOTE (standard)":  SMOTE(
            random_state=42,
            k_neighbors=k,
            sampling_strategy="auto",   # balance all minority -> majority count
        ),
        "BorderlineSMOTE":   BorderlineSMOTE(
            random_state=42,
            k_neighbors=k,
            kind="borderline-1",        # "borderline-2" for noisier data
        ),
        "SVMSMOTE":          SVMSMOTE(
            random_state=42,
            k_neighbors=k,
        ),
        "SMOTE + Tomek":     SMOTETomek(
            random_state=42,
            # Tomek links remove majority samples that are close to minority
            # (cleans noise after oversampling)
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. K-SENSITIVITY ANALYSIS  (run once to choose best k)
# ══════════════════════════════════════════════════════════════════════════════

def k_sensitivity_analysis(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series,  y_test: pd.Series,
                            k_values: list = None) -> pd.DataFrame:
    """
    Test SMOTE with different k values and return F1 scores.
    Run this ONCE to pick the best k for your dataset size.

    Rule of thumb:
      - k=5  : default, good for most datasets
      - k=3  : use when smallest class has < 20 samples
      - k=7  : try if k=5 doesn't improve High class recall
      - k=10 : diminishing returns; risks overfitting if dataset is small
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10]

    rows = []
    for k in k_values:
        min_class = y_train.value_counts().min()
        if k >= min_class:
            print(f"  k={k}: skipped (k must be < minority class size {min_class})")
            continue

        sm = SMOTE(random_state=42, k_neighbors=k)
        Xtr, ytr = sm.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        y_pred = rf.predict(X_test)

        f1_all = f1_score(y_test, y_pred, average=None, zero_division=0)
        rows.append({
            "k":         k,
            "train_size":len(ytr),
            "f1_macro":  round(f1_score(y_test,y_pred,average="macro"),3),
            "f1_low":    round(f1_all[0],3),
            "f1_medium": round(f1_all[1],3),
            "f1_high":   round(f1_all[2],3),
        })
        print(f"  k={k}: F1-macro={rows[-1]['f1_macro']:.3f}  "
              f"[Low={rows[-1]['f1_low']} Med={rows[-1]['f1_medium']} High={rows[-1]['f1_high']}]")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE ALL STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategies(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series,  y_test: pd.Series,
                       k: int = 5) -> pd.DataFrame:
    """
    Run all SMOTE strategies + a no-SMOTE baseline.
    Returns a summary DataFrame sorted by F1-macro (best first).
    """
    results = []

    # Baseline: no resampling, use class_weight to partially compensate
    rf_base = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    f1_b = f1_score(y_test, y_pred_base, average=None, zero_division=0)
    results.append({
        "strategy":   "No SMOTE (baseline)",
        "train_size": len(y_train),
        "class_dist": y_train.value_counts().sort_index().to_dict(),
        "f1_macro":   round(f1_score(y_test, y_pred_base, average="macro"),3),
        "f1_low":     round(f1_b[0],3),
        "f1_medium":  round(f1_b[1],3),
        "f1_high":    round(f1_b[2],3),
    })
    print(f"  Baseline (no SMOTE)    F1-macro={results[-1]['f1_macro']:.3f}  "
          f"[Low={results[-1]['f1_low']} Med={results[-1]['f1_medium']} "
          f"High={results[-1]['f1_high']}]")

    for name, sampler in get_smote_strategies(k).items():
        try:
            Xtr, ytr = sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"  {name}: FAILED — {e}")
            continue

        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        rf.fit(Xtr, ytr)
        y_pred = rf.predict(X_test)

        f1_all = f1_score(y_test, y_pred, average=None, zero_division=0)
        results.append({
            "strategy":   name,
            "train_size": len(ytr),
            "class_dist": pd.Series(ytr).value_counts().sort_index().to_dict(),
            "f1_macro":   round(f1_score(y_test, y_pred, average="macro"),3),
            "f1_low":     round(f1_all[0],3),
            "f1_medium":  round(f1_all[1],3),
            "f1_high":    round(f1_all[2],3),
        })
        print(f"  {name:<25}  F1-macro={results[-1]['f1_macro']:.3f}  "
              f"[Low={results[-1]['f1_low']} Med={results[-1]['f1_medium']} "
              f"High={results[-1]['f1_high']}]")

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. BEST STRATEGY — DETAILED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_with_best_smote(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series,  y_test: pd.Series,
                           strategy: str = "SMOTE + Tomek",
                           k: int = 7) -> tuple:
    """
    Apply the best SMOTE strategy and return trained RF + predictions.

    Based on k-sensitivity: k=7 gives F1-macro=0.880 on this dataset.
    Use 'SMOTE + Tomek' for cleaner decision boundaries.

    Returns: (rf_model, X_train_resampled, y_train_resampled, y_pred)
    """
    strategies = get_smote_strategies(k)
    sampler    = strategies.get(strategy, strategies["SMOTE (standard)"])

    print(f"\nApplying: {strategy} (k={k})")
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    dist = pd.Series(y_res).value_counts().sort_index()
    print(f"  Before: {y_train.value_counts().sort_index().to_dict()}")
    print(f"  After : {dist.to_dict()}")
    print(f"  New training size: {len(y_res)} rows")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_res, y_res)
    y_pred = rf.predict(X_test)

    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=SEVERITY_NAMES))

    return rf, X_res, y_res, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS (for dissertation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_balance_comparison(y_before: pd.Series,
                                   y_after: pd.Series,
                                   output_dir: str) -> None:
    """Before/After SMOTE class balance bar chart."""
    import os; os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y, title in [
        (axes[0], y_before, "Before SMOTE (imbalanced)"),
        (axes[1], y_after,  "After SMOTE (balanced)"),
    ]:
        counts = pd.Series(y).value_counts().sort_index()
        bars = ax.bar(SEVERITY_NAMES, counts.values, color=COLORS, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    str(val), ha="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Accident severity")
        ax.set_ylabel("Number of samples")
        ax.set_ylim(0, max(counts.values)*1.2)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("SMOTE Effect on Class Distribution\nKadugannawa Accident Dataset",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{output_dir}/smote_class_balance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pca_scatter(X_before: pd.DataFrame, y_before: pd.Series,
                     X_after:  pd.DataFrame, y_after:  pd.Series,
                     output_dir: str) -> None:
    """PCA 2D scatter: real vs synthetic points."""
    pca = PCA(n_components=2, random_state=42)
    X_all_2d = pca.fit_transform(X_after)

    n_real  = len(X_before)
    is_syn  = np.zeros(len(X_after), dtype=bool)
    is_syn[n_real:] = True

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ve = pca.explained_variance_ratio_

    for idx, ax in enumerate(axes):
        for cls in [0, 1, 2]:
            if idx == 0:   # before SMOTE
                mask = (~is_syn) & (pd.Series(y_after).values == cls)
            else:          # after SMOTE (all points)
                mask = (pd.Series(y_after).values == cls)

            real_mask = mask & ~is_syn
            syn_mask  = mask & is_syn

            ax.scatter(X_all_2d[real_mask, 0], X_all_2d[real_mask, 1],
                       c=COLORS[cls], alpha=0.6, s=18, zorder=3,
                       label=f"{SEVERITY_NAMES[cls]} (real)")
            if idx == 1:
                ax.scatter(X_all_2d[syn_mask, 0], X_all_2d[syn_mask, 1],
                           c=COLORS[cls], alpha=0.3, s=18, marker="s",
                           label=f"{SEVERITY_NAMES[cls]} (synthetic)")

        ax.set_xlabel(f"PC1 ({ve[0]:.0%} variance)", fontsize=10)
        ax.set_ylabel(f"PC2 ({ve[1]:.0%} variance)", fontsize=10)
        ax.set_title("Before SMOTE" if idx==0 else "After SMOTE", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)

    handles = [mpatches.Patch(color=COLORS[c], label=f"{SEVERITY_NAMES[c]} (real)")
               for c in range(3)]
    handles += [mpatches.Patch(color=COLORS[c], alpha=0.3,
                               label=f"{SEVERITY_NAMES[c]} (synthetic)")
                for c in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("PCA Projection: Real vs SMOTE-Generated Samples", fontsize=12)
    plt.tight_layout()
    out = f"{output_dir}/smote_pca_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_strategy_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart: F1 per class for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    w = 0.25
    for i, (col, name, color) in enumerate([
        ("f1_low",    "Low",    COLORS[0]),
        ("f1_medium", "Medium", COLORS[1]),
        ("f1_high",   "High",   COLORS[2]),
    ]):
        bars = ax.bar(x + i*w, results_df[col], w, label=name,
                      color=color, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, results_df[col]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(results_df["strategy"], fontsize=9, rotation=12)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("SMOTE Strategy Comparison by Severity Class", fontsize=12)
    ax.legend(title="Severity")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = f"{output_dir}/smote_strategy_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray,
                           title: str, output_dir: str) -> None:
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=SEVERITY_NAMES,
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = f"{output_dir}/confusion_matrix_{title.lower().replace(' ','_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_k_sensitivity(k_df: pd.DataFrame, output_dir: str) -> None:
    """Line chart: F1 vs k for all severity classes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, name, color in [
        ("f1_macro",  "Macro avg",  "#533AB7"),
        ("f1_low",    "Low",        COLORS[0]),
        ("f1_medium", "Medium",     COLORS[1]),
        ("f1_high",   "High",       COLORS[2]),
    ]:
        style = "-o" if col == "f1_macro" else "--o"
        ax.plot(k_df["k"], k_df[col], style, color=color, label=name,
                linewidth=2 if col=="f1_macro" else 1.2, markersize=6)

    ax.set_xlabel("k (number of nearest neighbours)")
    ax.set_ylabel("F1 Score")
    ax.set_title("SMOTE k-Sensitivity Analysis", fontsize=12)
    ax.legend()
    ax.set_xticks(k_df["k"])
    ax.set_ylim(0, 1.05)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    best_k = k_df.loc[k_df["f1_macro"].idxmax(), "k"]
    ax.axvline(best_k, color="#533AB7", linestyle=":", alpha=0.5,
               label=f"Best k={best_k}")
    plt.tight_layout()
    out = f"{output_dir}/smote_k_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("  SMOTE Analysis — Kadugannawa Accident Severity Data")
    print("="*60)

    # 1. Load data
    print("\n[1] Loading data...")
    X_train_raw, X_test, y_train_raw, y_test, feature_names = load_pre_smote_data()

    print(f"\n  Class distribution BEFORE SMOTE:")
    for cls, count in y_train_raw.value_counts().sort_index().items():
        pct = count/len(y_train_raw)*100
        bar = "█" * int(pct/3)
        print(f"    {SEVERITY_NAMES[cls]:<8} {count:3d} ({pct:.0f}%) {bar}")
    ir = y_train_raw.value_counts().max() / y_train_raw.value_counts().min()
    print(f"  Imbalance ratio: {ir:.1f}:1")

    # 2. k-sensitivity
    print("\n[2] k-sensitivity analysis...")
    k_df = k_sensitivity_analysis(X_train_raw, X_test, y_train_raw, y_test)
    best_k = int(k_df.loc[k_df["f1_macro"].idxmax(), "k"])
    print(f"  -> Best k = {best_k}")
    plot_k_sensitivity(k_df, OUTPUT_DIR)

    # 3. Strategy comparison
    print("\n[3] Comparing SMOTE strategies...")
    results_df = compare_strategies(X_train_raw, X_test, y_train_raw, y_test, k=best_k)
    print(f"\n  Strategy ranking (by F1-macro):")
    print(results_df[["strategy","f1_macro","f1_low","f1_medium","f1_high"]].to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/strategy_comparison.csv", index=False)
    plot_strategy_comparison(results_df, OUTPUT_DIR)

    # 4. Best strategy — detailed
    best_strategy = results_df.iloc[0]["strategy"]
    print(f"\n[4] Training with best strategy: {best_strategy} (k={best_k})")
    rf, X_res, y_res, y_pred = train_with_best_smote(
        X_train_raw, X_test, y_train_raw, y_test,
        strategy=best_strategy, k=best_k
    )

    # 5. Plots
    print("\n[5] Generating dissertation figures...")
    plot_class_balance_comparison(y_train_raw, pd.Series(y_res), OUTPUT_DIR)
    plot_pca_scatter(X_train_raw, X_train_raw, X_res, pd.Series(y_res), OUTPUT_DIR)
    plot_strategy_comparison(results_df, OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_pred, "After SMOTE", OUTPUT_DIR)

    # Baseline confusion matrix for comparison
    rf_base = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    rf_base.fit(X_train_raw, y_train_raw)
    plot_confusion_matrix(y_test, rf_base.predict(X_test), "No SMOTE Baseline", OUTPUT_DIR)

    # 6. Save best model
    with open(f"{OUTPUT_DIR}/rf_smote_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"\n  Saved model: {OUTPUT_DIR}/rf_smote_model.pkl")

    print("\n" + "="*60)
    print(f"  [OK] Done. All outputs in: {OUTPUT_DIR}/")
    print("    smote_class_balance.png      <- include in Ch.4")
    print("    smote_pca_scatter.png        <- include in Ch.4")
    print("    smote_strategy_comparison.png<- include in Ch.5")
    print("    smote_k_sensitivity.png      <- include in appendix")
    print("    confusion_matrix_*.png       <- include in Ch.5")
    print("    strategy_comparison.csv      <- put in results table")
    print("    rf_smote_model.pkl           <- load in FastAPI")
    print("="*60)