"""
preprocessing_pipeline.py
==========================
Full preprocessing pipeline for:
  "Vehicle Type-Based Alert System for Hazardous Pathways"
  (Kadugannawa Mountain Road Accident Prediction System)

Inputs:
  - kadugannawa_accidents_mock.csv  (or real police data with same schema)
  - export.xlsx                     (your daily weather export)
  - kadugannawa_road_segments.csv   (optional — from osm_road_geometry.py)

Outputs:
  - pipeline_output/X_train.csv      (SMOTE-balanced, scaled)
  - pipeline_output/X_test.csv       (scaled, no leakage)
  - pipeline_output/y_train.csv
  - pipeline_output/y_test.csv
  - pipeline_output/feature_names.csv
  - pipeline_output/scaler.pkl
  - pipeline_output/imputer.pkl
  - pipeline_output/pipeline_report.txt

Requirements:
  pip install pandas numpy scikit-learn imbalanced-learn openpyxl

Author: Generated for Vehicle Alert System research project
"""

import os
import pickle
import warnings
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ACCIDENTS_CSV  = "kadugannawa_accidents_mock.csv"
WEATHER_XLSX   = "export.xlsx"
OSM_ROADS_CSV  = "kadugannawa_road_segments.csv"   # set None if not available
OUTPUT_DIR     = "pipeline_output"

TEST_SIZE      = 0.20   # 80/20 split
RANDOM_STATE   = 42
SMOTE_K        = 5      # k-neighbors for SMOTE (use 3 if dataset < 200 rows)

# Severity target map
SEVERITY_MAP   = {"low": 0, "medium": 1, "high": 2}
SEVERITY_NAMES = ["Low", "Medium", "High"]

# Columns that would cause data leakage or are non-predictive
LEAKAGE_COLS = [
    "injuries",         # only known AFTER the accident
    "fatalities",       # only known AFTER the accident
    "risk_score",       # derived from severity — leakage!
]
DROP_COLS = [
    "accident_id", "date", "time", "day_of_week", "location_name",
    "reporting_officer", "severity",
    "road_curvature", "surface_condition", "weather_condition",  # replaced by encoded
    "latitude", "longitude",    # use only for spatial joins, not tabular ML
] + LEAKAGE_COLS

# ── Columns to standardise (continuous numerics only) ─────────────────────────
SCALE_COLS = [
    "driver_age", "driving_exp_yrs", "estimated_speed_kmh", "speed_limit_kmh",
    "precipitation_mm", "temp_avg_c", "visibility_km", "wind_speed_kmh",
    "pressure_hpa", "road_slope_deg", "elevation_m",
    "speed_excess_kmh", "speed_ratio", "exp_ratio",
    "hour_sin", "hour_cos",
    "wx_prcp", "wx_wspd", "wx_pres", "wx_temp_range",
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(accidents_path: str, weather_path: str,
                  osm_path: str | None = None) -> tuple:
    """Load all three source datasets. Returns (accidents_df, weather_df, osm_df|None)."""
    print("[1/9] Loading datasets...")

    accidents = pd.read_csv(accidents_path)
    print(f"      Accidents : {accidents.shape[0]} rows × {accidents.shape[1]} cols")

    weather = pd.read_excel(weather_path)
    print(f"      Weather   : {weather.shape[0]} rows × {weather.shape[1]} cols")

    osm = None
    if osm_path and os.path.exists(osm_path):
        osm = pd.read_csv(osm_path)
        print(f"      OSM roads : {osm.shape[0]} rows × {osm.shape[1]} cols")
    else:
        print("      OSM roads : not found — skipping (road geometry from accident CSV)")

    return accidents, weather, osm


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN & STANDARDISE DATES
# ══════════════════════════════════════════════════════════════════════════════

def clean_dates(accidents: pd.DataFrame, weather: pd.DataFrame) -> tuple:
    """Standardise date columns to YYYY-MM-DD string for consistent merging."""
    print("[2/9] Standardising dates...")

    accidents = accidents.copy()
    weather   = weather.copy()

    accidents["date"] = pd.to_datetime(accidents["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    weather["date"]   = pd.to_datetime(weather["date"],   errors="coerce").dt.strftime("%Y-%m-%d")

    acc_null_dates = accidents["date"].isna().sum()
    if acc_null_dates > 0:
        print(f"      ⚠ {acc_null_dates} accident rows with unparseable dates — dropping")
        accidents = accidents.dropna(subset=["date"])

    print(f"      Accidents date range : {accidents['date'].min()} → {accidents['date'].max()}")
    print(f"      Weather date range   : {weather['date'].min()} → {weather['date'].max()}")

    return accidents, weather


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN & IMPUTE WEATHER EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-null columns, impute gaps with linear interpolation + forward fill,
    and derive additional features from the daily weather data.
    """
    print("[3/9] Cleaning weather export...")

    weather = weather.copy()

    # Drop columns that are entirely null (snow, wdir, wpgt, tsun in your export)
    all_null = weather.columns[weather.isnull().all()].tolist()
    if all_null:
        print(f"      Dropping all-null cols: {all_null}")
        weather = weather.drop(columns=all_null)

    # Report missingness before imputation
    missing_before = weather.isnull().sum()
    missing_before = missing_before[missing_before > 0]
    if len(missing_before):
        print(f"      Missing before impute: {missing_before.to_dict()}")

    # Linear interpolation (best for time-series gaps), then fill edges
    numeric_cols = weather.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        weather[col] = (
            weather[col]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    # Derived weather features
    if "tmax" in weather.columns and "tmin" in weather.columns:
        weather["temp_range"] = (weather["tmax"] - weather["tmin"]).round(1)

    if "prcp" in weather.columns:
        weather["is_rainy"]      = (weather["prcp"] > 1.0).astype(int)
        weather["heavy_rain"]    = (weather["prcp"] > 5.0).astype(int)
        weather["rain_3day"]     = weather["prcp"].rolling(3, min_periods=1).sum().round(2)

    if "pres" in weather.columns:
        weather["pressure_drop"] = weather["pres"].diff().fillna(0).round(2)

    print(f"      Missing after impute : {weather[numeric_cols].isnull().sum().sum()} cells")
    return weather


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MERGE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(accidents: pd.DataFrame, weather: pd.DataFrame,
                   osm: pd.DataFrame | None) -> pd.DataFrame:
    """
    Left-join accidents onto weather by date.
    Optionally spatial-join OSM road features by location name.

    For dates outside the weather dataset's range, wx_* columns will be NaN
    and are filled by the accident CSV's own weather columns in Step 5.
    """
    print("[4/9] Merging datasets...")

    # Rename weather cols to wx_* prefix to avoid collision
    wx_rename = {
        "tavg": "wx_tavg", "tmin": "wx_tmin", "tmax": "wx_tmax",
        "prcp": "wx_prcp", "wspd": "wx_wspd", "pres": "wx_pres",
        "temp_range":    "wx_temp_range",
        "is_rainy":      "wx_is_rainy",
        "heavy_rain":    "wx_heavy_rain",
        "rain_3day":     "wx_rain_3day",
        "pressure_drop": "wx_pressure_drop",
    }
    weather_renamed = weather.rename(columns=wx_rename)

    df = accidents.merge(weather_renamed, on="date", how="left")
    coverage = df["wx_tavg"].notna().mean()
    print(f"      Weather merge coverage : {coverage:.1%} of accident rows matched")

    # OSM road join (by location name)
    if osm is not None and "location_name" in df.columns and "road_name" in osm.columns:
        osm_feats = osm[["road_name", "road_curvature", "road_slope_deg",
                          "speed_limit_kmh", "max_bearing_change_deg"]].copy()
        osm_feats = osm_feats.rename(columns={"road_name": "location_name",
                                               "road_curvature": "osm_curvature",
                                               "road_slope_deg": "osm_slope",
                                               "speed_limit_kmh": "osm_speed_lim"})
        df = df.merge(osm_feats, on="location_name", how="left")
        osm_match = df["osm_curvature"].notna().mean()
        print(f"      OSM feature match rate : {osm_match:.1%}")

    print(f"      Merged shape : {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — IMPUTE MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════

def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    """
    Two-stage imputation strategy:
      A) Fill wx_* gaps using accident CSV's own weather cols (cross-source fill)
      B) KNN impute for correlated continuous features
      C) Median impute for less-correlated or sparse features
    Returns (df_imputed, knn_imputer, median_imputer)
    """
    print("[5/9] Imputing missing values...")
    df = df.copy()

    # ── A: Cross-source fill for weather gaps ─────────────────────────────────
    # Accident CSV has its own weather cols (from simulation or real records).
    # Use them to fill wx_* NaNs for date-unmatched rows (pre-2023 accidents).
    cross_fill = {
        "wx_prcp": "precipitation_mm",
        "wx_tavg": "temp_avg_c",
        "wx_wspd": "wind_speed_kmh",
        "wx_pres": "pressure_hpa",
    }
    for wx_col, acc_col in cross_fill.items():
        if wx_col in df.columns and acc_col in df.columns:
            filled = df[wx_col].isna().sum()
            df[wx_col] = df[wx_col].fillna(df[acc_col])
            still_na = df[wx_col].isna().sum()
            if filled > 0:
                print(f"      Cross-fill {wx_col} ← {acc_col}: {filled - still_na} cells filled")

    # Fill remaining wx_* with column median
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    for col in wx_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── B: KNN imputer for correlated features ────────────────────────────────
    knn_cols = ["driver_age", "driving_exp_yrs", "visibility_km"]
    knn_cols = [c for c in knn_cols if c in df.columns]
    knn_imp  = KNNImputer(n_neighbors=SMOTE_K)
    df[knn_cols] = knn_imp.fit_transform(df[knn_cols])
    print(f"      KNN impute : {knn_cols}")

    # ── C: Median impute for remaining numeric gaps ───────────────────────────
    med_cols = ["estimated_speed_kmh", "wind_speed_kmh", "pressure_hpa"]
    med_cols = [c for c in med_cols if c in df.columns]
    med_imp  = SimpleImputer(strategy="median")
    df[med_cols] = med_imp.fit_transform(df[med_cols])
    print(f"      Median impute : {med_cols}")

    total_null = df.select_dtypes(include="number").isnull().sum().sum()
    print(f"      Numeric nulls remaining : {total_null}")

    return df, knn_imp, med_imp


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features that improve ML signal for accident prediction.
    Grouped into: vehicle, time, road, driver, weather, interaction.
    """
    print("[6/9] Engineering features...")
    df = df.copy()

    # ── Vehicle features ──────────────────────────────────────────────────────
    vehicle_risk  = {"motorcycle": 4, "lorry": 4, "bus": 3,
                     "car": 2, "three-wheeler": 3}
    df["vehicle_risk_weight"] = df["vehicle_type"].map(vehicle_risk).fillna(2)
    df["is_heavy_vehicle"]    = df["vehicle_type"].isin(["bus", "lorry"]).astype(int)

    # ── Speed features ────────────────────────────────────────────────────────
    df["speed_excess_kmh"] = (df["estimated_speed_kmh"] - df["speed_limit_kmh"]).clip(lower=0)
    df["speed_ratio"]      = (df["estimated_speed_kmh"] / df["speed_limit_kmh"].replace(0, 50)).round(3)

    # ── Time features ─────────────────────────────────────────────────────────
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([16, 17, 18, 19]).astype(int)
    df["is_night"]        = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)

    # Cyclical hour encoding (avoids 23→0 discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Sri Lanka monsoon seasonality
    df["month"]      = pd.to_datetime(df["date"]).dt.month
    df["is_monsoon"] = df["month"].isin([5, 6, 7, 8, 9]).astype(int)
    df["season_enc"] = df["month"].apply(
        lambda m: 2 if m in [5, 6, 7, 8, 9]       # SW monsoon — highest rain
        else (1 if m in [10, 11, 12, 1]             # NE monsoon — moderate rain
              else 0)                                # dry inter-monsoon
    )

    # ── Road features ─────────────────────────────────────────────────────────
    df["slope_cat_enc"] = pd.cut(
        df["road_slope_deg"],
        bins=[-np.inf, 3, 10, np.inf],
        labels=[0, 1, 2]
    ).astype(float)

    df["is_sharp_curve"] = (df["road_curvature"] == "sharp").astype(int)

    # High-risk compound condition: sharp + steep + rain (strong interaction term)
    df["high_risk_combo"] = (
        (df["road_curvature"] == "sharp") &
        (df["road_slope_deg"] > 10) &
        (df["precipitation_mm"] > 1.0)
    ).astype(int)

    # ── Driver features ───────────────────────────────────────────────────────
    df["young_driver"]  = (df["driver_age"] < 25).astype(int)
    df["inexperienced"] = (df["driving_exp_yrs"] < 3).astype(int)
    df["exp_ratio"]     = (
        df["driving_exp_yrs"] / (df["driver_age"] - 17).clip(lower=1)
    ).round(3)

    # ── Weather features ──────────────────────────────────────────────────────
    df["rain_intensity_num"] = pd.cut(
        df["precipitation_mm"],
        bins=[-0.01, 0.5, 5.0, 999],
        labels=[0, 1, 2]
    ).astype(float)

    df["visibility_cls_enc"] = pd.cut(
        df["visibility_km"],
        bins=[-np.inf, 0.3, 1.0, 3.0, np.inf],
        labels=[3, 2, 1, 0]      # higher = worse visibility
    ).astype(float)

    # ── Ordinal encodings ─────────────────────────────────────────────────────
    df["road_curvature_enc"] = df["road_curvature"].map(
        {"straight": 0, "mild": 1, "sharp": 2}
    ).fillna(0)
    df["surface_enc"] = df["surface_condition"].map(
        {"dry": 0, "wet": 1, "very_wet": 2, "oily": 3}
    ).fillna(0)
    df["weather_enc"] = df["weather_condition"].map(
        {"clear": 0, "light_rain": 1, "mist": 2, "foggy": 3, "heavy_rain": 4}
    ).fillna(0)

    # ── One-hot encodings (nominal categoricals) ──────────────────────────────
    df = pd.get_dummies(
        df,
        columns=["vehicle_type", "accident_type", "driver_gender"],
        prefix=["veh", "acc", "gen"],
        drop_first=False,
        dtype=int,
    )

    new_feats = [
        "vehicle_risk_weight", "is_heavy_vehicle", "speed_excess_kmh",
        "speed_ratio", "is_night", "is_monsoon", "slope_cat_enc",
        "is_sharp_curve", "high_risk_combo", "exp_ratio",
        "rain_intensity_num", "visibility_cls_enc",
    ]
    print(f"      New features created : {len(new_feats)} domain features + OHE columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENCODE TARGET & SELECT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def prepare_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Encode the severity target, drop leakage/ID columns,
    and return X (features), y (target), feature_names.
    """
    print("[7/9] Selecting features & encoding target...")

    df = df.copy()

    # Encode target
    df["severity_enc"] = df["severity"].map(SEVERITY_MAP)
    if df["severity_enc"].isna().any():
        unknown = df[df["severity_enc"].isna()]["severity"].unique()
        raise ValueError(f"Unknown severity values: {unknown}")

    y = df["severity_enc"].astype(int)

    # Build feature list
    all_drop = set(DROP_COLS + ["severity_enc"])
    feature_cols = [c for c in df.columns if c not in all_drop]

    X = df[feature_cols].copy()

    # Ensure all columns are numeric before imputation
    # (SimpleImputer only works on numeric types)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols].copy()
    
    # Remove columns that are entirely NaN
    X = X.dropna(axis=1, how='all')
    numeric_cols = X.columns.tolist()
    
    feature_cols = numeric_cols

    # Final safety impute — catches any remaining NaN from OHE or edge cases
    final_imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(final_imp.fit_transform(X), columns=feature_cols)

    total_null = X.isnull().sum().sum()
    if total_null > 0:
        raise ValueError(f"Still {total_null} NaN values in X after imputation!")

    print(f"      Final feature count : {len(feature_cols)}")
    print(f"      Target distribution : {y.value_counts().sort_index().to_dict()}")
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SPLIT, SMOTE, SCALE
# ══════════════════════════════════════════════════════════════════════════════

def split_smote_scale(X: pd.DataFrame, y: pd.Series,
                      feature_names: list) -> tuple:
    """
    Stratified train/test split → SMOTE on training set only → StandardScaler.

    CRITICAL: SMOTE and scaler are fit ONLY on training data to prevent leakage.
    The test set is transformed but never used to fit anything.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    print("[8/9] Splitting → SMOTE → Scaling...")

    # ── Stratified split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train : {X_train.shape} | classes: {y_train.value_counts().sort_index().to_dict()}")
    print(f"      Test  : {X_test.shape}  | classes: {y_test.value_counts().sort_index().to_dict()}")

    # ── SMOTE (training only!) ────────────────────────────────────────────────
    min_class_count = y_train.value_counts().min()
    k = min(SMOTE_K, min_class_count - 1)   # k must be < smallest class size
    if k < 1:
        print(f"      ⚠ Too few minority samples for SMOTE — using class_weight instead")
        X_train_sm, y_train_sm = X_train.copy(), y_train.copy()
    else:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        print(f"      After SMOTE : {X_train_sm.shape} | classes: "
              f"{pd.Series(y_train_sm).value_counts().sort_index().to_dict()}")

    # ── StandardScaler (fit on train, transform both) ─────────────────────────
    scale_cols = [c for c in SCALE_COLS if c in feature_names]
    scaler = StandardScaler()

    X_train_sm_df = pd.DataFrame(X_train_sm, columns=feature_names)
    X_test_df     = pd.DataFrame(X_test,     columns=feature_names)

    X_train_sm_df[scale_cols] = scaler.fit_transform(X_train_sm_df[scale_cols])
    X_test_df[scale_cols]     = scaler.transform(X_test_df[scale_cols])

    print(f"      Scaled {len(scale_cols)} continuous features")
    return X_train_sm_df, X_test_df, pd.Series(y_train_sm), y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(X_train, X_test, y_train, y_test,
                 scaler, knn_imp, med_imp, feature_names: list) -> None:
    """Save all pipeline artifacts to OUTPUT_DIR."""
    print("[9/9] Saving pipeline artifacts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv",   index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False, header=True)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv",   index=False, header=True)
    pd.Series(feature_names, name="feature").to_csv(
        f"{OUTPUT_DIR}/feature_names.csv", index=False
    )

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb")  as f: pickle.dump(scaler,  f)
    with open(f"{OUTPUT_DIR}/knn_imp.pkl", "wb") as f: pickle.dump(knn_imp, f)
    with open(f"{OUTPUT_DIR}/med_imp.pkl", "wb") as f: pickle.dump(med_imp, f)

    # Pipeline report
    report = textwrap.dedent(f"""
    PREPROCESSING PIPELINE REPORT
    ==============================
    Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Source files
    ------------
    Accidents  : {ACCIDENTS_CSV}
    Weather    : {WEATHER_XLSX}
    OSM roads  : {OSM_ROADS_CSV or 'not used'}

    Dataset sizes
    -------------
    Training (post-SMOTE) : {X_train.shape[0]} rows × {X_train.shape[1]} features
    Test (held-out)       : {X_test.shape[0]} rows × {X_test.shape[1]} features
    Test size             : {TEST_SIZE:.0%}

    Class distribution
    ------------------
    Training (after SMOTE):
      Low    (0) : {(y_train==0).sum()}
      Medium (1) : {(y_train==1).sum()}
      High   (2) : {(y_train==2).sum()}
    Test (original):
      Low    (0) : {(y_test==0).sum()}
      Medium (1) : {(y_test==1).sum()}
      High   (2) : {(y_test==2).sum()}

    Features ({len(feature_names)} total)
    --------
    {chr(10).join('  ' + f for f in feature_names)}

    Leakage prevention
    ------------------
    Dropped columns : {LEAKAGE_COLS}
    SMOTE applied   : training set ONLY (before: imbalanced, after: balanced)
    Scaler fitted   : training set ONLY (test transformed with train fit)

    Artifacts saved
    ---------------
    {OUTPUT_DIR}/X_train.csv       — Training features (SMOTE + scaled)
    {OUTPUT_DIR}/X_test.csv        — Test features (scaled, no leakage)
    {OUTPUT_DIR}/y_train.csv       — Training labels
    {OUTPUT_DIR}/y_test.csv        — Test labels
    {OUTPUT_DIR}/feature_names.csv — Ordered feature list
    {OUTPUT_DIR}/scaler.pkl        — Fitted StandardScaler
    {OUTPUT_DIR}/knn_imp.pkl       — Fitted KNN imputer (for inference)
    {OUTPUT_DIR}/med_imp.pkl       — Fitted median imputer (for inference)

    Next step
    ---------
    python train_model.py
    """).strip()

    with open(f"{OUTPUT_DIR}/pipeline_report.txt", "w") as f:
        f.write(report)

    print(f"      Saved to: {OUTPUT_DIR}/")
    print(f"      Artifacts: X_train, X_test, y_train, y_test, feature_names,")
    print(f"                 scaler.pkl, knn_imp.pkl, med_imp.pkl, pipeline_report.txt")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER — use saved artifacts for real-time prediction
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_single_request(request: dict, feature_names: list,
                               scaler, knn_imp, med_imp) -> np.ndarray:
    """
    Transform a single real-time API request into a model-ready feature vector.
    Use this in your FastAPI endpoint.

    Args:
        request: dict with keys matching the feature set (from driver + weather API)
        feature_names: list from feature_names.csv
        scaler, knn_imp, med_imp: loaded from .pkl files

    Returns:
        np.ndarray of shape (1, n_features) — ready for model.predict()

    Example:
        feature_names = pd.read_csv('pipeline_output/feature_names.csv')['feature'].tolist()
        scaler  = pickle.load(open('pipeline_output/scaler.pkl','rb'))
        knn_imp = pickle.load(open('pipeline_output/knn_imp.pkl','rb'))
        med_imp = pickle.load(open('pipeline_output/med_imp.pkl','rb'))
        X = preprocess_single_request(driver_request, feature_names, scaler, knn_imp, med_imp)
        risk = model.predict(X)[0]
    """
    row = {feat: 0 for feat in feature_names}   # default to 0

    # Map request fields directly
    direct = [
        "is_weekend", "hour", "elevation_m", "num_vehicles",
        "driver_age", "driving_exp_yrs", "estimated_speed_kmh", "speed_limit_kmh",
        "speeding", "precipitation_mm", "temp_avg_c", "visibility_km",
        "wind_speed_kmh", "pressure_hpa", "road_slope_deg",
    ]
    for col in direct:
        if col in request and col in row:
            row[col] = request[col]

    # Derived real-time features
    row["vehicle_risk_weight"] = {"motorcycle":4,"lorry":4,"bus":3,"car":2,"three-wheeler":3}.get(
        request.get("vehicle_type","car"), 2)
    row["is_heavy_vehicle"]  = int(request.get("vehicle_type") in ["bus","lorry"])
    row["speed_excess_kmh"]  = max(0, request.get("estimated_speed_kmh",50) - request.get("speed_limit_kmh",50))
    row["speed_ratio"]       = request.get("estimated_speed_kmh",50) / max(1, request.get("speed_limit_kmh",50))
    row["is_night"]          = int(request.get("hour",12) < 6 or request.get("hour",12) >= 20)
    row["is_morning_peak"]   = int(request.get("hour",12) in [7,8,9])
    row["is_evening_peak"]   = int(request.get("hour",12) in [16,17,18,19])
    row["hour_sin"]          = np.sin(2*np.pi*request.get("hour",12)/24)
    row["hour_cos"]          = np.cos(2*np.pi*request.get("hour",12)/24)
    row["is_sharp_curve"]    = int(request.get("road_curvature") == "sharp")
    row["road_curvature_enc"]= {"straight":0,"mild":1,"sharp":2}.get(request.get("road_curvature","straight"),0)
    row["weather_enc"]       = {"clear":0,"light_rain":1,"mist":2,"foggy":3,"heavy_rain":4}.get(
        request.get("weather_condition","clear"),0)
    row["rain_intensity_num"]= 2 if request.get("precipitation_mm",0)>5 else (1 if request.get("precipitation_mm",0)>0.5 else 0)
    row["visibility_cls_enc"]= 3 if request.get("visibility_km",10)<0.3 else (2 if request.get("visibility_km",10)<1 else (1 if request.get("visibility_km",10)<3 else 0))
    row["high_risk_combo"]   = int(request.get("road_curvature")=="sharp" and
                                    request.get("road_slope_deg",0)>10 and
                                    request.get("precipitation_mm",0)>1)

    # One-hot vehicle type
    for vt in ["car","bus","lorry","motorcycle","three-wheeler"]:
        k = f"veh_{vt}"
        if k in row:
            row[k] = int(request.get("vehicle_type") == vt)

    X = np.array([[row[f] for f in feature_names]])

    # Apply same scaling as training
    X_df = pd.DataFrame(X, columns=feature_names)
    scale_cols = [c for c in SCALE_COLS if c in feature_names]
    X_df[scale_cols] = scaler.transform(X_df[scale_cols])

    return X_df.values


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("\n" + "="*60)
    print("  Preprocessing Pipeline — Kadugannawa Accident Prediction")
    print("="*60 + "\n")

    accidents, weather, osm = load_datasets(ACCIDENTS_CSV, WEATHER_XLSX, OSM_ROADS_CSV)
    accidents, weather      = clean_dates(accidents, weather)
    weather                 = clean_weather(weather)
    df                      = merge_datasets(accidents, weather, osm)
    df, knn_imp, med_imp    = impute_missing(df)
    df                      = engineer_features(df)
    X, y, feature_names     = prepare_Xy(df)
    X_train, X_test, y_train, y_test, scaler = split_smote_scale(X, y, feature_names)
    save_outputs(X_train, X_test, y_train, y_test,
                 scaler, knn_imp, med_imp, feature_names)

    print("\n" + "="*60)
    print("  ✓ Pipeline complete. Ready for model training.")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print("="*60)
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    run_pipeline()