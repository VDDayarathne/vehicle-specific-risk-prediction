"""
merge_weather_accidents.py
==========================
Merge accident records with daily weather data for Kadugannawa Pass.

Handles the specific challenge of your dataset:
  - Accidents span 2019–2024
  - Weather export spans 2023–2026
  - Only 34% of accidents have a direct date match
  -> Uses 3-tier gap-filling strategy so ALL 500 rows get weather features

Inputs:
  kadugannawa_accidents_mock.csv  (or real police data)
  export.xlsx                     (your weather export)

Output:
  merged_accidents_weather.csv    (500 rows x 45 cols, 0 NaN in wx cols)

Usage:
  python merge_weather_accidents.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.accident_schema import normalize_accident_frame

ACCIDENTS_CSV = str(_ROOT / "data" / "raw" / "kadugannawa_accidents_mock.csv")
WEATHER_XLSX  = str(_ROOT / "data" / "raw" / "export.xlsx")
OUTPUT_CSV    = str(_ROOT / "data" / "processed" / "merged_accidents_weather.csv")


# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load(acc_path: str, wx_path: str):
    acc = pd.read_csv(acc_path)
    acc = normalize_accident_frame(acc)
    wx  = pd.read_excel(wx_path)
    print(f"Loaded: {len(acc)} accidents, {len(wx)} weather rows")
    return acc, wx


# ── Step 2: Standardise dates ─────────────────────────────────────────────────
def standardise_dates(acc: pd.DataFrame, wx: pd.DataFrame):
    """
    Both datasets must use the same date format for merge to work.
    Your weather export has timestamps ('2023-01-01 00:00:00') — strip them.
    """
    acc = acc.copy()
    wx  = wx.copy()

    # .dt.normalize() sets time to 00:00:00, then strftime strips it
    acc["date"] = pd.to_datetime(acc["date"], errors="coerce") \
                    .dt.normalize().dt.strftime("%Y-%m-%d")
    wx["date"]  = pd.to_datetime(wx["date"],  errors="coerce") \
                    .dt.normalize().dt.strftime("%Y-%m-%d")

    # Drop any rows with unparseable dates
    acc = acc.dropna(subset=["date"])
    wx  = wx.dropna(subset=["date"])

    print(f"Accident dates: {acc['date'].min()} -> {acc['date'].max()}")
    print(f"Weather  dates: {wx['date'].min()}  -> {wx['date'].max()}")

    # Report overlap
    acc_d = pd.to_datetime(acc["date"])
    wx_d  = pd.to_datetime(wx["date"])
    overlap_pct = acc_d.between(wx_d.min(), wx_d.max()).mean()
    print(f"Date overlap:   {overlap_pct:.0%} of accidents match weather dates directly")

    return acc, wx


# ── Step 3: Clean weather export ──────────────────────────────────────────────
def clean_weather(wx: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-null columns (snow, wdir, wpgt, tsun in your export).
    Interpolate gaps with linear interpolation + edge fill.
    """
    wx = wx.copy()

    # Drop columns that are entirely null
    all_null = wx.columns[wx.isnull().all()].tolist()
    if all_null:
        print(f"Dropping all-null cols: {all_null}")
        wx = wx.drop(columns=all_null)

    # Impute remaining gaps (time-series linear interpolation is best here)
    num_cols = wx.select_dtypes(include="number").columns
    for col in num_cols:
        wx[col] = wx[col] \
                    .interpolate(method="linear", limit_direction="both") \
                    .ffill().bfill()

    # Rename to wx_* prefix — prevents column name collision with accident data
    rename = {
        "tavg": "wx_tavg", "tmin": "wx_tmin", "tmax": "wx_tmax",
        "prcp": "wx_prcp", "wspd": "wx_wspd", "pres": "wx_pres",
    }
    wx = wx.rename(columns={k: v for k, v in rename.items() if k in wx.columns})
    print(f"Weather columns: {[c for c in wx.columns if c != 'date']}")
    return wx


# ── Step 4: Left join on date ──────────────────────────────────────────────────
def left_join(acc: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """
    Left join: keep ALL accident rows, attach weather where date matches.
    Rows outside the weather date range get NaN in wx_* columns — handled next.

    Why left join (not inner)?
      inner join would drop the 329 pre-2023 accidents — losing 66% of your data.
      left join preserves all 500 rows and lets you fill gaps in Step 5.
    """
    merged = acc.merge(wx, on="date", how="left")
    matched = merged["wx_tavg"].notna().sum()
    print(f"After left join: {len(merged)} rows  "
          f"({matched} with weather, {len(merged)-matched} without)")
    return merged


# ── Step 5: Three-tier gap filling ────────────────────────────────────────────
def fill_weather_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-tier strategy for the 329 accidents with no direct date match:

    Tier 1 — Accident's own weather columns (from police report / simulation)
              These are already in the accident CSV: precipitation_mm, temp_avg_c, etc.
              Best quality — recorded at the scene.

    Tier 2 — Seasonal monthly average from the weather export
              "January in Kadugannawa averages 25.9°C and 2.5mm rain"
              Good for temperature and pressure; rougher for precipitation.

    Tier 3 — Global median fallback (safety net for any remaining NaN)
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    # Build monthly averages from the weather export
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    monthly_avgs = df.groupby("month")[wx_cols].mean()

    # ── Tier 1: crossfill from accident's own weather columns ─────────────────
    crossfill = {
        "wx_prcp": "precipitation_mm",     # rain recorded at accident scene
        "wx_tavg": "temp_avg_c",           # temperature at time of accident
        "wx_wspd": "wind_speed_kmh",       # wind speed at accident
        "wx_pres": "pressure_hpa",         # barometric pressure
    }
    for wx_col, acc_col in crossfill.items():
        if wx_col in df.columns and acc_col in df.columns:
            n_before = df[wx_col].isna().sum()
            df[wx_col] = df[wx_col].fillna(df[acc_col])
            n_filled = n_before - df[wx_col].isna().sum()
            if n_filled > 0:
                print(f"  Tier 1: {wx_col} <- {acc_col}: {n_filled} cells filled")

    # ── Tier 2: seasonal monthly average for any still-missing wx cols ─────────
    for col in wx_cols:
        na_mask = df[col].isna()
        if na_mask.any():
            filled = df.loc[na_mask, "month"].map(monthly_avgs[col])
            df.loc[na_mask, col] = filled
            n = na_mask.sum() - df[col].isna().sum()
            print(f"  Tier 2: {col} seasonal fill: {n} cells")

    # ── Tier 3: global median safety net ──────────────────────────────────────
    for col in wx_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            print(f"  Tier 3: {col} median fallback used")

    remaining = df[wx_cols].isna().sum().sum()
    print(f"  NaN remaining in wx cols: {remaining}")
    return df


# ── Step 6: Derive ML-ready weather features ──────────────────────────────────
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional predictive features from the raw weather columns.
    These directly match the feature names used in preprocessing_pipeline.py.
    """
    df = df.copy()

    # Temperature spread (proxy for weather stability / mist risk)
    if "wx_tmax" in df.columns and "wx_tmin" in df.columns:
        df["wx_temp_range"] = (df["wx_tmax"] - df["wx_tmin"]).round(1)

    # Rain categories
    if "wx_prcp" in df.columns:
        df["wx_is_rainy"]   = (df["wx_prcp"] > 1.0).astype(int)
        df["wx_heavy_rain"] = (df["wx_prcp"] > 5.0).astype(int)
        df["wx_rain_cat"]   = pd.cut(
            df["wx_prcp"],
            bins=[-0.01, 0.5, 5.0, 999],
            labels=["none", "light", "heavy"],
        )
        # 3-day cumulative rain (saturated roads stay slippery after rain stops)
        df = df.sort_values("date")
        df["wx_rain_3day"] = (
            df["wx_prcp"]
            .rolling(window=3, min_periods=1)
            .sum()
            .round(2)
        )
        df = df.reset_index(drop=True)

    # Pressure drop: falling pressure signals incoming rain/storms
    if "wx_pres" in df.columns:
        df = df.sort_values("date")
        df["wx_pressure_drop"] = df["wx_pres"].diff().fillna(0).round(2)
        df = df.reset_index(drop=True)

    new_cols = [c for c in df.columns if c.startswith("wx_") and
                c not in ["wx_tavg","wx_tmin","wx_tmax","wx_prcp","wx_wspd","wx_pres"]]
    print(f"Derived features: {new_cols}")
    return df


# ── Step 7: Validate and save ─────────────────────────────────────────────────
def validate_and_save(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Final checks before saving:
    - No NaN in wx columns
    - All original accident rows preserved
    - Report column summary
    """
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    null_counts = df[wx_cols].isnull().sum()

    if null_counts.sum() > 0:
        print(f"WARNING: NaN still present: {null_counts[null_counts>0].to_dict()}")
    else:
        print(f"[OK] All {len(wx_cols)} wx columns complete (0 NaN)")

    print(f"[OK] Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[OK] Severity: {df['severity'].value_counts().sort_index().to_dict()}")

    df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def merge_pipeline(acc_path=ACCIDENTS_CSV, wx_path=WEATHER_XLSX, out=OUTPUT_CSV):
    print("=" * 56)
    print("  Weather x Accident Merge — Kadugannawa Pass")
    print("=" * 56)

    print("\n[1] Loading...")
    acc, wx = load(acc_path, wx_path)

    print("\n[2] Standardising dates...")
    acc, wx = standardise_dates(acc, wx)

    print("\n[3] Cleaning weather export...")
    wx = clean_weather(wx)

    print("\n[4] Left joining on date...")
    merged = left_join(acc, wx)

    print("\n[5] Filling weather gaps (3-tier)...")
    merged = fill_weather_gaps(merged)

    print("\n[6] Deriving ML features...")
    merged = derive_features(merged)

    print("\n[7] Validating and saving...")
    merged = validate_and_save(merged, out)

    print("\n" + "=" * 56)
    print("  [OK] Merge complete. Next: preprocessing_pipeline.py")
    print("=" * 56)
    return merged


if __name__ == "__main__":
    df = merge_pipeline()

    # Quick preview
    wx_preview_cols = ["date", "vehicle_type", "severity",
                       "wx_tavg", "wx_prcp", "wx_is_rainy",
                       "wx_temp_range", "wx_pressure_drop"]
    print("\nSample merged rows:")
    print(df[wx_preview_cols].sample(6, random_state=1).to_string(index=False))