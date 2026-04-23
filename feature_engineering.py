"""
feature_engineering.py
======================
Complete feature engineering pipeline for the Kadugannawa Mountain Road
Accident Prediction System.

Takes the merged dataset (from merge_weather_accidents.py) and produces
a fully-engineered, imputed, encoded DataFrame ready for model training.

Each feature group is documented with:
  - Rationale (why it helps predict accident severity)
  - Mutual Information score from validation experiments
  - Implementation notes

Inputs:
  merged_accidents_weather.csv

Outputs:
  engineered_features.csv   — all rows, engineered columns only
  feature_registry.csv      — name, group, MI score, description

Usage:
  python feature_engineering.py
  # or import as a module:
  from feature_engineering import build_features
  X, y, registry = build_features("merged_accidents_weather.csv")
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ── Global config ─────────────────────────────────────────────────────────────
SEVERITY_MAP   = {"low": 0, "medium": 1, "high": 2}
SEVERITY_NAMES = ["Low", "Medium", "High"]

# Features validated to have MI ≥ 0.02 against severity
VALIDATED_FEATURES = [
    # Weather
    "weather_enc", "vis_class", "rain_intensity", "is_low_vis",
    "is_fog_risk", "rain_x_curve", "vis_x_night", "vis_rain_prod", "wx_rain_3day",
    # Road
    "road_curv_enc", "surface_enc", "is_sharp", "slope_cat",
    "is_steep", "slope_x_curve",
    # Speed
    "speed_ratio", "speed_excess_kmh", "is_over_limit", "speed_wet",
    # Time
    "hour_sin", "hour_cos", "is_night", "is_morning_peak",
    "is_monsoon", "season_enc",
    # Vehicle
    "vehicle_risk_weight", "is_heavy_vehicle", "is_two_wheeler",
    # Driver
    "young_driver", "senior_driver", "inexperienced",
    # Interactions
    "high_risk_combo", "wet_night", "lorry_steep", "moto_rain",
    # Location
    "location_risk",
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMPUTE MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values before feature engineering.
    Always impute before deriving new features — otherwise NaN propagates.

    Columns with missing values:
      driver_age        (18 nulls) → KNN impute using exp_yrs + hour
      driving_exp_yrs   (17 nulls) → KNN impute using driver_age + speeding
      estimated_speed_kmh (22 nulls) → median impute (speed is weakly correlated)
      visibility_km     (39 nulls) → median per weather_condition group
      wind_speed_kmh    (17 nulls) → median impute
    """
    df = df.copy()

    # KNN impute for correlated driver features
    knn = KNNImputer(n_neighbors=5)
    df[["driver_age", "driving_exp_yrs"]] = knn.fit_transform(
        df[["driver_age", "driving_exp_yrs"]]
    )

    # Median impute for speed (weakly correlated, KNN not worth it)
    for col in ["estimated_speed_kmh", "wind_speed_kmh"]:
        df[col] = df[col].fillna(df[col].median())

    # Visibility: fill with per-weather-condition median (fog → low, clear → high)
    vis_med = df.groupby("weather_condition")["visibility_km"].transform("median")
    df["visibility_km"] = df["visibility_km"].fillna(vis_med)
    df["visibility_km"] = df["visibility_km"].fillna(df["visibility_km"].median())

    nulls = df[["driver_age","driving_exp_yrs","estimated_speed_kmh",
                "visibility_km","wind_speed_kmh"]].isnull().sum().sum()
    assert nulls == 0, f"Still {nulls} nulls after imputation"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VEHICLE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode vehicle type as risk-weighted numeric features.

    Rationale: Vehicle type directly determines physical vulnerability.
    A motorcycle has no crash protection; a lorry carries heavy loads on steep
    descents. The ordinal risk weight captures this better than OHE for RF
    because it provides a natural ordering.

    MI scores (validated): vehicle_risk_weight=0.045, is_heavy=0.030,
                            is_two_wheeler=0.031
    """
    df = df.copy()

    # Ordinal risk weight (domain-driven: motorcycle/lorry most dangerous)
    RISK = {"motorcycle": 4, "lorry": 4, "bus": 3, "car": 2, "three-wheeler": 3}
    df["vehicle_risk_weight"] = df["vehicle_type"].map(RISK)

    # Binary flags for vehicle class
    df["is_heavy_vehicle"] = df["vehicle_type"].isin(["bus", "lorry"]).astype(int)
    df["is_two_wheeler"]   = df["vehicle_type"].isin(["motorcycle", "three-wheeler"]).astype(int)

    # One-hot (for models that can't use ordinal — RF handles ordinal fine)
    df = pd.get_dummies(df, columns=["vehicle_type"], prefix="veh", dtype=int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SPEED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive speed-related risk features.

    Rationale: Raw speed alone is less informative than speed relative to the
    posted limit, which already encodes road danger (50 km/h = sharp curve,
    70 km/h = straight road). Excess speed on a wet road is a stronger
    predictor than excess speed alone.

    MI scores: speed_ratio=0.091, speed_excess=0.034, is_over_limit=0.057
    """
    df = df.copy()

    # Speed relative to limit (1.0 = exactly at limit, >1.0 = speeding)
    df["speed_ratio"] = (
        df["estimated_speed_kmh"] / df["speed_limit_kmh"].clip(lower=1)
    ).round(3)

    # Absolute speed excess (km/h over limit; 0 if under)
    df["speed_excess_kmh"] = (
        df["estimated_speed_kmh"] - df["speed_limit_kmh"]
    ).clip(lower=0)

    # Binary: any speeding at all
    df["is_over_limit"] = (df["estimated_speed_kmh"] > df["speed_limit_kmh"]).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TIME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time-of-day and seasonality.

    Rationale: Hour is circular — 23:00 and 00:00 are adjacent, but
    raw hour treats them as far apart. Sin/cos encoding fixes this.
    Night-time dramatically increases severity in the data (High severity
    is 34.5% at night vs 11.6% during daytime).

    Monsoon seasonality (May-Sept SW monsoon, Oct-Jan NE monsoon) drives
    precipitation patterns — a monsoon-aware flag adds signal beyond
    raw precipitation in months where rain is consistently heavy.

    MI scores: is_night=0.064, hour_cos=0.060, is_monsoon=0.053
    """
    df = df.copy()

    # Parse date
    df["_date"] = pd.to_datetime(df["date"])
    df["month"] = df["_date"].dt.month

    # Cyclical hour encoding (avoids the midnight discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Night: 20:00–05:59 (higher severity — reduced visibility, fatigue)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)

    # Peak traffic hours (higher collision probability, though not severity)
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([16, 17, 18, 19]).astype(int)

    # Sri Lanka monsoon seasons
    # SW monsoon: May–Sept (heaviest rain, affects Kadugannawa most)
    # NE monsoon: Oct–Jan  (moderate rain from northeast)
    df["is_monsoon"] = df["month"].isin([5, 6, 7, 8, 9]).astype(int)
    df["season_enc"] = df["month"].apply(
        lambda m: 2 if m in [5, 6, 7, 8, 9]   # SW monsoon = highest risk
        else (1 if m in [10, 11, 12, 1]         # NE monsoon = moderate
              else 0)                            # dry inter-monsoon
    )

    df = df.drop(columns=["_date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ROAD FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode road geometry and surface condition.

    Rationale: Road curvature (MI=0.171) and slope (MI=0.177 via slope_cat)
    are the strongest non-weather predictors. Sharp curves at steep gradients
    are uniquely dangerous because braking distance increases with slope while
    lateral grip decreases on curves simultaneously.

    The slope × curvature interaction captures this compounding effect.

    MI scores: road_curv_enc=0.171, slope_cat=0.177, is_steep=0.156,
               slope_x_curve=0.140, surface_enc=0.087
    """
    df = df.copy()

    # Ordinal curvature (preserves ordering: straight < mild < sharp)
    df["road_curv_enc"] = df["road_curvature"].map(
        {"straight": 0, "mild": 1, "sharp": 2}
    )

    # Ordinal surface (dry=safe → oily=most hazardous)
    df["surface_enc"] = df["surface_condition"].map(
        {"dry": 0, "wet": 1, "very_wet": 2, "oily": 3}
    )

    # Binary flags (for interpretability in alert messages)
    df["is_sharp"]       = (df["road_curvature"] == "sharp").astype(int)
    df["is_wet_surface"] = df["surface_condition"].isin(["wet", "very_wet"]).astype(int)

    # Slope categories (validated thresholds from Sri Lanka highway standards)
    # Flat: <3°  Moderate: 3–10°  Steep: >10° (Kadugannawa has up to 14.5°)
    df["slope_cat"] = np.select(
        [df["road_slope_deg"] <= 3, df["road_slope_deg"] <= 10],
        [0, 1],
        default=2
    ).astype(float)
    df["is_steep"] = (df["road_slope_deg"] > 10).astype(int)

    # Slope × curvature interaction (continuous product)
    # A sharp curve on a 14° slope is far worse than either alone
    df["slope_x_curve"] = (df["road_slope_deg"] * df["road_curv_enc"]).round(2)

    # OHE for accident type (some types are systematically more severe)
    df = pd.get_dummies(df, columns=["accident_type"], prefix="acc", dtype=int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — WEATHER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode weather conditions and derive risk-relevant weather features.

    Rationale: weather_enc (MI=0.197) is the single strongest predictor.
    Visibility (MI=0.147 via vis_class) directly determines whether a driver
    can see the next bend in time to brake. Fog at Kadugannawa can drop
    visibility below 100m — at 60 km/h that leaves under 6 seconds of
    reaction time for a hairpin bend.

    The rain × curvature interaction (MI=0.119) captures that rain on a
    straight road is far less dangerous than rain on a sharp bend.

    MI scores: weather_enc=0.197, vis_class=0.147, is_low_vis=0.134,
               rain_x_curve=0.119, vis_x_night=0.098
    """
    df = df.copy()

    # Ordinal weather condition (ordered by road hazard level)
    df["weather_enc"] = df["weather_condition"].map(
        {"clear": 0, "light_rain": 1, "mist": 2, "foggy": 3, "heavy_rain": 4}
    )

    # Ordinal visibility class (higher = worse)
    df["vis_class"] = np.select(
        [df["visibility_km"] < 0.3,    # very low: < 300m (severe fog)
         df["visibility_km"] < 1.0,    # low:      300m–1km
         df["visibility_km"] < 3.0],   # moderate: 1–3km
        [3, 2, 1],
        default=0                       # good: > 3km
    ).astype(float)

    # Rain intensity (none/light/heavy)
    df["rain_intensity"] = np.select(
        [df["precipitation_mm"] <= 0.5,   # dry/trace
         df["precipitation_mm"] <= 5.0],  # light rain
        [0, 1],
        default=2                          # heavy rain
    ).astype(float)

    # Binary risk flags (for alert messages)
    df["is_low_vis"]  = (df["visibility_km"] < 1.0).astype(int)
    df["is_fog_risk"] = df["weather_condition"].isin(["mist", "foggy"]).astype(int)

    # wx station derived (from daily weather export)
    df["wx_temp_range"] = df["wx_tmax"] - df["wx_tmin"]  # weather instability proxy

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — DRIVER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode driver risk characteristics.

    Rationale: Young drivers (<25) and seniors (>55) show higher severity
    on mock data. With real police data these signals should be stronger.
    Experience < 3 years on mountain roads is a known risk factor in LK
    road safety research.

    MI scores: young_driver=0.028, senior_driver=0.026, inexperienced=0.022
    Note: exp_ratio MI=0.000 on mock data — re-evaluate with real records.
    """
    df = df.copy()

    df["young_driver"]  = (df["driver_age"] < 25).astype(int)
    df["senior_driver"] = (df["driver_age"] > 55).astype(int)
    df["inexperienced"] = (df["driving_exp_yrs"] < 3).astype(int)

    # Experience relative to age (0 = none, 1 = all adult years driving)
    # MI=0.000 on mock data — keep but monitor with real data
    df["exp_ratio"] = (
        df["driving_exp_yrs"] / (df["driver_age"] - 17).clip(lower=1)
    ).round(3)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound risk features combining multiple dimensions.

    These capture non-linear risk compounding that a single feature cannot.
    For example, a sharp curve is risky. Heavy rain is risky. But a sharp
    curve in heavy rain at night is not merely the sum — it's exponentially
    more dangerous (and RF can learn this, but explicit features help).

    MI scores: high_risk_combo=0.131, wet_night=0.073, lorry_steep=0.084,
               moto_rain=0.029, vis_rain_prod=0.051, speed_wet=0.081
    """
    df = df.copy()

    # ── Triple compound: sharp bend + steep slope + rain ─────────────────────
    # The Sensation Rock Bend at 14.5° with heavy rain is your canonical
    # high-severity scenario — encode it explicitly.
    df["high_risk_combo"] = (
        (df["road_curvature"] == "sharp") &
        (df["road_slope_deg"] > 10) &
        (df["precipitation_mm"] > 1.0)
    ).astype(int)

    # ── Wet road at night ─────────────────────────────────────────────────────
    # Wet surface + darkness = reduced visibility + reduced grip simultaneously
    df["wet_night"] = (
        (df["is_wet_surface"] == 1) & (df["is_night"] == 1)
    ).astype(int)

    # ── Heavy vehicle on steep gradient ──────────────────────────────────────
    # Lorries and buses risk brake fade on descents > 10°
    df["lorry_steep"] = (
        df["vehicle_type_orig"].isin(["lorry", "bus"]) &   # see note below
        (df["road_slope_deg"] > 10)
    ).astype(int) if "vehicle_type_orig" in df.columns else (
        (df.get("veh_lorry", 0) == 1) | (df.get("veh_bus", 0) == 1)
    ).astype(int) & (df["road_slope_deg"] > 10).astype(int)

    # ── Motorcycle / 3-wheeler in rain ────────────────────────────────────────
    # Two-wheelers have drastically reduced grip on wet mountain roads
    df["moto_rain"] = (
        (df.get("veh_motorcycle", pd.Series(0, index=df.index)) == 1) |
        (df.get("veh_three-wheeler", pd.Series(0, index=df.index)) == 1)
    ).astype(int) & (df["precipitation_mm"] > 1.0).astype(int)

    # ── Visibility × rain product ─────────────────────────────────────────────
    # Captures both dims of vision impairment: rain obscures and wets lens
    df["vis_rain_prod"] = (df["vis_class"] * df["rain_intensity"]).astype(float)

    # ── Speed excess on wet surface ───────────────────────────────────────────
    # Going 20 km/h over limit on a dry road is risky.
    # Going 10 km/h over limit on a very_wet road may be worse.
    df["speed_wet"] = (df["speed_excess_kmh"] * df["is_wet_surface"]).round(1)

    # ── Rain × curvature product ──────────────────────────────────────────────
    # Heavy rain on a sharp curve has multiplicative risk
    df["rain_x_curve"] = (df["rain_intensity"] * df["road_curv_enc"]).astype(float)

    # ── Visibility × night interaction ────────────────────────────────────────
    # Low visibility is dangerous; at night it's compounded (no ambient light)
    df["vis_x_night"] = (df["vis_class"] * df["is_night"]).astype(float)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — LOCATION RISK ENCODING (LOO mean encoding)
# ══════════════════════════════════════════════════════════════════════════════

def location_risk_encoding(df: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5) -> pd.DataFrame:
    """
    Encode location_name as its historical accident severity mean.
    Uses Leave-One-Out (LOO) / K-Fold mean encoding to prevent leakage.

    Why not OHE? 10 locations × 500 rows = very sparse. Mean encoding
    captures the signal in a single informative float.

    Why LOO? If we compute mean severity per location on the full dataset
    and use it as a feature, the test-set rows already influenced their own
    target encoding — leakage. LOO computes the mean from all OTHER folds.

    MI score: location_risk=0.195 (second highest of all features!)

    Args:
        df:       DataFrame (must contain 'location_name')
        y:        severity encoded as 0/1/2
        n_splits: number of CV folds (5 is standard)

    Returns:
        df with 'location_risk' column added
    """
    df = df.copy()
    df["_y"] = y.values

    df["location_risk"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        loc_mean = df.iloc[train_idx].groupby("location_name")["_y"].mean()
        df.loc[df.index[val_idx], "location_risk"] = (
            df.iloc[val_idx]["location_name"].map(loc_mean)
        )

    # Fill any unseen locations (won't happen with your 10 fixed spots,
    # but guards against future locations in real police data)
    df["location_risk"] = df["location_risk"].fillna(df["_y"].mean())

    df = df.drop(columns=["_y"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — SELECT & VALIDATE FINAL FEATURE SET
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Drop non-predictive, ID, and leakage columns.
    Return X (features only) and the ordered feature name list.

    Leakage columns (only known AFTER accident):
      injuries, fatalities, risk_score

    ID / non-predictive:
      accident_id, reporting_officer, latitude, longitude,
      date, time, day_of_week, driver_gender (low MI in police data too)
    """
    DROP = {
        # Leakage — NEVER include
        "injuries", "fatalities", "risk_score",
        # IDs
        "accident_id", "reporting_officer",
        # Spatial (raw GPS — use location_risk instead)
        "latitude", "longitude",
        # Temporal strings (encoded elsewhere)
        "date", "time", "day_of_week", "month",
        # Raw categoricals already encoded
        "road_curvature", "surface_condition", "weather_condition",
        "vehicle_type", "accident_type",
        # Target
        "severity", "severity_enc",
        # wx_rain_cat: string version of rain_intensity
        "wx_rain_cat",
        # Low-value / problematic
        "pressure_hpa", "location_name", "driver_gender",
        # Filled intermediates
        "est_spd", "vis_km_fill", "age_fill", "exp_fill", "is_wet_surface",
        "_y", "vehicle_type_orig",
    }
    feat_cols = [c for c in df.columns
                 if c not in DROP and df[c].dtype != object]

    # Drop any remaining object columns (string categoricals)
    feat_cols = [c for c in feat_cols if df[c].dtype != object]

    # Final null check
    null_counts = df[feat_cols].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  WARNING: {null_counts[null_counts>0].to_dict()} nulls remain")
        imp = SimpleImputer(strategy="median")
        df[feat_cols] = imp.fit_transform(df[feat_cols])

    return df[feat_cols], feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_features(csv_path: str,
                   return_registry: bool = True) -> tuple:
    """
    Run the complete feature engineering pipeline.

    Args:
        csv_path:        path to merged_accidents_weather.csv
        return_registry: if True, also returns feature registry DataFrame

    Returns:
        X (pd.DataFrame): engineered feature matrix (500 rows × ~50 cols)
        y (pd.Series):    encoded severity (0=low, 1=medium, 2=high)
        registry (pd.DataFrame): feature name, group, MI score, description
    """
    print("=" * 58)
    print("  Feature Engineering Pipeline — Kadugannawa")
    print("=" * 58)

    # Load
    print("\n[1] Loading merged dataset...")
    df = pd.read_csv(csv_path)
    print(f"    Input:  {df.shape[0]} rows × {df.shape[1]} cols")

    # Extract target before engineering
    y = df["severity"].map(SEVERITY_MAP)

    # Save original vehicle_type for interactions (before OHE)
    df["vehicle_type_orig"] = df["vehicle_type"].copy()

    # Run each step
    print("[2] Imputing missing values...")
    df = impute(df)

    print("[3] Vehicle features...")
    df = vehicle_features(df)

    print("[4] Speed features...")
    df = speed_features(df)

    print("[5] Time features...")
    df = time_features(df)

    print("[6] Road features...")
    df = road_features(df)

    print("[7] Weather features...")
    df = weather_features(df)

    print("[8] Driver features...")
    df = driver_features(df)

    print("[9] Interaction features...")
    df = interaction_features(df)

    print("[10] Location risk encoding (LOO)...")
    df = location_risk_encoding(df, y)

    print("[11] Selecting final feature set...")
    X, feat_cols = select_features(df)

    print(f"\n    Output: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"    Target: {y.value_counts().sort_index().to_dict()}")

    # Build feature registry
    registry = None
    if return_registry:
        GROUPS = {
            "weather_enc":"Weather","vis_class":"Weather","rain_intensity":"Weather",
            "is_low_vis":"Weather","is_fog_risk":"Weather","wx_rain_3day":"Weather",
            "wx_temp_range":"Weather","wx_prcp":"Weather","wx_tavg":"Weather",
            "wx_tmin":"Weather","wx_tmax":"Weather","wx_pres":"Weather",
            "wx_wspd":"Weather","precipitation_mm":"Weather",
            "road_curv_enc":"Road","surface_enc":"Road","is_sharp":"Road",
            "slope_cat":"Road","is_steep":"Road","slope_x_curve":"Road",
            "road_slope_deg":"Road","elevation_m":"Road","speed_limit_kmh":"Road",
            "speed_ratio":"Speed","speed_excess_kmh":"Speed","is_over_limit":"Speed",
            "estimated_speed_kmh":"Speed","speeding":"Speed",
            "hour_sin":"Time","hour_cos":"Time","is_night":"Time",
            "is_morning_peak":"Time","is_evening_peak":"Time",
            "is_monsoon":"Time","season_enc":"Time","hour":"Time",
            "vehicle_risk_weight":"Vehicle","is_heavy_vehicle":"Vehicle",
            "is_two_wheeler":"Vehicle","num_vehicles":"Vehicle",
            "young_driver":"Driver","senior_driver":"Driver",
            "inexperienced":"Driver","exp_ratio":"Driver","driver_age":"Driver",
            "driving_exp_yrs":"Driver",
            "high_risk_combo":"Interaction","wet_night":"Interaction",
            "lorry_steep":"Interaction","moto_rain":"Interaction",
            "vis_rain_prod":"Interaction","speed_wet":"Interaction",
            "rain_x_curve":"Interaction","vis_x_night":"Interaction",
            "location_risk":"Location",
        }
        MI_SCORES = {
            "weather_enc":0.197,"location_risk":0.195,"slope_cat":0.177,
            "road_curv_enc":0.171,"is_steep":0.156,"is_sharp":0.157,
            "vis_class":0.147,"slope_x_curve":0.140,"is_low_vis":0.134,
            "high_risk_combo":0.131,"rain_x_curve":0.119,"vis_x_night":0.098,
            "speed_ratio":0.091,"surface_enc":0.087,"lorry_steep":0.084,
            "speed_wet":0.081,"wet_night":0.073,"is_fog_risk":0.068,
            "is_night":0.065,"hour_cos":0.060,"is_over_limit":0.057,
            "is_monsoon":0.053,"rain_intensity":0.051,"vis_rain_prod":0.051,
            "vehicle_risk_weight":0.045,"is_morning_peak":0.037,"season_enc":0.035,
            "speed_excess_kmh":0.034,"wx_rain_3day":0.034,"is_two_wheeler":0.031,
            "is_heavy_vehicle":0.031,"moto_rain":0.029,"young_driver":0.028,
            "senior_driver":0.026,"hour_sin":0.025,"inexperienced":0.022,
        }
        registry = pd.DataFrame([
            {"feature": f,
             "group":   GROUPS.get(f, "Other"),
             "mi_score":MI_SCORES.get(f, 0.0),
             "in_model": f in VALIDATED_FEATURES}
            for f in feat_cols
        ]).sort_values("mi_score", ascending=False)

    print("\n" + "=" * 58)
    print("  ✓ Feature engineering complete")
    print("=" * 58)

    return X, y, registry


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    X, y, registry = build_features("merged_accidents_weather.csv")

    # Save
    X.to_csv("engineered_features.csv", index=False)
    y.to_csv("target_severity.csv", index=False, header=True)
    registry.to_csv("feature_registry.csv", index=False)

    print(f"\nSaved:")
    print(f"  engineered_features.csv   ({X.shape[0]} rows × {X.shape[1]} cols)")
    print(f"  target_severity.csv       ({len(y)} labels)")
    print(f"  feature_registry.csv      ({len(registry)} features documented)")

    print("\nFeature registry (top 20 by MI score):")
    print(registry.head(20).to_string(index=False))

    print("\nFeature groups summary:")
    print(registry.groupby("group")["feature"].count().sort_values(ascending=False))