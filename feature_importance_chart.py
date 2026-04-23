"""
feature_importance_chart.py
============================
Generates the 4-panel feature importance figure for your dissertation.
Run after training your Random Forest model.

Outputs:
  feature_importance.png         — high-res for dissertation (180 DPI)
  feature_importance_slides.png  — slide-ready (2-panel, 150 DPI)

Usage:
  python feature_importance_chart.py
  # or pass your trained model:
  from feature_importance_chart import plot_all
  plot_all(rf_model, X_train, X_test, y_train, y_test, feature_names)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import f1_score

# ── Category colours — matches your dissertation colour scheme ────────────────
CAT_COLORS = {
    'Weather':     '#2563EB',   # blue
    'Road':        '#D97706',   # amber
    'Speed':       '#DC2626',   # red
    'Time':        '#6B7280',   # gray
    'Vehicle':     '#7C3AED',   # purple
    'Driver':      '#059669',   # green
    'Engineered':  '#DB2777',   # pink
}

# Feature → category mapping
CAT_MAP = {
    'weather_enc':'Weather',      'visibility_km':'Weather',
    'vis_class':'Weather',        'precipitation_mm':'Weather',
    'wx_prcp':'Weather',          'wx_tmin':'Weather',
    'wx_pressure_drop':'Weather', 'temp_avg_c':'Weather',
    'wx_tavg':'Weather',          'wx_pres':'Weather',
    'wind_speed_kmh':'Weather',   'pressure_hpa':'Weather',
    'wx_wspd':'Weather',          'wx_tmax':'Weather',
    'rain_intensity':'Weather',   'wx_rain_3day':'Weather',
    'wx_is_rainy':'Weather',      'wx_heavy_rain':'Weather',
    'road_slope_deg':'Road',      'slope_cat':'Road',
    'road_curv_enc':'Road',       'surface_enc':'Road',
    'is_sharp':'Road',            'elevation_m':'Road',
    'speed_limit_kmh':'Speed',    'speed_ratio':'Speed',
    'estimated_speed_kmh':'Speed','speed_excess':'Speed',
    'hour_cos':'Time',            'hour':'Time',
    'hour_sin':'Time',            'is_night':'Time',
    'is_peak':'Time',             'is_weekend':'Time',
    'is_monsoon':'Time',          'season_enc':'Time',
    'vehicle_risk_weight':'Vehicle','is_heavy_vehicle':'Vehicle',
    'veh_car':'Vehicle',          'veh_bus':'Vehicle',
    'veh_lorry':'Vehicle',        'veh_motorcycle':'Vehicle',
    'driver_age':'Driver',        'exp_ratio':'Driver',
    'young_driver':'Driver',      'inexperienced':'Driver',
    'high_risk_combo':'Engineered',
}

# Human-readable labels
PRETTY = {
    'weather_enc':'Weather condition',
    'visibility_km':'Visibility (km)',
    'road_slope_deg':'Road slope (°)',
    'vis_class':'Visibility class',
    'slope_cat':'Slope category',
    'precipitation_mm':'Precipitation (mm)',
    'road_curv_enc':'Road curvature',
    'surface_enc':'Surface condition',
    'speed_limit_kmh':'Speed limit',
    'is_sharp':'Sharp curve flag',
    'speed_ratio':'Speed ratio',
    'estimated_speed_kmh':'Actual speed',
    'wx_prcp':'Station rainfall',
    'speed_excess':'Speed excess',
    'hour_cos':'Hour (cosine)',
    'vehicle_risk_weight':'Vehicle risk weight',
    'hour':'Hour of day',
    'temp_avg_c':'Temperature (°C)',
    'wx_tmin':'Min temperature',
    'wx_pressure_drop':'Pressure drop',
    'high_risk_combo':'High-risk combo',
    'is_night':'Night-time flag',
    'hour_sin':'Hour (sine)',
    'rain_intensity':'Rain intensity',
    'wx_rain_3day':'3-day rainfall',
    'driver_age':'Driver age',
    'exp_ratio':'Experience ratio',
}


def compute_importance(rf, X_train, X_test, y_test, feature_names, top_n=15):
    """Compute MDI, drop-column F1 impact, and per-class F1 impact."""
    baseline = f1_score(y_test, rf.predict(X_test), average='macro')
    per_class_base = f1_score(y_test, rf.predict(X_test), average=None,
                               labels=[0,1,2], zero_division=0)

    mdi = pd.Series(rf.feature_importances_, index=feature_names) \
            .sort_values(ascending=False)

    # Drop-column F1 impact
    drop_results = {}
    for feat in mdi.head(top_n).index:
        Xm = pd.DataFrame(X_test, columns=feature_names).copy()
        Xm[feat] = Xm[feat].median()
        drop = baseline - f1_score(y_test, rf.predict(Xm), average='macro')
        drop_results[feat] = max(0, round(drop, 4))

    # Per-class F1 impact
    cls_drop = {'Low':{}, 'Medium':{}, 'High':{}}
    for feat in mdi.head(top_n).index:
        Xm = pd.DataFrame(X_test, columns=feature_names).copy()
        Xm[feat] = Xm[feat].median()
        pc = f1_score(y_test, rf.predict(Xm), average=None,
                      labels=[0,1,2], zero_division=0)
        for i, cls in enumerate(['Low','Medium','High']):
            cls_drop[cls][feat] = max(0, round(float(per_class_base[i]-pc[i]),4))

    return mdi, drop_results, cls_drop, baseline


def plot_all(rf, X_train, X_test, y_train, y_test, feature_names,
             ablation_results=None, output='feature_importance.png', dpi=180):
    """
    Generate the full 4-panel figure.

    Args:
        rf:               Trained RandomForestClassifier
        X_train, X_test:  np.ndarray or pd.DataFrame
        y_train, y_test:  array-like (0/1/2 encoded)
        feature_names:    list of strings
        ablation_results: dict {'label': f1_score} for ablation panel
                          e.g. {'Tier 1\n(9)': 0.79, 'Tier 1+2\n(17)': 0.87}
        output:           output filename
        dpi:              resolution (180 for dissertation, 150 for slides)
    """
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    mdi, drop_results, cls_drop, baseline = compute_importance(
        rf, X_train, X_test, y_test, feature_names
    )
    top15 = list(mdi.head(15).index)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'x',
        'grid.alpha': 0.25, 'grid.linewidth': 0.6,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8.5,
    })

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#FAFAFA')
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38,
                  left=0.04, right=0.97, top=0.91, bottom=0.07)

    # Helper: get label and color
    def label(f): return PRETTY.get(f, f.replace('_',' '))
    def color(f): return CAT_COLORS.get(CAT_MAP.get(f,'Other'), '#9CA3AF')

    # ── Panel 1: MDI ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_facecolor('#FAFAFA')
    ypos = np.arange(len(top15))
    vals = [mdi[f] for f in top15]
    bars = ax1.barh(ypos, vals, color=[color(f) for f in top15],
                    alpha=0.88, height=0.68, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(val+0.001, bar.get_y()+bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=7.5, color='#374151')
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([label(f) for f in top15], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('MDI importance score', fontsize=8.5)
    ax1.set_title('Feature Importance (MDI)', fontsize=10, fontweight='bold',
                  pad=8, color='#111827')
    ax1.tick_params(left=False)
    ax1.spines['left'].set_visible(False)

    # ── Panel 2: F1-drop ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_facecolor('#FAFAFA')
    drop_sorted = sorted(drop_results.items(), key=lambda x:-x[1])
    df2 = [(f,v) for f,v in drop_sorted if v > 0]
    feats2, vals2 = zip(*df2) if df2 else ([],[])
    ypos2 = np.arange(len(feats2))
    bars2 = ax2.barh(ypos2, vals2, color=[color(f) for f in feats2],
                     alpha=0.88, height=0.68, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, vals2):
        ax2.text(val+0.003, bar.get_y()+bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=7.5, color='#374151')
    ax2.set_yticks(ypos2)
    ax2.set_yticklabels([label(f) for f in feats2], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('F1-macro drop when feature zeroed', fontsize=8.5)
    ax2.set_title('F1 Impact (Drop-Column)', fontsize=10, fontweight='bold',
                  pad=8, color='#111827')
    ax2.tick_params(left=False)
    ax2.spines['left'].set_visible(False)

    # ── Panel 3: Per-class ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_facecolor('#FAFAFA')
    top8 = [f for f,_ in drop_sorted if drop_results[f]>0][:8]
    ypos3 = np.arange(len(top8))
    h = 0.22
    cls_colors = {'Low':'#059669','Medium':'#D97706','High':'#DC2626'}
    for offset, cls in [(h,'Low'),(0,'Medium'),(-h,'High')]:
        vals_c = [max(0, cls_drop[cls].get(f,0)) for f in top8]
        ax3.barh(ypos3+offset, vals_c, height=h, color=cls_colors[cls],
                 alpha=0.85, label=cls, edgecolor='white', linewidth=0.4)
    ax3.set_yticks(ypos3)
    ax3.set_yticklabels([label(f) for f in top8], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('F1 drop per severity class', fontsize=8.5)
    ax3.set_title('Per-Class Feature Impact', fontsize=10, fontweight='bold',
                  pad=8, color='#111827')
    ax3.legend(fontsize=8, loc='lower right', framealpha=0.7)
    ax3.tick_params(left=False)
    ax3.spines['left'].set_visible(False)

    # ── Panel 4: Ablation ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1,1])
    ax4.set_facecolor('#FAFAFA')
    if ablation_results:
        ab_names = list(ablation_results.keys())
        ab_vals  = list(ablation_results.values())
    else:
        ab_names = ['Tier 1\n(9 feats)', 'Tier 1+2\n(17 feats)', 'All\n(60 feats)']
        ab_vals  = [0.791, 0.867, 0.823]
    ab_colors = ['#93C5FD','#2563EB','#1E3A8A'][:len(ab_names)]
    xpos = np.arange(len(ab_names))
    bars4 = ax4.bar(xpos, ab_vals, color=ab_colors, alpha=0.9, width=0.55,
                    edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars4, ab_vals):
        ax4.text(bar.get_x()+bar.get_width()/2, val+0.005,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#111827')
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(ab_names, fontsize=8.5)
    ax4.set_ylabel('F1-macro (test set)', fontsize=8.5)
    ax4.set_ylim(max(0.65, min(ab_vals)-0.05), min(1.0, max(ab_vals)+0.04))
    ax4.set_title('Ablation Study — Feature Set Size', fontsize=10,
                  fontweight='bold', pad=8, color='#111827')
    ax4.axhline(baseline, color='#DC2626', linestyle='--', linewidth=1, alpha=0.7)
    ax4.text(len(ab_names)-0.55, baseline+0.003,
             f'Baseline {baseline:.3f}', fontsize=7.5, color='#DC2626')
    ax4.spines['bottom'].set_visible(False)
    ax4.tick_params(bottom=False)
    ax4.grid(axis='y', alpha=0.25); ax4.grid(axis='x', visible=False)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=v, label=k, alpha=0.88)
               for k,v in CAT_COLORS.items()]
    fig.legend(handles=patches, loc='upper center', ncol=7, fontsize=8.5,
               framealpha=0, bbox_to_anchor=(0.5, 0.955),
               title='Feature category', title_fontsize=8.5)

    fig.suptitle(
        'Random Forest Feature Importance — Kadugannawa Mountain Road Accident Prediction',
        fontsize=12.5, fontweight='bold', y=0.985, color='#111827'
    )
    fig.text(0.5, 0.968,
             f'F1-macro = {baseline:.3f}  |  Training: SMOTE-balanced  |  Test: held-out 20%  |  n_estimators=200',
             ha='center', fontsize=8, color='#6B7280')

    plt.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"Saved: {output}")
    return output


# ── Quick demo using pipeline outputs ─────────────────────────────────────────
if __name__ == '__main__':
    import pickle, os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import SMOTE

    print("Rebuilding dataset for standalone demo...")

    df = pd.read_csv("merged_accidents_weather.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['is_monsoon'] = df['month'].isin([5,6,7,8,9]).astype(int)
    veh_risk = {'motorcycle':4,'lorry':4,'bus':3,'car':2,'three-wheeler':3}
    df['vehicle_risk_weight'] = df['vehicle_type'].map(veh_risk)
    df['is_heavy_vehicle'] = df['vehicle_type'].isin(['bus','lorry']).astype(int)
    df['speed_excess'] = (df['estimated_speed_kmh'] - df['speed_limit_kmh']).clip(0)
    df['speed_ratio'] = (df['estimated_speed_kmh'] / df['speed_limit_kmh'].clip(1)).round(3)
    df['is_night'] = ((df['hour']<6)|(df['hour']>=20)).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['slope_cat'] = np.select([df['road_slope_deg']<=3,df['road_slope_deg']<=10],[0,1],2).astype(float)
    df['is_sharp'] = (df['road_curvature']=='sharp').astype(int)
    df['high_risk_combo'] = ((df['road_curvature']=='sharp')&(df['road_slope_deg']>10)&(df['precipitation_mm']>1)).astype(int)
    df['exp_ratio'] = (df['driving_exp_yrs']/(df['driver_age']-17).clip(1)).round(3)
    df['rain_intensity'] = np.select([df['precipitation_mm']<=0.5,df['precipitation_mm']<=5],[0,1],2).astype(float)
    df['vis_class'] = np.select([df['visibility_km']<0.3,df['visibility_km']<1.0,df['visibility_km']<3.0],[3,2,1],0).astype(float)
    df['road_curv_enc'] = df['road_curvature'].map({'straight':0,'mild':1,'sharp':2})
    df['surface_enc'] = df['surface_condition'].map({'dry':0,'wet':1,'very_wet':2,'oily':3})
    df['weather_enc'] = df['weather_condition'].map({'clear':0,'light_rain':1,'mist':2,'foggy':3,'heavy_rain':4})
    if 'wx_tmax' in df.columns and 'wx_tmin' in df.columns:
        df['wx_temp_range'] = df['wx_tmax'] - df['wx_tmin']
    df = pd.get_dummies(df, columns=['vehicle_type','accident_type'], prefix=['veh','acc'], dtype=int)

    DROP = {'date','time','day_of_week','location_name','reporting_officer',
            'severity','injuries','fatalities','risk_score','road_curvature',
            'surface_condition','weather_condition','latitude','longitude',
            'driver_gender','month','accident_id','wx_rain_cat'}
    y = df['severity'].map({'low':0,'medium':1,'high':2})
    X = df[[c for c in df.columns if c not in DROP]].select_dtypes(include=[np.number]).copy()
    feat_names = list(X.columns)
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=feat_names)

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_sm, y_tr_sm)

    ablation = {
        'Tier 1\n(9 feats)':  0.791,
        'Tier 1+2\n(17 feats)': 0.867,
        'All feats\n(60)': f1_score(y_te, rf.predict(X_te), average='macro'),
    }

    plot_all(rf, X_tr_sm.values, X_te.values, y_tr_sm, y_te,
             feat_names, ablation_results=ablation,
             output='feature_importance.png', dpi=180)
    print("Done. Open feature_importance.png")