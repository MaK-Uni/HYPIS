"""
HyPIS Ug  v8.1  — FAO-56 + XGBoost ML Integration
═══════════════════════════════════════════════════════════════════════════════
FIXES in v8.0 (over v7.9):
  ✔ CRITICAL — ML model loading migrated to st.cache_resource: loads once
    per server process, shows a spinner, and is shared across all sessions.
    Previously used a bare module-level function that ran at import time,
    before any Streamlit UI context existed.
  ✔ CRITICAL — _load_ml_model now validates feature count at load time:
    if model.n_features_in_ != len(ML_FEATURES), status shows a clear
    mismatch message rather than letting predict() crash later.
  ✔ ml_predict_iwr exception now reports type(e).__name__ for clearer
    triage (ValueError vs KeyError vs ImportError are all different problems).
  ✔ get_forecast: bare 'except Exception: return None' replaced with
    separate Timeout and general Exception handlers that surface the actual
    error via st.error() / st.warning().
  ✔ get_historical_weather: same exception surfacing applied.
  ✔ get_current_weather: exception comments added; fallback defaults preserved.
  ✔ Tab 2 IWR=0 UX: three-tier contextual message (SM>=80% / SM>=65% / SM<65%)
    explains WHY IWR is zero so users don't mistake correct behaviour for a bug.
  ✔ Sidebar warning message uses _MODEL_PATH.name dynamically instead of
    a hardcoded filename string.
  ✔ CRITICAL — Leading space removed from " Kampala (Makerere Uni)" key in
    LOCATIONS dict.  The space caused DISTRICT_SOIL.get() to always miss and
    fall back to "Custom Location" defaults (Loam FC=28%) instead of the
    correct Clay Loam values (FC=32%).  All Kampala soil lookups now correct.
  ✔ CRITICAL — MUARiK (Kabanyoro) key case fixed in DISTRICT_SOIL: was
    "MUARiK (kabanyoro)" (lowercase k) → now "MUARiK (Kabanyoro)" matching
    the LOCATIONS key.  MUARiK soil data (Sandy Clay Loam) now loads properly.
  ✔ CRITICAL — get_historical_weather() was requesting "windspeed_10m_max"
    from the ERA5 archive API which only exposes "wind_speed_10m_max"
    (underscore between 'wind' and 'speed').  This caused a KeyError caught
    silently → historical tab returned None for all queries.  Fixed to match
    estimate_sm() which already used the correct field name.
  ✔ Minor — Removed unused variable raw_pct_fc (computed but never read;
    sm_gauge() was called with the inline expression directly).
  ✔ Minor — Removed redundant irr_to_fc / vol_to_fc variables: when
    irrigate=True, irr_to_fc ≡ iwr1 and vol_to_fc ≡ vol1 (since
    nir1=dr_today and iwr1=dr_today/Ef), so the refill box now reuses iwr1/vol1.
  ✔ Docstring updated to v7.7 throughout.

FIXES in v7.6 (retained):
  ✔ CRITICAL — estimate_sm initial SM changed from 70% → 50% of available
    water content.
  ✔ CRITICAL — depletion_status() simplified to canonical FAO-56 rule.
  ✔ Decimal display — explicit .format() on all numeric columns.
  ✔ Today Summary table uses .style.format() instead of bare st.dataframe().

FIXES in v7.5 (retained):
  ✔ Rain always sourced from Open-Meteo forecast (no manual rain input).
  ✔ Forecast filter timezone-aware (Africa/Nairobi).
  ✔ SM% rounding uses round(...,1) not int() truncation.
  ✔ estimate_sm uses ETc = Kc×ET₀.

Author: Prosper BYARUHANGA · HyPIS App v8.1 · FAO-56 PM + XGBoost ML
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time as _time, subprocess, pathlib, io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta, date

_PD_VER = tuple(int(x) for x in pd.__version__.split(".")[:2])

def _styler_map(styler, func, subset=None):
    if _PD_VER >= (2, 0):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)

for _pkg in ("joblib", "xgboost", "openpyxl"):
    try:
        __import__(_pkg)
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", _pkg, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

try:
    import openpyxl  # noqa: F401
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False

st.set_page_config(page_title="HyPIS Ug – Uganda IWR", layout="wide",
                   initial_sidebar_state="expanded")
_HERE = os.path.dirname(os.path.abspath(__file__))

LOCATIONS = {
    "Kampala (Makerere Uni)": ( 0.33396,  32.56801, 1239.01),
    "MUARiK (Kabanyoro)":     ( 0.464533, 32.612517, 1178.97),
    "Mbarara":                 (-0.6133,   30.6544,   1433),
    "Isingiro (Kabuyanda)":   (-0.95658,  30.61432,  1364.59),
    "Gulu":                   ( 2.7746,   32.2990,   1105),
    "Jinja":                  ( 0.4244,   33.2041,   1137),
    "Mbale":                  ( 1.0804,   34.1751,   1155),
    "Kabale":                 (-1.2490,   29.9900,   1869),
    "Fort Portal":            ( 0.6710,   30.2750,   1537),
    "Masaka":                 (-0.3310,   31.7373,   1148),
    "Lira":                   ( 2.2499,   32.9002,   1074),
    "Soroti":                 ( 1.7153,   33.6107,   1130),
    "Arua":                   ( 3.0210,   30.9110,   1047),
    "Hoima":                  ( 1.4352,   31.3524,   1562),
    "Kasese":                 ( 0.1820,   30.0804,    933),
    "Tororo":                 ( 0.6920,   34.1810,   1148),
    "Moroto":                 ( 2.5340,   34.6650,   1390),
    "Custom Location":        (None,      None,      None),
}

DISTRICT_SOIL = {
    "Kampala (Makerere Uni)": {"fc":0.32,"pwp":0.18,"texture":"Clay Loam",       "source":"HWSD v2"},
    "MUARiK (Kabanyoro)":      {"fc":0.26,"pwp":0.12,"texture":"Sandy Clay Loam","source":"HWSD v2"},
    "Mbarara":                 {"fc":0.30,"pwp":0.15,"texture":"Loam",            "source":"HWSD v2"},
    "Isingiro (Kabuyanda)":   {"fc":0.28,"pwp":0.14,"texture":"Loam",            "source":"HWSD v2"},
    "Gulu":                   {"fc":0.24,"pwp":0.11,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Jinja":                  {"fc":0.31,"pwp":0.16,"texture":"Clay Loam",        "source":"HWSD v2"},
    "Mbale":                  {"fc":0.27,"pwp":0.13,"texture":"Loam",             "source":"HWSD v2"},
    "Kabale":                 {"fc":0.33,"pwp":0.19,"texture":"Clay",             "source":"HWSD v2"},
    "Fort Portal":            {"fc":0.29,"pwp":0.14,"texture":"Loam",             "source":"HWSD v2"},
    "Masaka":                 {"fc":0.25,"pwp":0.12,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Lira":                   {"fc":0.23,"pwp":0.10,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Soroti":                 {"fc":0.22,"pwp":0.09,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Arua":                   {"fc":0.21,"pwp":0.08,"texture":"Loamy Sand",       "source":"HWSD v2"},
    "Hoima":                  {"fc":0.28,"pwp":0.13,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Kasese":                 {"fc":0.35,"pwp":0.20,"texture":"Clay",             "source":"HWSD v2"},
    "Tororo":                 {"fc":0.26,"pwp":0.12,"texture":"Sandy Clay Loam",  "source":"HWSD v2"},
    "Moroto":                 {"fc":0.18,"pwp":0.08,"texture":"Sandy Loam",       "source":"HWSD v2"},
    "Custom Location":        {"fc":0.28,"pwp":0.14,"texture":"Loam (default)",   "source":"FAO-56 default"},
}

SOIL_OPTS = {
    "Sand":            {"fc":0.10,"pwp":0.05,"desc":"Very fast drainage, very low retention"},
    "Loamy Sand":      {"fc":0.14,"pwp":0.07,"desc":"Fast drainage, low retention"},
    "Sandy Loam":      {"fc":0.20,"pwp":0.09,"desc":"Moderate drainage, moderate retention"},
    "Sandy Clay Loam": {"fc":0.26,"pwp":0.12,"desc":"Moderate-high retention"},
    "Loam":            {"fc":0.28,"pwp":0.14,"desc":"Good balance of drainage and retention"},
    "Silt Loam":       {"fc":0.31,"pwp":0.15,"desc":"High retention, moderate drainage"},
    "Silt":            {"fc":0.33,"pwp":0.16,"desc":"High retention"},
    "Clay Loam":       {"fc":0.32,"pwp":0.18,"desc":"High retention, slow drainage"},
    "Silty Clay Loam": {"fc":0.35,"pwp":0.20,"desc":"Very high retention"},
    "Sandy Clay":      {"fc":0.28,"pwp":0.16,"desc":"Moderate-high retention"},
    "Silty Clay":      {"fc":0.38,"pwp":0.23,"desc":"Very high retention, poor drainage"},
    "Clay":            {"fc":0.40,"pwp":0.25,"desc":"Maximum retention, waterlogging risk"},
}

TEXTURE_MAD_ADJ = {
    "Sand":            +0.10,
    "Loamy Sand":      +0.08,
    "Sandy Loam":      +0.05,
    "Sandy Clay Loam": -0.03,
    "Loam":            +0.00,
    "Silt Loam":       -0.05,
    "Silt":            -0.05,
    "Clay Loam":       -0.05,
    "Silty Clay Loam": -0.08,
    "Sandy Clay":      -0.05,
    "Silty Clay":      -0.10,
    "Clay":            -0.10,
    "Loam (default)":  +0.00,
}

def adjust_mad_for_soil(mad_crop, texture):
    adj = TEXTURE_MAD_ADJ.get(texture, 0.0)
    if adj == 0.0:
        for k, v in TEXTURE_MAD_ADJ.items():
            if k.lower() in texture.lower():
                adj = v; break
    return round(max(0.10, min(0.90, mad_crop + adj)), 3)

TIMEZONE = "Africa/Nairobi"
_SIGMA   = 4.903e-9
_W2M     = 4.87 / np.log(67.8 * 10.0 - 5.42)

# ── ML MODEL CONFIG ──────────────────────────────────────────────────────────
# v8.0: 8 features — soil_pwp is the 7th feature (was missing in v7.8, causing
#        every prediction to silently return None via swallowed ValueError).
ML_FEATURES = [
    "tmean", "rh", "wind", "kc",
    "precipitation", "soil_fc", "soil_pwp", "root_depth",
]
_MODEL_PATH = pathlib.Path(_HERE) / "irrigation_xgboost_model_with_soil.pkl"

# v8.0: Use st.cache_resource — correct Streamlit primitive for shared ML objects.
# Loads once per server process, survives reruns, shows a spinner during cold load.
# Falls back gracefully if file is missing or version-mismatched.
@st.cache_resource(show_spinner="🤖 Loading XGBoost model…")
def _load_ml_model_cached():
    """
    Load the XGBoost model from disk.
    Returns (model, ok: bool, status_msg: str).
    Using st.cache_resource ensures the model is shared across all sessions
    and only loaded once — not on every page rerun.
    """
    try:
        import joblib
        import xgboost  # noqa — ensure xgboost is importable
        if not _MODEL_PATH.exists():
            return None, False, f"⚠️ Model file not found: {_MODEL_PATH.name}"
        model = joblib.load(str(_MODEL_PATH))
        # Validate feature count matches expectation
        n = getattr(model, "n_features_in_", None)
        if n is not None and n != len(ML_FEATURES):
            return None, False, (
                f"⚠️ Feature count mismatch: model expects {n}, "
                f"ML_FEATURES has {len(ML_FEATURES)}"
            )
        return model, True, "✅ XGBoost model loaded"
    except Exception as e:
        return None, False, f"⚠️ Model load error: {type(e).__name__}: {e}"

_ML_MODEL, ML_OK, ML_STATUS = _load_ml_model_cached()
ML_MODEL = _ML_MODEL  # kept for backwards-compat references

def ml_predict_iwr(tmean, rh, wind, kc, precipitation,
                   soil_fc, soil_pwp, root_depth, taw=None, dr_ratio=None):
    """
    Call the XGBRegressor to predict NIR (net irrigation requirement, mm/day).
    Returns the predicted value clipped to [0, TAW].
    Returns None if the model is unavailable or prediction fails.

    Features — must match model exactly (8 total):
      tmean         mean daily air temperature (°C)
      rh            mean relative humidity (%)
      wind          wind speed at 2 m height (m/s)
      kc            crop coefficient (dimensionless)
      precipitation daily rainfall (mm)
      soil_fc       field capacity, volumetric (0–1)
      soil_pwp      permanent wilting point, volumetric (0–1)
      root_depth    effective rooting depth (m)

    dr_ratio (optional, 0–1):  Dr / TAW at time of prediction.
      This is NOT a model feature (model was not retrained with it) but is
      used as a post-prediction scaling factor to make the stateless ML
      more soil-state-aware:
        · dr_ratio = 0   → soil at FC → scale ML output toward 0
        · dr_ratio = 1   → soil at PWP → ML output unchanged
      This is a lightweight bridge until the model is retrained with Dr/TAW
      as an explicit feature.

    NOTE: The ML model is stateless — it has no knowledge of the current
    soil depletion (Dr). It outputs a pattern-based NIR estimate given
    weather + soil + crop conditions. Divergence from FAO-56 IWR is
    expected when Pe >= ETc (soil is full). FAO-56 is authoritative for
    the irrigation trigger; ML serves as a cross-check.
    """
    if not ML_OK or ML_MODEL is None:
        return None
    try:
        X = pd.DataFrame([[
            float(tmean), float(rh), float(wind), float(kc),
            float(precipitation), float(soil_fc),
            float(soil_pwp), float(root_depth),
        ]], columns=ML_FEATURES)
        pred  = float(ML_MODEL.predict(X)[0])
        upper = float(taw) if taw and taw > 0 else 50.0
        raw_pred = max(0.0, min(upper, pred))

        # dr_ratio scaling: bridge the stateless→state-aware gap.
        # When the soil is near FC (dr_ratio ≈ 0) the ML's raw prediction
        # overstates demand. Scale it proportionally to how depleted the
        # soil actually is, so the ML panel shows realistic values.
        # FAO-56 IWR is unchanged — only the ML cross-check value is adjusted.
        if dr_ratio is not None:
            dr_ratio_clamped = max(0.0, min(1.0, float(dr_ratio)))
            # Use a sqrt curve so scaling isn't too aggressive at mid-depletion
            import math
            scale = math.sqrt(dr_ratio_clamped)
            raw_pred = raw_pred * scale

        return round(raw_pred, 3)
    except Exception as e:
        # Surface the real error into the sidebar ML status panel
        global ML_STATUS
        ML_STATUS = f"⚠️ Prediction error: {type(e).__name__}: {e}"
        return None

def ml_agreement(fao_nir, ml_nir, dr_mm=0., raw_mm=0., rainfall_mm=0.):
    """
    Context-aware FAO-56 vs ML agreement classifier.  v8.0

    Returns: (css_class, icon, short_label, explanation, pct_diff)

    Design principles:
    - Never use ❌ when divergence is architecturally expected
    - ML is stateless; when soil is replenished (Dr≈0) ML will always
      predict demand based on weather alone — that is correct ML behaviour,
      not an error
    - Messaging is calm, actionable, and explains WHY rather than just flagging
    """
    if ml_nir is None:
        return ("ml-panel", "🤖",
                "ML unavailable",
                "Place the model file next to app.py to enable ML cross-check.",
                None)

    # ── Both agree: no irrigation ─────────────────────────────────────────────
    if fao_nir <= 0 and ml_nir <= 0:
        return ("ml-agree", "✅",
                "Both agree — no irrigation needed",
                "FAO-56 and ML both indicate no net water requirement today.", 0.)

    # ── FAO = 0, ML > 0 (most common case — ML is stateless) ─────────────────
    if fao_nir <= 0 and ml_nir > 0:
        dr_str = f"Dr = {dr_mm:.1f} mm (below RAW = {raw_mm:.1f} mm)" if raw_mm > 0 else "soil adequately replenished"
        return (
            "ml-info", "🔵",
            f"ML sees weather demand · FAO-56 confirms soil is adequate",
            (f"ML predicts {ml_nir:.1f} mm based on temperature, humidity, and wind patterns. "
             f"FAO-56 accounts for current soil state ({dr_str}) and confirms no irrigation "
             f"is needed — the soil buffer covers today's crop demand. "
             f"ML is suppressed. This divergence is expected and correct."),
            100.
        )

    # ── FAO > 0, ML ≈ 0 (ML does not confirm irrigation) ────────────────────
    if fao_nir > 0 and ml_nir <= 0:
        return (
            "ml-warn", "⚠️",
            "FAO-56 triggers irrigation — ML does not confirm",
            (f"FAO-56 soil balance indicates root-zone depletion (Dr = {dr_mm:.1f} mm) "
             f"has exceeded the trigger threshold. ML does not confirm, possibly because "
             f"it lacks current soil state. FAO-56 is authoritative — irrigate as scheduled."),
            100.
        )

    # ── Both non-zero: compare magnitude ──────────────────────────────────────
    ref = max(fao_nir, ml_nir, 0.01)
    pct = abs(fao_nir - ml_nir) / ref * 100.

    if pct <= 15:
        return (
            "ml-agree", "✅",
            f"Strong agreement ({pct:.0f}% diff)",
            f"FAO-56 = {fao_nir:.1f} mm · ML = {ml_nir:.1f} mm · Hybrid blends both.", pct
        )
    if pct <= 40:
        return (
            "ml-warn", "⚠️",
            f"Moderate deviation ({pct:.0f}%)",
            (f"FAO-56 = {fao_nir:.1f} mm · ML = {ml_nir:.1f} mm.  "
             f"FAO-56 remains the irrigation trigger."), pct
        )
    return (
        "ml-info", "ℹ️",
        f"ML estimate differs ({pct:.0f}%) — FAO-56 used",
        (f"FAO-56 = {fao_nir:.1f} mm · ML = {ml_nir:.1f} mm.  "
         f"ML is a weather-pattern model and does not track daily soil depletion.  "
         f"FAO-56 is authoritative. The hybrid applies 80/20 weighting, "
         f"keeping ML contribution small when disagreement is large."), pct
    )


# ── HYBRID DECISION ENGINE ────────────────────────────────────────────────────
# ── ML SAFETY FILTER ──────────────────────────────────────────────────────────
_ML_MAX_MM      = 25.0   # safety ceiling on ML prediction (mm/day)
_ML_DR_RATIO_MIN= 0.15   # suppress ML if Dr/TAW < this (soil not depleted enough)
_ML_RAIN_SUPP   = 5.0    # suppress ML if precipitation > this mm

def _constrain_ml(ml: float, dr_mm: float, taw_mm: float, rainfall_mm: float) -> float:
    """Apply safety filters to raw ML prediction."""
    ml = max(0., min(ml, _ML_MAX_MM))
    dr_ratio = dr_mm / taw_mm if taw_mm > 0 else 0.
    if dr_ratio < _ML_DR_RATIO_MIN:
        return 0.
    if rainfall_mm > _ML_RAIN_SUPP:
        return 0.
    return ml



crop_params = {
    "Tomatoes":       {"ini":0.60,"mid":1.15,"end":0.80,"zr":0.70,"mad":0.40},
    "Cabbages":       {"ini":0.70,"mid":1.05,"end":0.95,"zr":0.50,"mad":0.45},
    "Maize":          {"ini":0.30,"mid":1.20,"end":0.60,"zr":1.00,"mad":0.55},
    "Beans":          {"ini":0.40,"mid":1.15,"end":0.75,"zr":0.60,"mad":0.45},
    "Rice":           {"ini":1.05,"mid":1.30,"end":0.95,"zr":0.50,"mad":0.20},
    "Potatoes":       {"ini":0.50,"mid":1.15,"end":0.75,"zr":0.60,"mad":0.35},
    "Onions":         {"ini":0.70,"mid":1.05,"end":0.95,"zr":0.30,"mad":0.30},
    "Peppers":        {"ini":0.60,"mid":1.10,"end":0.80,"zr":0.50,"mad":0.30},
    "Cassava":        {"ini":0.40,"mid":0.85,"end":0.70,"zr":1.00,"mad":0.60},
    "Bananas":        {"ini":0.50,"mid":1.00,"end":0.80,"zr":0.90,"mad":0.35},
    "Wheat":          {"ini":0.70,"mid":1.15,"end":0.40,"zr":1.00,"mad":0.55},
    "Sorghum":        {"ini":0.30,"mid":1.00,"end":0.55,"zr":1.00,"mad":0.55},
    "Groundnuts":     {"ini":0.40,"mid":1.15,"end":0.75,"zr":0.50,"mad":0.50},
    "Sweet Potatoes": {"ini":0.50,"mid":1.15,"end":0.75,"zr":1.00,"mad":0.65},
    "Sunflower":      {"ini":0.35,"mid":1.10,"end":0.35,"zr":1.00,"mad":0.45},
    "Soybeans":       {"ini":0.40,"mid":1.15,"end":0.50,"zr":0.60,"mad":0.50},
}

STAGE_LABELS = {"ini":"🌱 Initial","mid":"🌿 Mid-Season","end":"🍂 End-Season"}

WMO_DESC = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    51:"Light drizzle",53:"Moderate drizzle",55:"Dense drizzle",
    61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
    80:"Slight showers",81:"Moderate showers",82:"Violent showers",
    95:"Thunderstorm",96:"Thunderstorm+hail",99:"Heavy thunderstorm+hail",
}

def wmo_icon(code):
    c = int(code or 0)
    if c == 0: return "☀️"
    if c in (1,2,3): return "🌤️"
    if 51 <= c <= 67: return "🌧️"
    if 80 <= c <= 82: return "🌦️"
    if 95 <= c <= 99: return "⛈️"
    return "🌥️"

st.markdown("""<style>
:root{--hb:#1a5fc8;--hg:#0b6b1b;--hr:#b81c1c;--bg:#f4f8f2;--sf:#fff;
  --bd:#dbe9db;--tx:#17301b;--gn:#0b6b1b;--gd:#075214;--gs:#e7f3e6;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:var(--bg)!important;color:var(--tx)!important;}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important;}
[data-testid="stMetric"]{background:var(--sf);border:1px solid var(--bd);
  border-radius:12px;padding:.5rem .7rem;}
[data-testid="stMetricLabel"] p{font-size:.76rem!important;margin:0!important;}
[data-testid="stMetricValue"] div{font-size:1.05rem!important;font-weight:700!important;}
div[data-baseweb="tab-list"]{gap:.3rem;background:transparent!important;}
button[data-baseweb="tab"]{background:var(--gs)!important;border:1px solid #b8d1b8!important;
  border-radius:999px!important;color:var(--gd)!important;padding:.35rem .75rem!important;font-size:.83rem!important;}
button[data-baseweb="tab"]>div{color:var(--gd)!important;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--gn)!important;border-color:var(--gn)!important;}
button[data-baseweb="tab"][aria-selected="true"]>div{color:#fff!important;}
[data-baseweb="select"]>div,div[data-baseweb="input"]>div,
.stNumberInput>div>div,.stTextInput>div>div{
  background:var(--sf)!important;color:var(--tx)!important;border-color:#c9d9c9!important;}
.stButton>button,.stDownloadButton>button{background:var(--gn)!important;color:#fff!important;
  border:1px solid var(--gn)!important;border-radius:10px!important;}
.stButton>button:hover{background:var(--gd)!important;}
section[data-testid="stSidebar"]{background:#eef5ec!important;}
button[title="Fork this app"],[data-testid="stToolbarActionButtonIcon"],
[data-testid="stBottomBlockContainer"],.stDeployButton,footer{display:none!important;}
.block-container{padding-top:.8rem!important;}
.hx-outer{border-radius:20px;overflow:hidden;margin:0 0 10px 0;
  background:linear-gradient(90deg,var(--hb) 0%,var(--hb) 33.3%,
  var(--hg) 33.3%,var(--hg) 66.6%,var(--hr) 66.6%,var(--hr) 100%);padding:9px 9px 7px 9px;}
.hx-panel{background:#fff;border:2px solid #d0ddd0;border-radius:14px;padding:9px 16px 7px 16px;}
.hx-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.hx-wm{font-family:Georgia,serif;font-size:2.4rem;font-weight:700;line-height:1;letter-spacing:-1px;flex-shrink:0;}
.hx-wm .H{color:#1a5fc8;}.hx-wm .y{color:#0b6b1b;}.hx-wm .P{color:#b81c1c;}
.hx-wm .I{color:#1a5fc8;}.hx-wm .S{color:#0b6b1b;}
.hx-wm .Ug{color:#b81c1c;font-size:1.4rem;vertical-align:middle;margin-left:4px;}
.hx-sub{font-family:Georgia,serif;font-size:.95rem;flex:1 1 160px;color:#444;}
.hx-auth{margin:4px 0 0 4px;font-family:Georgia,serif;font-size:.78rem;color:#ddd;}
.hx-auth strong{color:#fff;}
.geo-panel{background:#fff;border:1.5px solid #b8d4f8;border-radius:14px;
  padding:10px 16px;margin:6px 0 10px 0;font-size:.86rem;color:#14324d;}
.geo-panel b{color:#1a5fc8;}
.geo-coord{font-family:monospace;background:#eef4ff;padding:2px 6px;border-radius:6px;font-size:.82rem;}
.mad-panel{background:#f0f4ff;border:1px solid #7b9ed9;border-radius:10px;padding:9px 14px;font-size:.85rem;margin:4px 0;}
.soil-panel{background:#fef9ee;border:1px solid #e0c97a;border-radius:10px;padding:8px 14px;font-size:.85rem;margin:4px 0;}
.kc-stage{background:#e8f6ea;border:1px solid #a8d8a8;border-radius:10px;padding:6px 14px;font-size:.85rem;color:#073f12;margin:4px 0;font-weight:600;}
.nir-box{background:#fff3cd;border:1px solid #ffc107;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.iwr-box{background:#d4edda;border:1px solid #28a745;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;font-weight:600;}
.vol-box{background:#cfe2ff;border:1px solid #0d6efd;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.no-irr-box{background:#e8f5e9;border:1px solid #43a047;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.warn-raw{background:#fff8e1;border:1.5px solid #ffa000;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;font-weight:600;}
.warn-pwp{background:#ffebee;border:1.5px solid #c62828;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;font-weight:700;}
.warn-fc{background:#fff0f0;border:1px solid #d73027;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.refill-box{background:#e8eaf6;border:1px solid #5c6bc0;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.wb-summary{background:linear-gradient(135deg,#e8f5e9 0%,#e3f2fd 100%);border:1.5px solid #81c784;border-radius:14px;padding:12px 18px;margin:8px 0;font-size:.88rem;}
.past-hdr{background:linear-gradient(90deg,#0b6b1b 0%,#1a8a2e 100%);color:#fff;
  border-radius:10px 10px 0 0;padding:8px 16px;font-weight:700;font-size:.93rem;}
.today-hdr{background:linear-gradient(90deg,#1a5fc8 0%,#2a7fd4 100%);color:#fff;
  border-radius:10px 10px 0 0;padding:8px 16px;font-weight:700;font-size:.93rem;}
.live-dot{width:7px;height:7px;background:#22c55e;border-radius:50%;
  display:inline-block;margin-right:4px;animation:blink 1.4s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.ml-agree{background:#e8f5e9;border:1.5px solid #43a047;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.ml-warn{background:#fff8e1;border:1.5px solid #ffa000;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.ml-diff{background:#ffebee;border:1.5px solid #c62828;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.ml-panel{background:#f3f0ff;border:1px solid #7c4dff;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.ml-info{background:#e3f2fd;border:1.5px solid #1565c0;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;}
.ml-suppress{background:#f5f5f5;border:1px solid #9e9e9e;border-radius:10px;padding:8px 14px;font-size:.87rem;margin:4px 0;color:#555;}
.hybrid-box{background:linear-gradient(135deg,#e8f5e9 0%,#e3f2fd 100%);border:1.5px solid #1976d2;border-radius:10px;padding:9px 14px;font-size:.87rem;margin:4px 0;}
</style>""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = _time.time()
_el = _time.time() - st.session_state["last_refresh"]
if _el >= 3600:
    st.cache_data.clear()
    st.session_state["last_refresh"] = _time.time()
    st.rerun()
_rem = max(0, 3600 - int(_el))

st.markdown("""<div class="hx-outer"><div class="hx-panel"><div class="hx-row">
<span style="font-size:1.5rem;">&#127807;</span>
<span class="hx-wm">
  <span class="H">H</span><span class="y">y</span><span class="P">P</span>
  <span class="I">I</span><span class="S">S</span><span class="Ug"> Ug</span>
</span>
<span class="hx-sub">HydroPredict · IrrigSched</span>
</div></div>
<div class="hx-auth">by: Prosper <strong>BYARUHANGA</strong></div>
</div>""", unsafe_allow_html=True)

_now_str = datetime.now().strftime("%d %b %Y %H:%M")
st.caption(
    f'<span class="live-dot"></span> Live &middot; <b>{_now_str}</b>'
    f" &nbsp;·&nbsp; Auto-refresh in <b>{_rem//3600}h {(_rem%3600)//60}m</b>"
    f" &nbsp;·&nbsp; pandas {pd.__version__}",
    unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("📍 Location — Uganda")
loc_name = st.sidebar.selectbox("Select District / Site",
                                list(LOCATIONS.keys()), index=0, key="loc_sel")
_lcoords = LOCATIONS[loc_name]

if loc_name == "Custom Location":
    _clat  = st.sidebar.number_input("Latitude",  value=0.3380, format="%.4f", key="clat")
    _clon  = st.sidebar.number_input("Longitude", value=32.5680, format="%.4f", key="clon")
    _celev = st.sidebar.number_input("Elevation (m a.s.l.)", value=1189, key="celev")
    LAT, LON, ELEV = _clat, _clon, _celev
    SITE_NAME = f"Custom ({LAT:.4f}°, {LON:.4f}°)"
else:
    LAT, LON, ELEV = _lcoords
    SITE_NAME = loc_name

GMAPS_URL = f"https://maps.google.com/?q={LAT},{LON}"
GMAPS_SAT = f"https://maps.google.com/maps?q={LAT},{LON}&ll={LAT},{LON}&z=14&t=k"

_dsoil   = DISTRICT_SOIL.get(loc_name, DISTRICT_SOIL["Custom Location"])
SITE_FC  = _dsoil["fc"];  SITE_PWP = _dsoil["pwp"]
SITE_TXT = _dsoil["texture"]; SITE_SRC = _dsoil["source"]

st.sidebar.markdown(f"**📍 {SITE_NAME}**  \n`Lat {LAT}°` · `Lon {LON}°` · `{ELEV} m`  \n"
                    f"[🗺️ Maps]({GMAPS_URL}) | [🛰️ Sat]({GMAPS_SAT})")
st.sidebar.markdown("---\n### 🌍 Soil Type")
st.sidebar.info(f"**Auto-loaded:** {SITE_TXT}  \nFC: **{SITE_FC*100:.0f}%** · PWP: **{SITE_PWP*100:.0f}%**  \nSource: {SITE_SRC}")

soil_override = st.sidebar.checkbox("Override soil type", value=False, key="soil_ov")
if soil_override:
    soil_sel_s = st.sidebar.selectbox("Soil Type", list(SOIL_OPTS.keys()), key="soil_sel_s")
    so = SOIL_OPTS[soil_sel_s]
    ACTIVE_FC = so["fc"]; ACTIVE_PWP = so["pwp"]; ACTIVE_TXT = soil_sel_s
else:
    ACTIVE_FC = SITE_FC; ACTIVE_PWP = SITE_PWP; ACTIVE_TXT = SITE_TXT

if not ML_OK:
    st.sidebar.warning(
        f"{ML_STATUS}  \n"
        f"Place `{_MODEL_PATH.name}` next to `app.py`  \n"
        f"FAO-56 results remain fully functional without ML."
    )

st.sidebar.markdown("---\n### 📐 Field Area")
area_ha = st.sidebar.number_input(
    "Field Area (ha)", value=1.0, min_value=0.1, step=0.1, key="area_g",
    help="Used to compute total water volume required (m³ and litres)"
)

# ── GEO PANEL ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="geo-panel">
    📍 <b>Site:</b> {SITE_NAME} &nbsp;|&nbsp; Uganda<br>
    🌐 <b>Coordinates:</b>
      <span class="geo-coord">Lat {LAT}°</span>
      <span class="geo-coord">Lon {LON}°</span>
      <span class="geo-coord">Elev {ELEV} m a.s.l.</span>
    &nbsp;&nbsp;
    <a href="{GMAPS_URL}" target="_blank">🗺️ Google Maps</a>
    &nbsp;|&nbsp;
    <a href="{GMAPS_SAT}" target="_blank">🛰️ Satellite</a><br>
    🌍 <b>Soil ({SITE_SRC}):</b> {ACTIVE_TXT} &nbsp;·&nbsp;
      FC = <b>{ACTIVE_FC*100:.0f}%</b> &nbsp;·&nbsp; PWP = <b>{ACTIVE_PWP*100:.0f}%</b>
    </div>""", unsafe_allow_html=True)

# ── FAO-56 PENMAN-MONTEITH ────────────────────────────────────────────────────
def et0_pm(tmax, tmin, rh_max, rh_min, u2, rs, elev=None, doy=None, lat_deg=None):
    if elev is None: elev = ELEV
    if lat_deg is None: lat_deg = LAT
    try:
        tmax=float(tmax); tmin=float(tmin)
        rh_max=max(0.,min(100.,float(rh_max))); rh_min=max(0.,min(100.,float(rh_min)))
        u2=max(0.,float(u2)); rs=max(0.,float(rs))
        doy=int(doy) if doy else int(datetime.now().strftime("%j"))
    except Exception: return 0.0
    Gsc=0.0820; tm=(tmax+tmin)/2.
    P=101.3*((293.-0.0065*elev)/293.)**5.26; gamma=0.000665*P
    esmax=0.6108*np.exp(17.27*tmax/(tmax+237.3))
    esmin=0.6108*np.exp(17.27*tmin/(tmin+237.3)); es=(esmax+esmin)/2.
    ea=max(0.,min(es,(rh_max/100.*esmin+rh_min/100.*esmax)/2.))
    estm=0.6108*np.exp(17.27*tm/(tm+237.3))
    Delta=4098.*estm/(tm+237.3)**2.
    b=2.*np.pi*doy/365.; dr=1.+0.033*np.cos(b)
    phi=np.radians(abs(lat_deg)); ds=0.409*np.sin(b-1.39)
    ws=np.arccos(np.clip(-np.tan(phi)*np.tan(ds),-1.,1.))
    Ra=max(0.,(24.*60./np.pi)*Gsc*dr*(ws*np.sin(phi)*np.sin(ds)+np.cos(phi)*np.cos(ds)*np.sin(ws)))
    Rso=max(0.,(0.75+2e-5*elev)*Ra); Rns=0.77*rs
    fcd=max(0.,min(1.,1.35*(rs/max(Rso,.1))-.35))
    Rnl=max(0.,_SIGMA*((tmax+273.16)**4+(tmin+273.16)**4)/2.*(0.34-0.14*np.sqrt(max(0.,ea)))*fcd)
    Rn=max(0.,Rns-Rnl)
    num=0.408*Delta*Rn+gamma*(900./(tm+273.))*u2*(es-ea)
    den=Delta+gamma*(1.+0.34*u2)
    return max(0.,round(num/den,3)) if den>0 else 0.

def et0_hargreaves(tmax, tmin, doy=None, lat_deg=None):
    if lat_deg is None: lat_deg = LAT
    doy=doy or int(datetime.now().strftime("%j"))
    b=2.*np.pi*doy/365.; dr=1.+0.033*np.cos(b); phi=np.radians(abs(lat_deg))
    ds=0.409*np.sin(b-1.39); ws=np.arccos(np.clip(-np.tan(phi)*np.tan(ds),-1.,1.))
    Ra=max(0.,(24.*60./np.pi)*0.0820*dr*(ws*np.sin(phi)*np.sin(ds)+np.cos(phi)*np.cos(ds)*np.sin(ws)))
    tm=(tmax+tmin)/2.; td=max(0.,tmax-tmin)
    return round(max(0.,0.0023*Ra*(tm+17.8)*td**0.5),3)

# ── SOIL WATER HELPERS ────────────────────────────────────────────────────────
def compute_taw(fc, pwp, zr):
    return (fc - pwp) * zr * 1000.

def compute_raw(taw, mad):
    return mad * taw

def eff_rain(p):
    p = float(p or 0)
    if p <= 0:    return 0.
    if p <= 25.4: return p*(125.-0.6*p)/125.
    return p - 12.7 - 0.1*p

def kc_from_stage(stage, crop):
    return crop_params[crop][stage]

def compute_volume(iwr_mm, area_ha):
    """Convert depth (mm) × area (ha) → volume in m³ and litres."""
    v = float(iwr_mm) * float(area_ha) * 10.   # 1 mm × 1 ha = 10 m³
    return {"vol_m3": round(v, 1), "vol_L": round(v * 1000., 0)}


# ── DEPLETION STATUS — canonical FAO-56 ───────────────────────────────────────
def depletion_status(dr, raw, taw):
    """Canonical FAO-56 trigger: Dr > RAW → irrigate. IWR (mm) = Dr."""
    if dr <= 0:
        return "🟢 Field Capacity", False, "Soil at FC — no irrigation needed"
    if dr <= raw * 0.5:
        return "✅ Adequate moisture", False, \
               f"Dr={dr:.1f} mm — well within safe range, no irrigation"
    if dr <= raw:
        return ("🟡 Monitor — nearing MAD", False,
                f"Dr={dr:.1f} mm approaching RAW={raw:.1f} mm — "
                f"apply {round(dr,1)} mm now to refill, or wait if rain is forecast")
    if dr <= taw * 0.85:
        return ("⚠️ Below MAD — irrigate", True,
                f"Dr={dr:.1f} mm > RAW={raw:.1f} mm — crop stress beginning; irrigate now")
    return ("🔴 Near wilting point — URGENT", True,
            f"Dr={dr:.1f} mm ≈ TAW={taw:.1f} mm — immediate irrigation!")

# ── CORE DAILY WATER BALANCE (FAO-56 Dr approach) ────────────────────────────
def run_water_balance(daily_df, crop, soil, planting_ts, sm_pct,
                      stage_override=None, mad_eff=None):
    cp   = crop_params[crop]
    zr   = cp["zr"]
    mad  = mad_eff if mad_eff is not None else cp["mad"]
    taw  = compute_taw(soil["fc"], soil["pwp"], zr)
    raw  = compute_raw(taw, mad)

    theta = soil["pwp"] + (sm_pct/100.) * (soil["fc"] - soil["pwp"])
    theta = min(theta, soil["fc"])
    dr    = max(0., (soil["fc"] - theta) * zr * 1000.)

    df = daily_df.copy()

    if "rh_max" not in df.columns:
        rh_col = "rh_mean" if "rh_mean" in df.columns else ("rh" if "rh" in df.columns else None)
        if rh_col:
            df["rh_max"] = (df[rh_col] + 10).clip(upper=100)
            df["rh_min"] = (df[rh_col] - 10).clip(lower=0)
        else:
            df["rh_max"] = 70.; df["rh_min"] = 50.

    if stage_override:
        df["kc"] = crop_params[crop][stage_override]
    else:
        df["kc"] = df.index.map(
            lambda d: crop_params[crop]["ini"]
            if (d - planting_ts).days < 30
            else (crop_params[crop]["mid"]
                  if (d - planting_ts).days < 90
                  else crop_params[crop]["end"])
        )

    df["ET0"] = df.apply(lambda r: et0_pm(
        r["tmax"], r["tmin"], r["rh_max"], r["rh_min"],
        r["wind"], r["rs"],
        doy=r.name.timetuple().tm_yday, lat_deg=LAT, elev=ELEV,
    ), axis=1)
    df["ETc"] = (df["kc"] * df["ET0"]).round(3)

    prec_col = "precipitation" if "precipitation" in df.columns else (
               "precip" if "precip" in df.columns else None)
    if prec_col:
        df["Pe"] = df[prec_col].fillna(0.).apply(eff_rain)
    else:
        df["Pe"] = 0.
        prec_col = "precipitation"
        df[prec_col] = 0.

    dr_vals=[]; nir_vals=[]; iwr_vals=[]; status_vals=[]; sm_vals=[]; note_vals=[]
    ml_nir_vals = []

    for _, row in df.iterrows():
        pe_r    = row["Pe"];  etc_r = row["ETc"]
        dr_new  = max(0., min(taw, dr - pe_r + etc_r))
        nir_day = round(max(0., etc_r - pe_r), 3)

        # ML cross-check — dr_ratio-scaled so ML ≈ 0 when soil is at FC
        _tmean    = (row["tmax"] + row["tmin"]) / 2.
        _rh       = row.get("rh_mean", 60.)
        _wind     = row.get("wind", 1.5)
        _prec     = row.get("precipitation", row.get("precip", 0.))
        _dr_ratio = dr_new / taw if taw > 0 else 0.
        _ml_raw   = ml_predict_iwr(_tmean, _rh, _wind, row["kc"],
                                   _prec, soil["fc"], soil["pwp"], zr,
                                   taw=taw, dr_ratio=_dr_ratio)
        # Safety-constrain: suppress ML when soil is moist or rain occurred
        _ml = _constrain_ml(float(_ml_raw), dr_new, taw, _prec) \
              if _ml_raw is not None else None
        ml_nir_vals.append(round(_ml, 3) if _ml is not None else None)

        # FAO-56 authoritative trigger
        lbl, irrigate, note = depletion_status(dr_new, raw, taw)
        if irrigate:
            iwr_mm = round(dr_new, 3)   # IWR = Dr (refill depth in mm)
            dr = 0.
        else:
            iwr_mm = 0.
            dr = dr_new

        sm_now = max(0., min(100., round((1. - dr/taw)*100, 1))) if taw > 0 else 70

        dr_vals.append(round(dr, 2))
        nir_vals.append(nir_day)
        iwr_vals.append(iwr_mm)
        status_vals.append(lbl)
        sm_vals.append(sm_now)
        note_vals.append(note)

    df["Dr_mm"]  = dr_vals
    df["SM_pct"] = sm_vals
    df["NIR"]    = nir_vals
    df["IWR"]    = iwr_vals
    df["Status"] = status_vals
    df["Note"]   = note_vals
    df["ML_NIR"] = ml_nir_vals
    return df, taw, raw

# ── SOIL MOISTURE ESTIMATION — ERA5 10-day back-run ──────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def estimate_sm(fc, pwp, zr, lat=None, lon=None, elev=None, default_kc=1.0):
    """
    FIX v7.6: Initial SM changed from 70% → 50% of TAW.

    WHY: In Uganda's bi-modal rainy climate, starting at 70% SM (Dr=30%TAW)
    + 10 days of ERA5 rain kept Dr well below RAW for most crops.  The slider
    defaulted to 85-95% SM, meaning Dr_start was tiny and the 5-day window
    was never long enough to deplete past RAW → irrigation NEVER triggered.

    At 50% SM start (Dr=50%TAW), the ERA5 back-run correctly reflects dry
    spells: frequent light rain still keeps soil moist (correct), while 2+
    consecutive dry days raise Dr above RAW and trigger irrigation (also
    correct).  This matches the FAO-56 recommendation to initialise at 50%
    of TAW when actual soil moisture is unknown (Allen et al. 1998, §8.3.2).
    """
    lat = lat or LAT; lon = lon or LON; elev = elev or ELEV
    try:
        end_  = date.today() - timedelta(days=1)
        start_= end_ - timedelta(days=10)
        r = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date={start_}&end_date={end_}"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,"
            f"shortwave_radiation_sum,wind_speed_10m_max,"
            f"relative_humidity_2m_max,relative_humidity_2m_min&timezone={TIMEZONE}",
            timeout=12).json()
        d = r.get("daily",{}); dates = d.get("time",[])
        taw_ = (fc-pwp)*zr*1000.
        # FIX v7.6: start at 50% SM (Dr=50%TAW) instead of 70% SM (Dr=30%TAW)
        # This is the FAO-56 §8.3.2 default when actual SM is unknown.
        theta = pwp + 0.50*(fc-pwp)
        dr_   = max(0., (fc-theta)*zr*1000.)
        for i in range(len(dates)):
            tx=d["temperature_2m_max"][i]; tn=d["temperature_2m_min"][i]
            if tx is None or tn is None: continue
            rh_mx=d["relative_humidity_2m_max"][i] or 70
            rh_mn=d["relative_humidity_2m_min"][i] or 50
            wk=(d["wind_speed_10m_max"][i] or 7.2)/3.6*_W2M
            rs_i=d["shortwave_radiation_sum"][i] or 18.
            prec=d["precipitation_sum"][i] or 0.
            doy_i=datetime.strptime(dates[i],"%Y-%m-%d").timetuple().tm_yday
            et0_i=et0_pm(tx,tn,rh_mx,rh_mn,wk,rs_i,elev=elev,doy=doy_i,lat_deg=lat)
            etc_i = et0_i * default_kc
            pe_i  = eff_rain(prec)
            dr_   = max(0., min(taw_, dr_ - pe_i + etc_i))
        sm_est = int(max(0, min(100, (1-dr_/taw_)*100))) if taw_>0 else 60
        return sm_est
    except Exception:
        return 55  # conservative default: just above typical RAW

# ── EXCEL EXPORT HELPER ───────────────────────────────────────────────────────
def df_to_excel_bytes(df_dict: dict):
    if not OPENPYXL_OK:
        return None
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet[:31])
        return buf.getvalue()
    except Exception:
        return None

def _show_download_buttons(dl_csv, dl_xlsx, fn_base, csv_key, xlsx_key):
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download CSV", dl_csv,
            f"{fn_base}.csv", "text/csv", key=csv_key)
    with dl_col2:
        if dl_xlsx is not None:
            st.download_button(
                "⬇️ Download Excel (.xlsx)", dl_xlsx,
                f"{fn_base}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=xlsx_key)
        else:
            st.info("Excel export unavailable on this Python version — CSV available above.")

# ── SM GAUGE CHART ────────────────────────────────────────────────────────────
def sm_gauge(sm_pct, raw_pct_of_taw, title="Soil Moisture"):
    color = "#22c55e" if sm_pct > 60 else ("#f59e0b" if sm_pct > 35 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sm_pct,
        delta={"reference": 100, "suffix":"%"},
        title={"text": title, "font":{"size":14}},
        gauge={
            "axis": {"range":[0,100], "ticksuffix":"%"},
            "bar":  {"color": color, "thickness":0.28},
            "steps":[
                {"range":[0, raw_pct_of_taw*50], "color":"#fee2e2"},
                {"range":[raw_pct_of_taw*50, raw_pct_of_taw*100], "color":"#fef3c7"},
                {"range":[raw_pct_of_taw*100, 100], "color":"#dcfce7"},
            ],
            "threshold":{"line":{"color":"#b91c1c","width":3},
                         "thickness":0.75,"value": raw_pct_of_taw*100},
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ── WEATHER APIs ──────────────────────────────────────────────────────────────
_ch = f"{datetime.now().strftime('%Y%m%d%H')}_{LAT}_{LON}"

@st.cache_data(ttl=3600, show_spinner=False)
def get_current_weather(_k, lat, lon, elev):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,shortwave_radiation,weather_code"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max,shortwave_radiation_sum,precipitation_sum,weather_code"
            f"&forecast_days=1&timezone={TIMEZONE}", timeout=12).json()
        cur=r.get("current",{}); d=r.get("daily",{})
        tmax=d.get("temperature_2m_max",[None])[0]; tmin=d.get("temperature_2m_min",[None])[0]
        rh_mx=d.get("relative_humidity_2m_max",[70])[0] or 70
        rh_mn=d.get("relative_humidity_2m_min",[50])[0] or 50
        wk=(d.get("windspeed_10m_max",[7.2])[0] or 7.2)/3.6*_W2M
        rs=d.get("shortwave_radiation_sum",[18.])[0] or 18.
        prec=d.get("precipitation_sum",[0.])[0] or 0.
        wcode=d.get("weather_code",[0])[0] or 0
        tc=cur.get("temperature_2m",25)
        tmax=tmax or tc+4; tmin=tmin or tc-4
        return {"tmax":round(tmax,1),"tmin":round(tmin,1),"rh_max":rh_mx,"rh_min":rh_mn,
                "rh_mean":round((rh_mx+rh_mn)/2,1),"wind":round(wk,3),"rs":round(rs,1),
                "precip":round(prec,1),"wcode":wcode,
                "description":WMO_DESC.get(int(wcode),f"Code {wcode}"),
                "source":"Open-Meteo ICON+GFS (live)"}
    except requests.Timeout:
        return None   # handled silently — UI falls back to defaults with a warning
    except Exception:
        return None   # network errors handled by caller; defaults applied below

@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(_k, lat, lon, elev):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max,shortwave_radiation_sum,precipitation_sum,weather_code"
            f"&forecast_days=7&timezone={TIMEZONE}", timeout=12).json()
        d=r.get("daily",{})
        if not d: return None
        rows=[]
        for i in range(len(d["time"])):
            wk=(d["windspeed_10m_max"][i] or 7.2)/3.6*_W2M
            rh_mx=d["relative_humidity_2m_max"][i] or 70
            rh_mn=d["relative_humidity_2m_min"][i] or 50
            rows.append({"date":pd.to_datetime(d["time"][i]),
                "tmax":d["temperature_2m_max"][i] or 28,"tmin":d["temperature_2m_min"][i] or 16,
                "rh_max":rh_mx,"rh_min":rh_mn,"rh_mean":round((rh_mx+rh_mn)/2,1),
                "wind":round(wk,3),"rs":d["shortwave_radiation_sum"][i] or 18.,
                "precipitation":d["precipitation_sum"][i] or 0.,
                "weather_code":d["weather_code"][i] or 0})
        df=pd.DataFrame(rows).set_index("date")
        today_tz = pd.Timestamp.now(tz=TIMEZONE).normalize().tz_localize(None)
        return df[df.index >= today_tz].head(5)
    except requests.Timeout:
        st.warning("⚠️ Open-Meteo API timed out — check your connection and try again.")
        return None
    except Exception as e:
        st.error(f"⚠️ Forecast fetch failed: {type(e).__name__}: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_weather(start_date, end_date, lat, lon):
    try:
        r = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"wind_speed_10m_max,shortwave_radiation_sum,precipitation_sum"
            f"&timezone={TIMEZONE}", timeout=25).json()
        d=r.get("daily",{})
        if not d: return None
        rh_mx=d.get("relative_humidity_2m_max",[]); rh_mn=d.get("relative_humidity_2m_min",[])
        df=pd.DataFrame({
            "date":pd.to_datetime(d["time"]),
            "tmax":[x or 28 for x in d["temperature_2m_max"]],
            "tmin":[x or 16 for x in d["temperature_2m_min"]],
            "rh_max":[(a or 70) for a in rh_mx],"rh_min":[(a or 50) for a in rh_mn],
            "rh_mean":[((a or 70)+(b or 50))/2 for a,b in zip(rh_mx,rh_mn)],
            "wind":[(x or 7.2)/3.6*_W2M for x in d["wind_speed_10m_max"]],
            "rs":[x or 18. for x in d["shortwave_radiation_sum"]],
            "precipitation":[x or 0. for x in d["precipitation_sum"]],
        }).set_index("date")
        df["rh"]=df["rh_mean"]
        return df.dropna(subset=["tmax","tmin"])
    except requests.Timeout:
        st.warning("⚠️ ERA5 archive API timed out — historical data unavailable right now.")
        return None
    except Exception as e:
        st.error(f"⚠️ Historical weather fetch failed: {type(e).__name__}: {e}")
        return None

# ── FETCH TODAY'S LIVE WEATHER ────────────────────────────────────────────────
with st.spinner(f"📡 Fetching live weather for {SITE_NAME}…"):
    wx = get_current_weather(_ch, LAT, LON, ELEV)

if wx:
    lt=wx["tmax"]; ln=wx["tmin"]; lr_max=wx["rh_max"]; lr_min=wx["rh_min"]
    lr_mean=wx["rh_mean"]; lw=wx["wind"]; ls=wx["rs"]; lp=wx["precip"]
else:
    lt,ln,lr_max,lr_min,lr_mean,lw,ls,lp = 28.,16.,70.,50.,60.,1.5,18.,0.
_doy = int(datetime.today().strftime("%j"))

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Today's IWR", "☁️ 5-Day Forecast", "📅 Historical"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TODAY'S IWR
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header(f"📊 Today's IWR — {SITE_NAME}")
    st.caption(f"📡 {wx['source'] if wx else 'Weather unavailable'}")

    if wx:
        st.success(
            f"✅ **{wx['description']}** · 📡 Forecast Rain: **{lp} mm** · "
            f"Lat {LAT}° / Lon {LON}° / {ELEV} m")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🌡️ Tmax / Tmin",  f"{lt}°C / {ln}°C")
        c2.metric("💧 RH min–max",   f"{lr_min:.0f}–{lr_max:.0f}%", f"Mean {lr_mean:.0f}%")
        c3.metric("🌬️ Wind (2 m)",   f"{lw:.2f} m/s")
        c4.metric("☀️ Solar Rad",    f"{ls:.1f} MJ/m²/d")
        c5.metric("🌧️ Forecast Rain", f"{lp:.1f} mm", "📡 predicted")
    else:
        st.warning("⚠️ Weather unavailable — using default values")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Crop & Growth Stage")
        cr1    = st.selectbox("Crop", list(crop_params.keys()), key="cr1")
        cp1    = crop_params[cr1]
        stage1 = st.radio("Growing Stage", list(STAGE_LABELS.keys()),
                          format_func=lambda x: STAGE_LABELS[x],
                          key="stg1", horizontal=True)
        kc1    = kc_from_stage(stage1, cr1)

        mad_crop  = cp1["mad"]
        mad_adj   = adjust_mad_for_soil(mad_crop, ACTIVE_TXT)
        mad_delta = mad_adj - mad_crop

        st.markdown(
            f'<div class="kc-stage">Kc ({STAGE_LABELS[stage1]}) = <b>{kc1:.3f}</b> '
            f'· Zr = {cp1["zr"]:.2f} m</div>', unsafe_allow_html=True)

        _taw_p = compute_taw(ACTIVE_FC, ACTIVE_PWP, cp1["zr"])
        _raw_p = compute_raw(_taw_p, mad_adj)
        st.markdown(
            f'<div class="mad-panel">'
            f'📐 <b>FAO-56 Irrigation Thresholds</b><br>'
            f'TAW = <b>{_taw_p:.1f} mm</b> &nbsp;·&nbsp; '
            f'Crop MAD = <b>{mad_crop:.2f}</b>'
            f'{"" if mad_delta==0 else f" → soil-adjusted to <b>{mad_adj:.2f}</b>"}<br>'
            f'RAW (trigger) = <b>{_raw_p:.1f} mm</b> &nbsp;·&nbsp; '
            f'Irrigate when Dr &gt; <b>{_raw_p:.1f} mm</b><br>'
            f'<small>🟢 Dr≤{_raw_p*0.5:.1f} OK · 🟡 Dr≤{_raw_p:.1f} Monitor · '
            f'⚠️ Dr&gt;{_raw_p:.1f} Irrigate · 🔴 Dr&gt;{_taw_p*0.85:.1f} Urgent</small></div>',
            unsafe_allow_html=True)

        st.markdown(
            f"| Stage | Ini Kc | Mid Kc | End Kc | MAD |\n|---|---|---|---|---|\n"
            f"| {cr1} | {cp1['ini']} | {cp1['mid']} | {cp1['end']} | {mad_crop:.2f} → **{mad_adj:.2f}** |")

    with col2:
        st.subheader("🌦️ Weather Input (24-hour period)")
        tmax_in = st.number_input("Tmax (°C)",            value=float(lt),      key="t1")
        tmin_in = st.number_input("Tmin (°C)",            value=float(ln),      key="t2")
        rh_in   = st.number_input("RH mean (%)",          value=float(lr_mean), min_value=0., max_value=100., key="rh1")
        wind_in = st.number_input("Wind 2m (m/s)",        value=float(lw),      min_value=0., key="w1")
        rs_in   = st.number_input("Solar Rad (MJ/m²/d)",  value=float(ls),      min_value=0., key="rs1")

        prec_in = lp
        st.info(f"🌧️ **Rainfall (Open-Meteo forecast): {prec_in:.1f} mm**")

        soil_obj = {"fc": ACTIVE_FC, "pwp": ACTIVE_PWP}
        _sm_def  = estimate_sm(ACTIVE_FC, ACTIVE_PWP, cp1["zr"], LAT, LON, ELEV)
        # v7.7: quick-set buttons for season simulation
        # FIX v7.7b: must set st.session_state["sm1"] (the slider key) directly —
        # setting a separate "sm1_val" key and passing it as value= is ignored by
        # Streamlit once the slider key already exists in session_state.
        st.markdown("**🌡️ Season scenario:**")
        _t1, _t2, _t3 = st.columns(3)
        if _t1.button("☀️ Dry (30%)",  key="sm1_dry", use_container_width=True):
            st.session_state["sm1"] = 30
        if _t2.button("🌤️ Mod (55%)", key="sm1_mod", use_container_width=True):
            st.session_state["sm1"] = 55
        if _t3.button("🌧️ Wet (80%)", key="sm1_wet", use_container_width=True):
            st.session_state["sm1"] = 80
        sm_pct = st.slider("Current Soil Moisture (% of FC)", 0, 100, _sm_def, key="sm1")

        if sm_pct >= 95:
            st.markdown(
                '<div class="warn-fc">⚠️ <b>Soil near Field Capacity</b> — '
                'do NOT irrigate; waterlogging risk.</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="soil-panel">🌍 <b>{ACTIVE_TXT}</b> · '
            f'FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%<br>'
            f'</div>', unsafe_allow_html=True)

    if st.button("🧮 Calculate Today's IWR + Past 5 Days Water History",
                 type="primary", use_container_width=True, key="calc1"):

        mad_eff1 = adjust_mad_for_soil(cp1["mad"], ACTIVE_TXT)
        taw1     = compute_taw(ACTIVE_FC, ACTIVE_PWP, cp1["zr"])
        raw1     = compute_raw(taw1, mad_eff1)

        rh_mx1  = min(100., rh_in+10.); rh_mn1 = max(0., rh_in-10.)
        et0_fao = et0_pm(tmax_in, tmin_in, rh_mx1, rh_mn1, wind_in, rs_in, doy=_doy)
        et0_h   = et0_hargreaves(tmax_in, tmin_in, doy=_doy)
        etc1    = round(kc1 * et0_fao, 3)
        pe1     = eff_rain(prec_in)
        nir1_day = round(max(0., etc1 - pe1), 3)

        theta1   = ACTIVE_PWP + (sm_pct/100.)*(ACTIVE_FC-ACTIVE_PWP)
        theta1   = min(theta1, ACTIVE_FC)
        dr_start = max(0., (ACTIVE_FC - theta1)*cp1["zr"]*1000.)
        dr_today = max(0., min(taw1, dr_start - pe1 + etc1))

        status_lbl, irrigate, note_today = depletion_status(dr_today, raw1, taw1)

        if irrigate:
            nir1     = round(dr_today, 3)
            iwr1     = nir1
            dr_after = 0.
        else:
            nir1     = 0.
            iwr1     = 0.
            dr_after = dr_today

        sm_now = max(0., min(100., round((1-dr_today/taw1)*100, 1))) if taw1>0 else 70
        sm_aft = max(0., min(100., round((1-dr_after/taw1)*100, 1))) if taw1>0 else 100

        st.markdown(
            f'<div class="wb-summary">'
            f'<b>🌊 Water Balance Summary</b> — {cr1} · {ACTIVE_TXT}<br>'
            f'TAW = <b>{taw1:.1f} mm</b> &nbsp;|&nbsp; '
            f'RAW threshold = <b>{raw1:.1f} mm</b> (MAD={mad_eff1:.2f}) &nbsp;|&nbsp; '
            f'ET₀ PM = <b>{et0_fao:.2f} mm/d</b> &nbsp;|&nbsp; '
            f'ETc = <b>{etc1:.2f} mm/d</b><br>'
            f'Pe = <b>{pe1:.2f} mm</b> &nbsp;|&nbsp; '
            f'NIR = <b>{nir1_day:.2f} mm/d</b> &nbsp;|&nbsp; '
            f'Dr = <b>{dr_today:.2f} mm</b> &nbsp;|&nbsp; '
            f'SM = <b>{sm_now:.1f}% of FC</b><br>'
            f'<b>Decision: {status_lbl}</b></div>',
            unsafe_allow_html=True)

        gcol1, gcol2 = st.columns([1,2])
        with gcol1:
            st.plotly_chart(
                sm_gauge(sm_now, 1-(raw1/taw1) if taw1>0 else 0.5,
                         f"SM% of FC — {cr1}"),
                use_container_width=True)
        with gcol2:
            if "🔴" in status_lbl:
                st.markdown(
                    f'<div class="warn-pwp">🔴 <b>SOIL AT/NEAR WILTING POINT</b> — '
                    f'Dr = {dr_today:.1f} mm ≈ TAW = {taw1:.1f} mm — '
                    f'Immediate irrigation required!</div>', unsafe_allow_html=True)
            elif "⚠️" in status_lbl:
                st.markdown(
                    f'<div class="warn-raw">⚠️ <b>Soil moisture below MAD threshold</b> — '
                    f'Dr = {dr_today:.1f} mm > RAW = {raw1:.1f} mm — '
                    f'Schedule irrigation today.</div>', unsafe_allow_html=True)
            elif "🟡" in status_lbl:
                st.markdown(
                    f'<div class="nir-box">🟡 <b>Monitor — approaching MAD threshold</b><br>'
                    f'Dr = {dr_today:.1f} mm · RAW = {raw1:.1f} mm · not yet depleted past threshold<br>'
                    f'Pre-emptive: apply <b>{round(dr_today,2):.2f} mm</b> now to refill, '
                    f'or wait if rain is forecast</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="no-irr-box">{status_lbl}<br>{note_today}</div>',
                    unsafe_allow_html=True)

            st.markdown(
                f'<div class="nir-box">📐 <b>Daily NIR = {nir1_day:.2f} mm</b> '
                f'= ETc ({etc1:.2f}) − Pe ({pe1:.2f}) &nbsp;·&nbsp; '
                f'Dr = <b>{dr_today:.2f} mm</b> &nbsp;·&nbsp; '
                f'TAW = {taw1:.1f} mm · RAW = {raw1:.1f} mm</div>',
                unsafe_allow_html=True)

        # ── PAST 5 DAYS CONTEXT ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="past-hdr">📅 PAST 5 DAYS — Soil & Crop Water History</div>',
                    unsafe_allow_html=True)
        st.caption("Showing how soil moisture and crop water need evolved BEFORE today.")

        past_end_d   = date.today() - timedelta(days=1)
        past_start_d = past_end_d - timedelta(days=4)
        with st.spinner("📡 Fetching ERA5 past 5 days…"):
            hist_ctx = get_historical_weather(str(past_start_d), str(past_end_d), LAT, LON)

        past_r = None
        if hist_ctx is not None and not hist_ctx.empty:
            past_r, _, _ = run_water_balance(
                hist_ctx, cr1, soil_obj,
                pd.Timestamp(date.today()-timedelta(days=45)),
                sm_pct, stage_override=stage1, mad_eff=mad_eff1)

            def _style_row(val):
                v = str(val)
                if "⚠️" in v or "🔴" in v: return "background:#ffe0e0;font-weight:bold"
                if "🟢" in v or "✅" in v:  return "background:#e8f5e9"
                if "🟡" in v:               return "background:#fff9c4"
                return ""

            rows_past = []
            for dt, row in past_r.iterrows():
                dr_val  = float(row["Dr_mm"])
                irw_val = float(row["IWR"])
                vol     = compute_volume(irw_val, area_ha)
                rows_past.append({
                    "Date":     dt.strftime("%Y-%m-%d (%a)"),
                    "Rain mm":  round(float(row.get("precipitation",0.)),1),
                    "ETc mm/d": round(float(row["ETc"]),2),
                    "Pe mm":    round(float(row["Pe"]),2),
                    "Dr mm":    round(dr_val,2),
                    "SM %FC":   round(float(row["SM_pct"]),1),
                    "NIR mm":   round(float(row["NIR"]),2),
                    "IWR mm":   round(irw_val,2),
                    "Vol m³":   vol["vol_m3"],
                    "Vol L":    int(vol["vol_L"]),
                    "Status":   row["Status"],
                    "Note":     row["Note"],
                })
            past_df = pd.DataFrame(rows_past).set_index("Date")

            _num_fmt = {
                "Rain mm":  "{:.1f}", "ETc mm/d": "{:.2f}", "Pe mm":  "{:.2f}",
                "Dr mm":    "{:.2f}", "SM %FC":   "{:.1f}", "NIR mm": "{:.2f}",
                "IWR mm":   "{:.2f}", "Vol m³":   "{:.1f}", "Vol L":  "{:.0f}",
            }
            styled = _styler_map(past_df.style, _style_row, subset=["Status"])
            styled = styled.format(_num_fmt, na_rep="—")

            st.markdown(
                f'<div class="mad-panel">'
                f'📐 <b>Reference thresholds — {cr1} / {ACTIVE_TXT}</b><br>'
                f'TAW = <b>{taw1:.1f} mm</b> · RAW = <b>{raw1:.1f} mm</b> · '
                f'MAD = <b>{mad_eff1:.2f}</b> (soil-adjusted from {cp1["mad"]:.2f})<br>'
                f'<small>Irrigate when Dr &gt; RAW. NIR = ETc – Pe (daily crop deficit).</small></div>',
                unsafe_allow_html=True)

            st.dataframe(styled, use_container_width=True)

            fig_ctx = go.Figure()
            fig_ctx.add_scatter(
                x=past_r.index.strftime("%a %d"),
                y=past_r["Dr_mm"].astype(float),
                mode="lines+markers+text", name="Dr Deficit (mm)",
                text=past_r["Dr_mm"].round(1).astype(str)+" mm",
                textposition="top center",
                line=dict(color="#e6550d",width=2.5), marker=dict(size=9))
            fig_ctx.add_bar(
                x=past_r.index.strftime("%a %d"),
                y=past_r["precipitation"].astype(float),
                name="Rain mm", marker_color="#1a5fc8", opacity=0.4, yaxis="y2")
            fig_ctx.add_bar(
                x=past_r.index.strftime("%a %d"),
                y=past_r["NIR"].astype(float),
                name="NIR mm/d", marker_color="#0b6b1b", opacity=0.5, yaxis="y2")
            fig_ctx.add_hline(y=raw1, line_dash="dash", line_color="#756bb1",
                              annotation_text=f"RAW={raw1:.1f} mm — irrigate above this")
            fig_ctx.add_hline(y=taw1, line_dash="dot",  line_color="#d73027",
                              annotation_text=f"TAW={taw1:.1f} mm — wilting risk")
            fig_ctx.update_layout(
                title="Past 5 Days: Root-Zone Depletion (Dr) vs Thresholds",
                yaxis=dict(title="Depletion Dr (mm)"),
                yaxis2=dict(title="mm/d",overlaying="y",side="right"),
                barmode="group", legend=dict(x=0,y=1.12,orientation="h"),
                height=320, plot_bgcolor="#f4f8f2", paper_bgcolor="#f4f8f2")
            st.plotly_chart(fig_ctx, use_container_width=True)
        else:
            st.info("ℹ️ ERA5 past data unavailable — archive may lag 3–5 days.")

        # ── TODAY'S RESULT ────────────────────────────────────────────────────
        st.markdown('<div class="today-hdr">💧 TODAY\'S IRRIGATION DECISION</div>',
                    unsafe_allow_html=True)

        st.info(
            f"📍 **{SITE_NAME}** · Lat {LAT}° · Lon {LON}° · {ELEV} m  \n"
            f"🌱 **{cr1}** · Stage: **{STAGE_LABELS[stage1]}** · Kc = **{kc1:.3f}**  \n"
            f"🌍 Soil: **{ACTIVE_TXT}** · FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%  \n"
            f"📐 TAW = **{taw1:.1f} mm** · RAW (trigger) = **{raw1:.1f} mm** "
            f"(crop MAD {cp1['mad']:.2f} → soil-adj. **{mad_eff1:.2f}**)"
        )

        # ── ML cross-check for today (dr_ratio-scaled + safety-constrained)
        _dr_ratio_today = dr_today / taw1 if taw1 > 0 else 0.
        _ml_today_scaled = ml_predict_iwr(
            (tmax_in+tmin_in)/2., rh_in, wind_in, kc1,
            prec_in, ACTIVE_FC, ACTIVE_PWP, cp1["zr"],
            taw=taw1, dr_ratio=_dr_ratio_today)
        # Apply safety constraints (same filters as used in run_water_balance)
        _ml_today = None
        if _ml_today_scaled is not None:
            _ml_today = _constrain_ml(
                _ml_today_scaled, dr_today, taw1, prec_in
            )

        r1,r2,r3,r4,r5,r6,r7,r8 = st.columns(8)
        r1.metric("ET₀ PM mm/d",   f"{et0_fao:.2f}")
        r2.metric("ET₀ H mm/d",    f"{et0_h:.2f}")
        r3.metric("ETc mm/d",      f"{etc1:.2f}", f"Kc={kc1:.3f}")
        r4.metric("Pe mm",         f"{pe1:.2f}")
        r5.metric("NIR mm/d",      f"{nir1_day:.2f}", "ETc − Pe")
        r6.metric("Dr mm",         f"{dr_today:.2f}", f"RAW={raw1:.1f}")
        r7.metric("SM % FC",       f"{sm_now:.1f}%", f"→{sm_aft:.1f}% after")
        if _ml_today is not None:
            r8.metric("🤖 ML NIR mm/d", f"{_ml_today:.2f}", f"FAO={nir1_day:.2f}")
        else:
            r8.metric("🤖 ML NIR mm/d", "N/A", "model not loaded")

        st.markdown("---")

        # ── ML vs FAO-56 agreement panel ─────────────────────────────────────
        _ml_css, _ml_icon, _ml_lbl, _ml_exp, _ml_pct = ml_agreement(
            nir1_day, _ml_today,
            dr_mm=dr_today, raw_mm=raw1, rainfall_mm=prec_in
        )
        if _ml_today is not None:
            _ml_txt = (
                f"<b>\U0001f916 XGBoost ML Cross-Check</b>"
                f"&nbsp;\u00b7&nbsp;<b>ML NIR = {_ml_today:.2f} mm/d</b>"
                f"&nbsp;|&nbsp;<b>FAO-56 NIR = {nir1_day:.2f} mm/d</b>"
                f"&nbsp;|&nbsp;Dr = {dr_today:.1f} mm / RAW = {raw1:.1f} mm<br>"
                f"{_ml_icon}&nbsp;<b>{_ml_lbl}</b><br>"
                f"<small style=\'color:#444;line-height:1.6\'>{_ml_exp}</small>"
            )
        else:
            _ml_css = "ml-panel"
            _ml_txt = (
                f"<b>\U0001f916 XGBoost ML</b> \u00b7 Model not loaded \u2014 "
                f"FAO-56 result is authoritative.&nbsp;"
                f"Place <code>{_MODEL_PATH.name}</code> next to <code>app.py</code>."
            )
        st.markdown(f'<div class="{_ml_css}">{_ml_txt}</div>', unsafe_allow_html=True)
        if irrigate:
            st.markdown(
                f'<div class="refill-box">🔧 <b>Refill to FC:</b> '
                f'IWR = <b>{iwr1:.2f} mm</b></div>',
                unsafe_allow_html=True)
            vol1 = compute_volume(iwr1, area_ha)
            st.markdown(
                f'<div class="iwr-box">💧 <b>IWR = {iwr1:.2f} mm/day</b> · '
                f'<b>{vol1["vol_m3"]:.1f} m³</b> = <b>{vol1["vol_L"]:,.0f} L</b> '
                f'for {area_ha} ha</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="vol-box">🪣 <b>Volume to apply: {vol1["vol_m3"]:.1f} m³ '
                f'= {vol1["vol_L"]:,.0f} litres</b> for <b>{area_ha} ha</b><br>'
                f'After irrigation → SM: <b>{sm_aft:.1f}% of FC</b> ✅</div>',
                unsafe_allow_html=True)
            st.warning(
                f"⚠️ **Irrigate today** — Dr = {dr_today:.2f} mm > RAW = {raw1:.1f} mm  \n"
                f"Apply **{iwr1:.2f} mm** = {vol1['vol_m3']:.1f} m³ ({vol1['vol_L']:,.0f} L)")
        else:
            vol1 = compute_volume(0., area_ha)
            st.markdown(
                f'<div class="no-irr-box">✅ <b>No irrigation needed today</b><br>'
                f'{note_today}</div>',
                unsafe_allow_html=True)
            if pe1 >= etc1:
                st.success(
                    f"🌧️ **Rain covers crop demand today.**  \n"
                    f"Pe = {pe1:.2f} mm ≥ ETc = {etc1:.2f} mm → NIR = 0, IWR = 0.")
            else:
                st.success(
                    f"✅ **Soil moisture adequate — no irrigation needed.**  \n"
                    f"Dr = {dr_today:.2f} mm ≤ RAW = {raw1:.1f} mm. "
                    f"NIR = {nir1_day:.2f} mm/d — soil buffer covers this today.")

        # ── TODAY'S SUMMARY TABLE ─────────────────────────────────────────────
        st.markdown("#### 📋 Today's Daily Summary")
        today_tbl = pd.DataFrame([{
            "Rain used mm":  round(prec_in,1),
            "ETc mm/d":      round(etc1,2),
            "Pe mm":         round(pe1,2),
            "Dr mm":         round(dr_today,2),
            "SM %FC":        round(sm_now,1),
            "NIR mm":        round(nir1_day,2),
            "IWR mm":        round(iwr1,2),
            "Vol m³":        vol1["vol_m3"],
            "Vol L":         int(vol1["vol_L"]),
            "ET₀ PM mm/d":   round(et0_fao,2),
            "ET₀ H mm/d":    round(et0_h,2),
            "Kc":            round(kc1,3),
            "TAW mm":        round(taw1,1),
            "RAW mm":        round(raw1,1),
            "MAD adj":       round(mad_eff1,3),
            "Status":        status_lbl,
            "ML NIR mm":     round(_ml_today, 3) if _ml_today is not None else None,
        }], index=[datetime.today().strftime("%Y-%m-%d (%a)")])

        _today_fmt = {
            "Rain used mm":  "{:.1f}", "ETc mm/d":    "{:.2f}", "Pe mm":      "{:.2f}",
            "Dr mm":         "{:.2f}", "SM %FC":      "{:.1f}", "NIR mm":     "{:.2f}",
            "IWR mm":        "{:.2f}", "Vol m³":      "{:.1f}", "Vol L":      "{:.0f}",
            "ET₀ PM mm/d":   "{:.2f}", "ET₀ H mm/d":  "{:.2f}", "Kc":        "{:.3f}",
            "TAW mm":        "{:.1f}", "RAW mm":      "{:.1f}", "MAD adj":    "{:.3f}",
            "ML NIR mm":     "{:.3f}",
        }
        st.dataframe(today_tbl.style.format(_today_fmt, na_rep="—"),
                     use_container_width=True)

        # ── DOWNLOADS ─────────────────────────────────────────────────────────
        if past_r is not None:
            combined_rows = []
            for dt, row in past_r.iterrows():
                _v = compute_volume(float(row["IWR"]), area_ha)
                combined_rows.append({
                    "Date": dt.strftime("%Y-%m-%d"), "Period": "Past",
                    "Rain mm": round(float(row.get("precipitation",0.)),1),
                    "ETc mm/d": round(float(row["ETc"]),2),
                    "Pe mm": round(float(row["Pe"]),2),
                    "Dr mm": round(float(row["Dr_mm"]),2),
                    "SM %FC": round(float(row["SM_pct"]),1),
                    "NIR mm": round(float(row["NIR"]),2),
                    "IWR mm": round(float(row["IWR"]),2),
                    "Vol m³": _v["vol_m3"],
                    "Vol L":  int(_v["vol_L"]),
                    "Status": row["Status"],
                })
            combined_rows.append({
                "Date": datetime.today().strftime("%Y-%m-%d"), "Period": "TODAY",
                "Rain mm": round(prec_in,1), "ETc mm/d": round(etc1,2),
                "Pe mm": round(pe1,2), "Dr mm": round(dr_today,2),
                "SM %FC": round(sm_now,1), "NIR mm": round(nir1_day,2),
                "IWR mm": round(iwr1,2),
                "Vol m³": vol1["vol_m3"], "Vol L": int(vol1["vol_L"]),
                "Status": status_lbl,
            })
            combined_df = pd.DataFrame(combined_rows).set_index("Date")
            dl_csv  = combined_df.to_csv().encode()
            dl_xlsx = df_to_excel_bytes({"Past5Days+Today": combined_df, "TodaySummary": today_tbl})
        else:
            dl_csv  = today_tbl.to_csv().encode()
            dl_xlsx = df_to_excel_bytes({"TodaySummary": today_tbl})

        _fn = f"HyPIS_{SITE_NAME.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}"
        _show_download_buttons(dl_csv, dl_xlsx, _fn, "dl_today_csv", "dl_today_xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — 5-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header(f"☁️ 5-Day IWR Forecast — {SITE_NAME}")
    st.caption("FAO-56 Penman-Monteith · Dr > RAW trigger")

    fc_c1, fc_c2 = st.columns(2)
    with fc_c1:
        cr2    = st.selectbox("Crop", list(crop_params.keys()), key="cr2")
        cp2    = crop_params[cr2]
        stage2 = st.radio("Growing Stage", list(STAGE_LABELS.keys()),
                          format_func=lambda x: STAGE_LABELS[x], key="stg2", horizontal=True)
        kc2    = kc_from_stage(stage2, cr2)
        mad2   = adjust_mad_for_soil(cp2["mad"], ACTIVE_TXT)
        st.markdown(f'<div class="kc-stage">Kc = <b>{kc2:.3f}</b> · '
                    f'MAD = {cp2["mad"]:.2f} → soil-adj. <b>{mad2:.2f}</b></div>',
                    unsafe_allow_html=True)
    with fc_c2:
        planting2 = st.date_input("Planting Date",
                                  value=datetime.today().date()-timedelta(days=45), key="plant2")
        soil2   = {"fc":ACTIVE_FC,"pwp":ACTIVE_PWP}
        # v7.7: quick-set buttons for dry/wet season simulation
        # FIX v7.7b: set slider key "sm2" directly — value= param is ignored once key exists
        st.markdown("**🌡️ Season scenario (sets starting soil moisture):**")
        _sc1, _sc2, _sc3 = st.columns(3)
        if _sc1.button("☀️ Dry (30%)",  key="sm2_dry",  use_container_width=True):
            st.session_state["sm2"] = 30
        if _sc2.button("🌤️ Mod (55%)", key="sm2_mod",  use_container_width=True):
            st.session_state["sm2"] = 55
        if _sc3.button("🌧️ Wet (80%)", key="sm2_wet",  use_container_width=True):
            st.session_state["sm2"] = 80
        _sm2_default = estimate_sm(ACTIVE_FC,ACTIVE_PWP,cp2["zr"],LAT,LON,ELEV)
        sm_pct2 = st.slider("Starting SM (% of FC)", 0, 100, _sm2_default, key="sm2")
        st.markdown(f'<div class="soil-panel">🌍 <b>{ACTIVE_TXT}</b> · '
                    f'FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%<br>'
                    f'<small>💡 <b>Uganda rainy season (Mar–May, Oct–Nov):</b> soil stays moist '
                    f'→ use ☀️ Dry to simulate dry season or irrigation planning.</small></div>',
                    unsafe_allow_html=True)

    if st.button("📥 Get 5-Day Forecast", type="primary",
                 use_container_width=True, key="fc_btn"):
        with st.spinner("Fetching forecast…"):
            daily = get_forecast(_ch, LAT, LON, ELEV)

        if daily is None or daily.empty:
            st.warning("⚠️ Forecast unavailable.")
        else:
            daily_r, taw2, raw2 = run_water_balance(
                daily, cr2, soil2, pd.Timestamp(planting2),
                sm_pct2, stage_override=stage2, mad_eff=mad2)

            daily_r["Vol_m3"] = daily_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_m3"])
            daily_r["Vol_L"]  = daily_r["IWR"].apply(
                lambda x: float(compute_volume(x, area_ha)["vol_L"]))

            nd     = (daily_r["IWR"]>0).sum()
            rain_d = (daily_r.get("precipitation", pd.Series(0,index=daily_r.index))>1).sum()

            st.info(
                f"📐 TAW = **{taw2:.1f} mm** · RAW = **{raw2:.1f} mm** "
                f"(MAD {cp2['mad']:.2f}→{mad2:.2f})  \n"
                f"Irrigation fires when Dr > {raw2:.1f} mm")

            if nd > 0:
                st.warning(
                    f"🗓️ **{nd} irrigation event(s)** · {rain_d} rainy day(s)  \n"
                    f"Total IWR = **{daily_r['IWR'].sum():.1f} mm** · "
                    f"Total Volume = **{daily_r['Vol_m3'].sum():.1f} m³** "
                    f"({daily_r['Vol_L'].sum():,.0f} L) for {area_ha} ha")
            else:
                rain_total = daily_r.get("precipitation", pd.Series(0,index=daily_r.index)).sum()
                # v8.0: context-aware UX message — explains WHY IWR is zero
                # so users don't mistake correct behaviour for a bug
                if sm_pct2 >= 80:
                    st.info(
                        f"ℹ️ **No irrigation needed — soil already near field capacity.**  \n"
                        f"Starting SM = **{sm_pct2}%** (Dr well below RAW = {raw2:.1f} mm) "
                        f"and forecast rain ({rain_total:.1f} mm) covers crop demand.  \n"
                        f"This is correct FAO-56 behaviour — not a bug.  \n"
                        f"👉 Press **☀️ Dry (30%)** to simulate a dry-season scenario "
                        f"and see multi-day irrigation events.")
                elif sm_pct2 >= 65:
                    st.success(
                        f"✅ **No irrigation needed over next 5 days.**  \n"
                        f"Soil starts at **{sm_pct2}% SM** (Dr below RAW = {raw2:.1f} mm) "
                        f"and forecast rain ({rain_total:.1f} mm total) keeps it replenished.  \n"
                        f"⚠️ **Uganda rainy season (Mar–May, Oct–Nov): this is expected.**  \n"
                        f"To see dry-season irrigation needs: press **☀️ Dry (30%)** above.")
                else:
                    st.success(
                        f"✅ **No irrigation needed over next 5 days.**  \n"
                        f"({rain_d} rainy day(s) · Dr stays below RAW = {raw2:.1f} mm throughout)")

            cols2 = st.columns(len(daily_r))
            for i,(dt,row) in enumerate(daily_r.iterrows()):
                icon  = wmo_icon(row.get("weather_code",0))
                _f_val = float(row["IWR"])
                if _f_val > 0:
                    lbl = f"💧 {_f_val:.1f} mm IWR"
                    _vol_d = row["Vol_m3"] if "Vol_m3" in row.index else compute_volume(_f_val, area_ha)["vol_m3"]
                    dlt = f"🪣 {_vol_d:.1f} m³ · {icon}"
                else:
                    lbl = f"{icon} No irrigation"
                    dlt = f"Dr={row['Dr_mm']:.1f}mm · NIR={row['NIR']:.1f}mm"
                cols2[i].metric(dt.strftime("%a %d"), lbl, dlt)

            st.subheader("📋 5-Day Forecast Table")
            cols_fc = ["tmax","tmin","rh_mean","precipitation","Pe",
                       "ET0","kc","ETc","Dr_mm","SM_pct",
                       "NIR","ML_NIR","IWR","Vol_m3","Vol_L","Status"]
            tb2 = daily_r[[c for c in cols_fc if c in daily_r.columns]].copy()
            tb2.rename(columns={
                "tmax":"Tmax °C","tmin":"Tmin °C","rh_mean":"RH %",
                "precipitation":"Rain mm","Pe":"Pe mm",
                "ET0":"ET₀ mm/d","kc":"Kc","ETc":"ETc mm/d",
                "Dr_mm":"Dr mm","SM_pct":"SM %FC",
                "NIR":"NIR mm","ML_NIR":"🤖 ML NIR mm","IWR":"IWR mm",
                "Vol_m3":"Vol m³","Vol_L":"Vol L","Status":"Status",
            }, inplace=True)
            tb2.index = tb2.index.strftime("%Y-%m-%d (%a)")
            _tb2_fmt = {
                "Tmax °C":"{:.1f}","Tmin °C":"{:.1f}","RH %":"{:.0f}",
                "Rain mm":"{:.1f}","Pe mm":"{:.2f}","ET₀ mm/d":"{:.2f}",
                "Kc":"{:.3f}","ETc mm/d":"{:.2f}","Dr mm":"{:.2f}",
                "SM %FC":"{:.1f}","NIR mm":"{:.2f}","🤖 ML NIR mm":"{:.2f}",
                "IWR mm":"{:.2f}","Vol m³":"{:.1f}","Vol L":"{:.0f}",
            }
            st.dataframe(tb2.style.format(_tb2_fmt, na_rep="—"), use_container_width=True)
            st.caption(
                "**IWR mm** = irrigation depth (mm/day).  "
                "**Vol m³ / Vol L** = total volume for your field area.  "
                "**🤖 ML NIR mm** = XGBoost cross-check (scaled for soil state)."
            )

            # ── v8.1: FORECAST CONSISTENCY TRACKER ──────────────────────────
            # Store this forecast in session_state. On the next run, compare
            # today's computed IWR to what was forecast for today last time.
            # This implements the "decision locking" concept from the design docs:
            # if yesterday said IWR=0 for today, we explain whether that holds.
            _fc_key = "last_5day_forecast"
            _prev_fc = st.session_state.get(_fc_key, {})
            # Store current forecast: date_str → {"IWR": float, "Hybrid": float}
            _new_fc = {
                dt.strftime("%Y-%m-%d"): {
                    "IWR": float(row["IWR"]),
                                        "SM": float(row.get("SM_pct", 100.)),
                }
                for dt, row in daily_r.iterrows()
            }
            st.session_state[_fc_key] = _new_fc

            # Compare: for any day that was in the previous forecast, has the
            # IWR decision changed significantly?
            _consistency_notes = []
            for _dstr, _cur in _new_fc.items():
                if _dstr in _prev_fc:
                    _prev_iwr = _prev_fc[_dstr]["IWR"]
                    _cur_iwr  = _cur["IWR"]
                    _prev_was_zero = _prev_iwr <= 0
                    _cur_is_nonzero = _cur_iwr > 0
                    if _prev_was_zero and _cur_is_nonzero:
                        _consistency_notes.append(
                            f"**{_dstr}**: Previous forecast said no irrigation (0 mm), "
                            f"today's run now shows FAO IWR = {_cur_iwr:.1f} mm. "
                            f"Likely cause: updated weather data changed Dr or Pe."
                        )
                    elif not _prev_was_zero and not _cur_is_nonzero:
                        _consistency_notes.append(
                            f"**{_dstr}**: Previous forecast indicated irrigation "
                            f"({_prev_iwr:.1f} mm), today's run now shows 0 mm. "
                            f"Likely cause: rain forecast increased Pe above ETc."
                        )

            if _consistency_notes and _prev_fc:
                with st.expander("🔄 Forecast Consistency — changes since last run", expanded=True):
                    st.caption(
                        "FAO-56 recomputes from fresh weather data every run. "
                        "Changes between runs reflect updated forecasts — not instability.")
                    for _note in _consistency_notes:
                        st.markdown(f"⚠️ {_note}")
                    st.markdown(
                        "*This is expected behaviour: a 5-day forecast updates as new weather "
                        "data arrives. The irrigation trigger (Dr > RAW) is physically consistent "
                        "— only the weather inputs change.*")

            # ── v8.0: per-day ML agreement summary ───────────────────────────
            if ML_OK and "ML_NIR" in daily_r.columns:
                with st.expander("🤖 ML Cross-Check · Per-day agreement & hybrid decisions", expanded=False):
                    st.caption(
                        "**FAO-56** is the irrigation trigger — it knows today's soil state (Dr). "
                        "**ML NIR (adj)** is soil-state adjusted: near 0 when soil is full, "
                        "higher when soil is depleted. Divergence from FAO is expected — "
                        "ML cannot override the physics.")
                    prec_col2 = "precipitation" if "precipitation" in daily_r.columns else None
                    for dt, row2 in daily_r.iterrows():
                        _ml2   = row2.get("ML_NIR", None)
                        _ml2r  = row2.get("ML_NIR_Raw", None)
                        _nir2  = float(row2.get("NIR", 0.))
                        _dr2   = float(row2.get("Dr_mm", 0.))
                        _rain2 = float(row2.get(prec_col2, 0.)) if prec_col2 else 0.
                        _fao2  = float(row2.get("IWR", 0.))
                        _conf2 = row2.get("Confidence", None)

                        _css2, _icon2, _lbl2, _exp2, _pct2 = ml_agreement(
                            _nir2, _ml2, dr_mm=_dr2, raw_mm=raw2, rainfall_mm=_rain2
                        )
                        _conf2_str = (
                            f"🟢 {int(_conf2*100)}%" if _conf2 and _conf2 >= 0.85
                            else (f"🟡 {int(_conf2*100)}%" if _conf2 and _conf2 >= 0.65
                                  else (f"🟠 {int(_conf2*100)}%" if _conf2 else ""))
                        ) if _conf2 is not None else ""

                        _raw_note = (
                            f" <small style='color:#888'>(raw XGB: {_ml2r:.1f} mm)</small>"
                            if _ml2r is not None and _ml2 is not None and abs((_ml2r or 0.) - (_ml2 or 0.)) > 0.5
                            else ""
                        )
                        st.markdown(
                            f'<div class="{_css2}" style="margin-bottom:4px">'
                            f'<b>{dt.strftime("%a %d %b")}</b> &nbsp;·&nbsp; '
                            f'ML(adj) = {(f"{_ml2:.2f} mm" if _ml2 is not None else "N/A")}'
                            f'{_raw_note}'
                            f' &nbsp;·&nbsp; FAO NIR = {_nir2:.2f} mm'
                            f' &nbsp;·&nbsp; FAO IWR = {_fao2:.2f} mm'
                            f' &nbsp;·&nbsp; {_icon2} <b>{_lbl2}</b>'
                            f' &nbsp;{_conf2_str}<br>'
                            f'<small style="color:#444">{_exp2}</small></div>',
                            unsafe_allow_html=True)

            fig2d = go.Figure()
            fig2d.add_scatter(x=daily_r.index.strftime("%a %d"), y=daily_r["Dr_mm"].astype(float),
                              mode="lines+markers+text", name="Dr mm",
                              text=daily_r["Dr_mm"].round(1).astype(str),
                              textposition="top center",
                              line=dict(color="#e6550d",width=2))
            if "precipitation" in daily_r.columns:
                fig2d.add_bar(x=daily_r.index.strftime("%a %d"),
                              y=daily_r["precipitation"].astype(float),
                              name="Rain mm", marker_color="#1a5fc8",opacity=0.4,yaxis="y2")
            fig2d.add_bar(x=daily_r.index.strftime("%a %d"), y=daily_r["IWR"].astype(float),
                          name="IWR mm", marker_color="#17a2b8",opacity=0.7,yaxis="y2")
            fig2d.add_bar(x=daily_r.index.strftime("%a %d"), y=daily_r["NIR"].astype(float),
                          name="NIR mm/d", marker_color="#0b6b1b",opacity=0.5,yaxis="y2")
            # ML NIR (adj) — soil-state-aware; shows near 0 when soil is full
            if "ML_NIR" in daily_r.columns and daily_r["ML_NIR"].notna().any():
                fig2d.add_scatter(
                    x=daily_r.index.strftime("%a %d"),
                    y=daily_r["ML_NIR"].fillna(0).astype(float),
                    mode="lines+markers", name="🤖 ML NIR (adj)",
                    line=dict(color="#7c4dff", width=2, dash="dot"),
                    marker=dict(size=7, symbol="diamond"), yaxis="y2")
            fig2d.add_hline(y=raw2,line_dash="dash",line_color="#756bb1",
                            annotation_text=f"RAW={raw2:.1f} mm")
            fig2d.add_hline(y=taw2,line_dash="dot", line_color="#d73027",
                            annotation_text=f"TAW={taw2:.1f} mm")
            fig2d.update_layout(
                title=f"5-Day: Dr, NIR, IWR — {SITE_NAME}",
                yaxis=dict(title="Dr (mm)"),
                yaxis2=dict(title="mm/d",overlaying="y",side="right"),
                barmode="group",legend=dict(x=0,y=1.12,orientation="h"),
                height=400,plot_bgcolor="#f4f8f2",paper_bgcolor="#f4f8f2")
            st.plotly_chart(fig2d, use_container_width=True)

            dl_fc_csv  = tb2.to_csv().encode()
            dl_fc_xlsx = df_to_excel_bytes({"5DayForecast": tb2})
            _fn_fc = f"HyPIS_forecast_{SITE_NAME.replace(' ','_')}_{date.today()}"
            _show_download_buttons(dl_fc_csv, dl_fc_xlsx, _fn_fc, "dl_fc_csv", "dl_fc_xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header(f"📅 Historical IWR Analysis — {SITE_NAME}")
    st.caption("ERA5 Archive · FAO-56 Penman-Monteith · Dr > RAW trigger")

    st.info(
        f"📍 **{SITE_NAME}** · `{LAT}°, {LON}°, {ELEV} m`  \n"
        f"🌍 Soil: **{ACTIVE_TXT}** · FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%  \n"
        f"*(ERA5 may lag 3–5 days — yesterday is the closest available)*")

    pc1,pc2,pc3 = st.columns(3)
    yesterday = date.today()-timedelta(days=1)
    if pc1.button("📅 Yesterday",    use_container_width=True, key="h_yest"):
        st.session_state["h_start"]=yesterday; st.session_state["h_end"]=yesterday
    if pc2.button("📅 Last 7 Days",  use_container_width=True, key="h_7"):
        st.session_state["h_start"]=yesterday-timedelta(days=6); st.session_state["h_end"]=yesterday
    if pc3.button("📅 Last 30 Days", use_container_width=True, key="h_30"):
        st.session_state["h_start"]=yesterday-timedelta(days=29); st.session_state["h_end"]=yesterday

    h_start = st.session_state.get("h_start", yesterday-timedelta(days=6))
    h_end   = st.session_state.get("h_end",   yesterday)
    st.markdown(f"**Period:** `{h_start}` → `{h_end}` ({(h_end-h_start).days+1} days)")

    hc1,hc2 = st.columns(2)
    with hc1:
        cr3    = st.selectbox("Crop",list(crop_params.keys()),key="cr3")
        cp3    = crop_params[cr3]
        stage3 = st.radio("Growing Stage",list(STAGE_LABELS.keys()),
                          format_func=lambda x:STAGE_LABELS[x],key="stg3",horizontal=True)
        kc3    = kc_from_stage(stage3, cr3)
        mad3   = adjust_mad_for_soil(cp3["mad"], ACTIVE_TXT)
        st.markdown(f'<div class="kc-stage">Kc={kc3:.3f} · MAD {cp3["mad"]:.2f}→{mad3:.2f}</div>',
                    unsafe_allow_html=True)
    with hc2:
        planting3 = st.date_input("Planting Date",value=date.today()-timedelta(days=45),key="plant3")
        soil3  = {"fc":ACTIVE_FC,"pwp":ACTIVE_PWP}
        # v7.7: quick-set buttons
        # FIX v7.7b: set slider key "sm3" directly
        st.markdown("**🌡️ Season scenario:**")
        _h1, _h2, _h3 = st.columns(3)
        if _h1.button("☀️ Dry (30%)",  key="sm3_dry", use_container_width=True):
            st.session_state["sm3"] = 30
        if _h2.button("🌤️ Mod (55%)", key="sm3_mod", use_container_width=True):
            st.session_state["sm3"] = 55
        if _h3.button("🌧️ Wet (80%)", key="sm3_wet", use_container_width=True):
            st.session_state["sm3"] = 80
        _sm3_def = estimate_sm(ACTIVE_FC,ACTIVE_PWP,cp3["zr"],LAT,LON,ELEV)
        sm3 = st.slider("Starting SM (% of FC)", 0, 100, _sm3_def, key="sm3")
        st.markdown(f'<div class="soil-panel">🌍 <b>{ACTIVE_TXT}</b> · '
                    f'FC={ACTIVE_FC*100:.0f}% · PWP={ACTIVE_PWP*100:.0f}%</div>',
                    unsafe_allow_html=True)

    if st.button("📥 Retrieve Historical Data", type="primary",
                 use_container_width=True, key="hist_btn"):
        with st.spinner("Fetching ERA5 archive…"):
            hist = get_historical_weather(str(h_start),str(h_end),LAT,LON)

        if hist is None or hist.empty:
            st.warning("⚠️ No ERA5 data for this period.")
        else:
            hist_r, taw3, raw3 = run_water_balance(
                hist,cr3,soil3,pd.Timestamp(planting3),sm3,
                stage_override=stage3, mad_eff=mad3)

            hist_r["Vol_m3"] = hist_r["IWR"].apply(
                lambda x: compute_volume(x, area_ha)["vol_m3"])
            hist_r["Vol_L"]  = hist_r["IWR"].apply(
                lambda x: float(compute_volume(x, area_ha)["vol_L"]))
            hist_r["ET0_H"]   = [et0_hargreaves(r["tmax"],r["tmin"],
                                  doy=int(d.strftime("%j")),lat_deg=LAT)
                                 for d,r in hist_r.iterrows()]
            # v7.8: ML cross-check per row (already in ML_NIR from run_water_balance,
            # but recompute explicitly here to ensure it's populated)
            if "ML_NIR" not in hist_r.columns or hist_r["ML_NIR"].isna().all():
                hist_r["ML_NIR"] = [
                    ml_predict_iwr(
                        (r["tmax"]+r["tmin"])/2., r.get("rh_mean",60.),
                        r.get("wind",1.5), r.get("kc", cp3["mid"]),
                        r.get("precipitation",0.), ACTIVE_FC, ACTIVE_PWP, cp3["zr"],
                        taw=taw3,
                        dr_ratio=min(1., float(r.get("Dr_mm",0.))/taw3) if taw3>0 else 0.)
                    for _,r in hist_r.iterrows()]
            st.info(
                f"📐 TAW = **{taw3:.1f} mm** · RAW = **{raw3:.1f} mm** "
                f"(MAD {cp3['mad']:.2f}→{mad3:.2f})  \n"
                f"Irrigation fires when Dr > {raw3:.1f} mm")

            _ml_h_col  = hist_r["ML_NIR"].dropna()
            _ml_h_mean = _ml_h_col.mean() if len(_ml_h_col) else None
            _ml_h_dev  = (abs(hist_r["NIR"] - hist_r["ML_NIR"].fillna(hist_r["NIR"]))).mean() if ML_OK else None
            m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
            m1.metric("📆 Days",        len(hist_r))
            m2.metric("🌧️ Rain Total",  f"{hist_r['precipitation'].sum():.1f} mm")
            m3.metric("💧 NIR Total",   f"{hist_r['NIR'].sum():.1f} mm")
            m4.metric("💧 IWR Total",   f"{hist_r['IWR'].sum():.1f} mm")
            m5.metric("🪣 Vol Total",   f"{hist_r['Vol_m3'].sum():.1f} m³")
            m6.metric("🚿 Irrig Days",  str((hist_r["IWR"]>0).sum()))
            m7.metric("🌧️ Rain Days",   str((hist_r["precipitation"]>1).sum()))
            if ML_OK and _ml_h_dev is not None:
                m8.metric("🤖 ML Avg Dev", f"{_ml_h_dev:.2f} mm",
                          f"ML mean NIR {_ml_h_mean:.2f} mm" if _ml_h_mean else "n/a")
            else:
                m8.metric("🤖 ML", "N/A", "model not loaded")

            st.subheader("📋 Historical Table")
            _ht_sel = ["tmax","tmin","rh_mean","precipitation","Pe",
                       "ET0","ET0_H","kc","ETc","Dr_mm","SM_pct",
                       "NIR","ML_NIR","IWR","Vol_m3","Vol_L","Status","Note"]
            ht = hist_r[[c for c in _ht_sel if c in hist_r.columns]].copy()
            _ht_col_map = {
                "tmax":"Tmax °C","tmin":"Tmin °C","rh_mean":"RH %",
                "precipitation":"Rain mm","Pe":"Pe mm",
                "ET0":"ET₀ PM mm","ET0_H":"ET₀ H mm","kc":"Kc","ETc":"ETc mm/d",
                "Dr_mm":"Dr mm","SM_pct":"SM %FC",
                "NIR":"NIR mm","ML_NIR":"🤖 ML NIR mm","IWR":"IWR mm",
                "Vol_m3":"Vol m³","Vol_L":"Vol L",
                "Status":"Status","Note":"Decision Note",
            }
            ht.rename(columns=_ht_col_map, inplace=True)
            ht.index = ht.index.strftime("%Y-%m-%d (%a)")
            _ht_fmt = {
                "Tmax °C":   "{:.1f}", "Tmin °C":   "{:.1f}", "RH %":      "{:.0f}",
                "Rain mm":   "{:.1f}", "Pe mm":      "{:.2f}",
                "ET₀ PM mm": "{:.2f}", "ET₀ H mm":  "{:.2f}", "Kc":        "{:.3f}",
                "ETc mm/d":  "{:.2f}", "Dr mm":      "{:.2f}", "SM %FC":    "{:.1f}",
                "NIR mm":    "{:.2f}", "🤖 ML NIR mm":"{:.2f}", "IWR mm":  "{:.2f}",
                "Vol m³":    "{:.1f}", "Vol L":      "{:.0f}",
            }
            st.dataframe(ht.style.format(_ht_fmt, na_rep="—"), use_container_width=True)

            fig3d = go.Figure()
            fig3d.add_scatter(x=hist_r.index,y=hist_r["Dr_mm"].astype(float),mode="lines",
                              name="Dr mm",line=dict(color="#e6550d",width=1.5))
            fig3d.add_bar(x=hist_r.index,y=hist_r["precipitation"].astype(float),
                          name="Rain mm",marker_color="#1a5fc8",opacity=0.35,yaxis="y2")
            fig3d.add_bar(x=hist_r.index,y=hist_r["IWR"].astype(float),
                          name="IWR mm",marker_color="#17a2b8",opacity=0.7,yaxis="y2")
            fig3d.add_bar(x=hist_r.index,y=hist_r["NIR"].astype(float),
                          name="NIR mm/d",marker_color="#0b6b1b",opacity=0.4,yaxis="y2")
            # v7.8: ML NIR overlay
            if "ML_NIR" in hist_r.columns and hist_r["ML_NIR"].notna().any():
                fig3d.add_scatter(x=hist_r.index, y=hist_r["ML_NIR"].astype(float),
                                  mode="lines", name="🤖 ML NIR mm",
                                  line=dict(color="#7c4dff",width=1.5,dash="dot"),
                                  yaxis="y2")
            fig3d.add_hline(y=raw3,line_dash="dash",line_color="#756bb1",
                            annotation_text=f"RAW={raw3:.1f} mm — irrigate above")
            fig3d.add_hline(y=taw3,line_dash="dot", line_color="#d73027",
                            annotation_text=f"TAW={taw3:.1f} mm — wilting risk")
            fig3d.update_layout(
                title="Historical Dr, NIR, IWR vs Thresholds",
                yaxis=dict(title="Dr (mm)"),
                yaxis2=dict(title="mm",overlaying="y",side="right"),
                barmode="overlay",legend=dict(x=0,y=1.12,orientation="h"),
                height=420,plot_bgcolor="#f4f8f2",paper_bgcolor="#f4f8f2")
            st.plotly_chart(fig3d, use_container_width=True)

            fig3e = go.Figure()
            fig3e.add_scatter(x=hist_r.index,y=hist_r["ET0"].astype(float),name="ET₀ PM",
                              mode="lines",line=dict(color="#1a5fc8",width=1.5))
            fig3e.add_scatter(x=hist_r.index,y=hist_r["ET0_H"].astype(float),
                              name="ET₀ Hargreaves",
                              mode="lines",line=dict(color="#b81c1c",width=1,dash="dot"))
            fig3e.add_scatter(x=hist_r.index,y=hist_r["ETc"].astype(float),name="ETc",
                              mode="lines",line=dict(color="#0b6b1b",width=1.5,dash="dash"))
            fig3e.update_layout(title="ET₀ & ETc",yaxis_title="mm/d",
                                height=300,plot_bgcolor="#f4f8f2",paper_bgcolor="#f4f8f2",
                                legend=dict(x=0,y=1.12,orientation="h"))
            st.plotly_chart(fig3e, use_container_width=True)

            irrig_d = hist_r[hist_r["IWR"]>0]
            if not irrig_d.empty:
                fig3v = go.Figure()
                fig3v.add_bar(x=irrig_d.index,y=irrig_d["Vol_m3"].astype(float),
                              name="Volume m³",marker_color="#0d6efd",
                              text=irrig_d["Vol_m3"].round(1).astype(str)+" m³",
                              textposition="outside")
                fig3v.update_layout(
                    title=f"Irrigation Volume (m³) — {area_ha} ha",
                    yaxis_title="m³",height=280,
                    plot_bgcolor="#f4f8f2",paper_bgcolor="#f4f8f2")
                st.plotly_chart(fig3v, use_container_width=True)
            else:
                st.success("✅ No irrigation events in this period — "
                           "rain and soil moisture were sufficient throughout.")

            dl_h_csv  = ht.to_csv().encode()
            dl_h_xlsx = df_to_excel_bytes({"Historical": ht})
            _fn_h = f"HyPIS_hist_{SITE_NAME.replace(' ','_')}_{h_start}_{h_end}"
            _show_download_buttons(dl_h_csv, dl_h_xlsx, _fn_h, "dl_hist_csv", "dl_hist_xlsx")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"HyPIS Ug · {SITE_NAME} ({LAT}°, {LON}°, {ELEV} m) · Byaruhanga Prosper"
)
