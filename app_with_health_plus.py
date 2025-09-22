
# app.py — Property Price Estimator (with Model Health badge + MAE history)
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -------------------- CONFIG --------------------
ARTIFACTS_DIR_DEFAULT = "artifacts"
MODEL_FILENAME = "modelo_xgb_pipeline.joblib"
GLOBAIS_FILENAME = "faixa_globais.json"
BINNED_FILENAME  = "faixas_binned.json"   # optional
FEATURE_COLS_JSON = "feature_cols.json"   # optional (from preprocess)
METRICS_JSON = "metrics.json"             # optional (contains test/val MAE)

# Fallback expected columns (if we cannot infer from the pipeline/JSON)
FALLBACK_EXPECTED = [
    "Rooms", "Bedroom2", "Bathroom", "Car", "Distance",
    "Latitude", "Longitude", "Propertycount", "Month", "Year",
    "Type", "Method",
]

# Normalized name -> canonical expected name (for uploads)
NORMALIZED_TO_EXPECTED = {
    "rooms": "Rooms",
    "bedroom2": "Bedroom2",
    "bathroom": "Bathroom",
    "car": "Car",
    "distance": "Distance",
    "latitude": "Latitude",
    "lattitude": "Latitude",
    "longitude": "Longitude",
    "longtitude": "Longitude",
    "propertycount": "Propertycount",
    "month": "Month",
    "year": "Year",
    "type": "Type",
    "method": "Method",
}

# Reference MAE (adjust to your validation result if desired)
MAE_REFERENCE_DEFAULT = 149_000
# -------------------------------------------------


# ----------------- HELPERS -----------------------
def normalize_name(s: str) -> str:
    import unicodedata, re
    s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def expected_columns_from_pipeline(pipe) -> Optional[List[str]]:
    """Try to extract raw expected columns from the ColumnTransformer (pipeline['prep'])."""
    try:
        ct = pipe.named_steps.get("prep") or pipe["prep"]
    except Exception:
        return None

    cols: List[str] = []
    tr_list = getattr(ct, "transformers_", None) or getattr(ct, "transformers", None) or []
    for name, trans, columns in tr_list:
        if name == "remainder" or columns is None:
            continue
        if isinstance(columns, (list, tuple, pd.Index, np.ndarray)):
            cols.extend([str(c) for c in columns])

    # unique preserving order
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered or None


@st.cache_resource
def load_model_and_schema(model_path: Path, artifacts_dir: Path) -> Tuple[object, List[str]]:
    pipe = joblib.load(model_path)

    # 1) From pipeline
    cols = expected_columns_from_pipeline(pipe)

    # 2) From preprocess JSON
    if not cols:
        json_path = artifacts_dir / FEATURE_COLS_JSON
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            cols = list(info.get("features_all") or [])

    # 3) Fallback
    if not cols:
        cols = FALLBACK_EXPECTED

    return pipe, cols


@st.cache_resource
def load_bands(globais_path: Path, binned_path: Path) -> Tuple[Dict, pd.DataFrame]:
    with open(globais_path, "r", encoding="utf-8") as f:
        globais = json.load(f)

    bands_df = pd.DataFrame()
    if binned_path.exists():
        # Format from advanced_models.py: {"q": 10, "bins": [{pred_min, pred_max, p10, p90, ...}, ...]}
        with open(binned_path, "r", encoding="utf-8") as f:
            binned = json.load(f)
        bins = binned.get("bins") if isinstance(binned, dict) else binned
        if isinstance(bins, list):
            bands_df = pd.DataFrame(bins)
            bands_df.rename(columns={"y_pred_min": "pred_min", "y_pred_max": "pred_max"}, inplace=True)
            if {"pred_min", "pred_max"}.issubset(bands_df.columns):
                bands_df = bands_df.sort_values("pred_min").reset_index(drop=True)
            else:
                bands_df = pd.DataFrame()
    return globais, bands_df


def interval_from_bands(yhat: float, bands: pd.DataFrame, globais: Dict) -> Tuple[float, float]:
    if bands is None or bands.empty:
        return yhat + float(globais.get("p10", 0.0)), yhat + float(globais.get("p90", 0.0))
    row = bands[(yhat >= bands.get("pred_min", pd.Series(dtype=float))) & (yhat <= bands.get("pred_max", pd.Series(dtype=float)))]
    if row.empty:
        return yhat + float(globais.get("p10", 0.0)), yhat + float(globais.get("p90", 0.0))
    return yhat + float(row.iloc[0].get("p10", 0.0)), yhat + float(row.iloc[0].get("p90", 0.0))


def round_5k(x: float) -> int:
    return int(round(float(x) / 5000.0) * 5000)


def suggested_price(yhat: float, mae_ref: float, objective: str = "balanced") -> int:
    if objective == "fast_sale":
        return round_5k(yhat - 0.30 * mae_ref)
    if objective == "max_margin":
        return round_5k(yhat + 0.20 * mae_ref)
    return round_5k(yhat)


def classify(asking_price: float, lower: float, upper: float) -> str:
    if pd.isna(asking_price):
        return ""
    if asking_price < lower:
        return "BELOW MARKET"
    if asking_price > upper:
        return "ABOVE MARKET"
    return "FAIR RANGE"


def coerce_expected_dataframe(df_in: pd.DataFrame, expected_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize/rename uploaded columns to what the pipeline expects; returns df and missing list."""
    norm_map = {c: normalize_name(c) for c in df_in.columns}
    df_tmp = df_in.rename(columns=norm_map)

    rename_to_expected: Dict[str, str] = {}
    for exp in expected_cols:
        norm = normalize_name(exp)
        if norm in df_tmp.columns:
            rename_to_expected[norm] = exp
        else:
            synonym = NORMALIZED_TO_EXPECTED.get(norm)
            if synonym and normalize_name(synonym) in df_tmp.columns:
                rename_to_expected[normalize_name(synonym)] = exp

    df_tmp = df_tmp.rename(columns=rename_to_expected)

    missing = [c for c in expected_cols if c not in df_tmp.columns]
    for c in missing:
        df_tmp[c] = np.nan

    df_tmp = df_tmp[expected_cols]
    return df_tmp, missing


# ---------- Model health helpers (Reference MAE) ----------
def classify_mae(mae: Optional[float], ref: Optional[float]) -> Tuple[str, str]:
    """Return (label, emoji) based on mae/ref ratio with tighter thresholds (1.0 / 1.2)."""
    try:
        if mae is None or ref is None or ref <= 0:
            return "—", "⚪"
        ratio = float(mae) / float(ref)
    except Exception:
        return "—", "⚪"

    if ratio <= 1.0:
        return "OK", "🟢"
    if ratio <= 1.2:
        return "Attention", "🟠"
    return "Retrain", "🔴"


def status_color(label: str) -> str:
    return {
        "OK": "#4CAF50",         # green
        "Attention": "#FFC107",  # amber
        "Retrain": "#F44336",    # red
    }.get(label, "#9E9E9E")      # grey fallback


def log_mae(live_mae: Optional[float], status: str, artifacts_dir: Path) -> None:
    """Append a row to mae_log.csv inside artifacts_dir if live_mae is available."""
    if live_mae is None:
        return
    try:
        log_path = artifacts_dir / "mae_log.csv"
        row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mae": float(live_mae),
            "status": status,
        }])
        if log_path.exists():
            row.to_csv(log_path, mode="a", header=False, index=False)
        else:
            row.to_csv(log_path, index=False)
    except Exception as e:
        st.warning(f"Could not log MAE: {e}")
# -------------------------------------------------


# ----------------- UI / LAYOUT -------------------
st.set_page_config(page_title="Property Price Estimator", layout="centered")
st.title("Property Price Estimator — Model Demo")

with st.sidebar:
    st.header("Settings")
    artifacts_dir_str = st.text_input("Artifacts folder", ARTIFACTS_DIR_DEFAULT)
    mae_ref = st.number_input("Reference MAE", min_value=0, value=MAE_REFERENCE_DEFAULT, step=1000)
    st.caption(
        "The folder must contain: "
        f"`{MODEL_FILENAME}`, `{GLOBAIS_FILENAME}`, `{BINNED_FILENAME}` (optional), and ideally `{FEATURE_COLS_JSON}`."
    )

    # --- Model Health (Baseline) ---
    baseline_mae = None
    metrics_path = Path(artifacts_dir_str) / METRICS_JSON
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            # Accept multiple key variants
            baseline_mae = m.get("test_mae") or m.get("val_mae") or m.get("mae")
        except Exception:
            baseline_mae = None

    label, dot = classify_mae(baseline_mae, mae_ref)
    st.markdown("### Model Health")
    st.write(f"**Baseline MAE:** {baseline_mae:,.0f}" if baseline_mae else "**Baseline MAE:** —")
    st.write(f"**Status:** {dot} {label}")
    st.caption("Status thresholds: OK (≤1.0×), Attention (≤1.2×), Retrain (>1.2× of Reference MAE).")

# --- Color badge under the main title ---
bg = status_color(label)
st.markdown(f"""
<div style='padding:10px;border-radius:8px;background-color:{bg};color:white;text-align:center;font-size:18px;margin-bottom:8px;'>
<b>Model Status:</b> {dot} {label}
</div>
""", unsafe_allow_html=True)

# Prepare paths
artifacts_dir = Path(artifacts_dir_str)
model_path   = artifacts_dir / MODEL_FILENAME
globais_path = artifacts_dir / GLOBAIS_FILENAME
binned_path  = artifacts_dir / BINNED_FILENAME

# Load artifacts
if not model_path.exists():
    st.error(f"Model not found: {model_path.resolve()}")
    st.stop()
if not globais_path.exists():
    st.error(f"Global band file not found: {globais_path.resolve()}")
    st.stop()

pipe, EXPECTED_COLS = load_model_and_schema(model_path, artifacts_dir)
globais, bands = load_bands(globais_path, binned_path)

tabs = st.tabs(["Single property", "Batch (Excel/CSV)"])

# ----------------- TAB 1: SINGLE -----------------
with tabs[0]:
    st.subheader("Single property (form)")

    # Form fields (reasonable defaults for Melbourne)
    col1, col2 = st.columns(2)
    with col1:
        Rooms = st.number_input("Rooms", 1, 10, 3)
        Bedroom2 = st.number_input("Bedroom2", 0, 10, 3)
        Bathroom = st.number_input("Bathroom", 1, 6, 2)
        Car = st.number_input("Car", 0, 6, 1)
        # Distance as integer (no decimals), step=1
        Distance = st.number_input("Distance (km to CBD)", min_value=0, max_value=100, value=8, step=1)
        Month = st.number_input("Month", 1, 12, 8)
    with col2:
        Year = st.number_input("Year", 1900, 2100, 2017)
        Latitude = st.number_input("Latitude", -90.0, 90.0, -37.8136)
        Longitude = st.number_input("Longitude", -180.0, 180.0, 144.9631)
        Propertycount = st.number_input("Propertycount", 0, 200000, 8500)
        Type = st.selectbox("Type", ["H", "U", "T"])
        Method = st.selectbox("Method", ["S", "SP", "VB", "PI", "SA", "SN"])

    asking_price = st.number_input("Asking price (optional)", min_value=0, value=0, step=1000, format="%d")

    if st.button("Estimate"):
        payload = {
            "Rooms": Rooms, "Bedroom2": Bedroom2, "Bathroom": Bathroom, "Car": Car,
            "Distance": Distance, "Latitude": Latitude, "Longitude": Longitude,
            "Propertycount": Propertycount, "Month": Month, "Year": Year,
            "Type": Type, "Method": Method,
        }
        df = pd.DataFrame([payload])
        df, _ = coerce_expected_dataframe(df, EXPECTED_COLS)

        yhat = float(pipe.predict(df)[0])
        lo, hi = interval_from_bands(yhat, bands, globais)

        st.metric("Predicted price (ŷ)", f"{yhat:,.0f}")
        st.write(f"Empirical interval (~80%): **{lo:,.0f}** to **{hi:,.0f}**")

        # Suggested prices
        price_bal  = suggested_price(yhat, mae_ref, "balanced")
        price_fast = suggested_price(yhat, mae_ref, "fast_sale")
        price_max  = suggested_price(yhat, mae_ref, "max_margin")
        st.write("**Suggested prices:**")
        st.write(f"- Balanced: **{price_bal:,.0f}**  |  Fast sale: **{price_fast:,.0f}**  |  Max margin: **{price_max:,.0f}**")

        # Classification vs asking price
        if asking_price and asking_price > 0:
            gap = asking_price - yhat
            gap_pct = gap / yhat if yhat != 0 else np.nan
            cls = classify(asking_price, lo, hi)
            st.write("---")
            st.write(f"Asking price: **{asking_price:,.0f}**")
            st.write(f"Gap: **{gap:,.0f}**  ({gap_pct*100:.1f}%)  →  **{cls}**")

# ----------------- TAB 2: BATCH ------------------
with tabs[1]:
    st.subheader("Batch (upload Excel/CSV)")
    st.caption("Expected columns: " + ", ".join(EXPECTED_COLS) + ". Optional: `AskingPrice` for classification; optional `Price` to compute Live MAE.")
    up = st.file_uploader("Upload .xlsx (first sheet) or .csv", type=["xlsx","csv"])

    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up)
        else:
            xls = pd.ExcelFile(up)
            df_in = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

        df_pred, missing = coerce_expected_dataframe(df_in, EXPECTED_COLS)
        if any(missing):
            st.warning(f"Missing columns were added as NaN (imputed by the model): {missing}")

        # Predict
        yhat = pipe.predict(df_pred)
        results = df_in.copy()
        results["yhat"] = yhat

        # Intervals
        lowers, uppers = [], []
        for v in yhat:
            lo, hi = interval_from_bands(float(v), bands, globais)
            lowers.append(lo); uppers.append(hi)
        results["lower"] = lowers
        results["upper"] = uppers

        # Suggested prices (use current sidebar 'mae_ref')
        results["suggested_balanced"]   = results["yhat"].apply(lambda v: suggested_price(v, mae_ref, "balanced"))
        results["suggested_fast_sale"]  = results["yhat"].apply(lambda v: suggested_price(v, mae_ref, "fast_sale"))
        results["suggested_max_margin"] = results["yhat"].apply(lambda v: suggested_price(v, mae_ref, "max_margin"))

        # Classification (if asking price available)
        candidate_names = {"askingprice", "asking_price", "asking", "price", "preco_anunciado"}
        col_price = next((c for c in results.columns if normalize_name(c) in candidate_names), None)
        if col_price and normalize_name(col_price) != "price":  # avoid clashing with ground-truth "Price"
            results["gap"] = results[col_price] - results["yhat"]
            results["gap_pct"] = results["gap"] / results["yhat"]
            results["classification"] = results.apply(
                lambda r: classify(r[col_price], r["lower"], r["upper"]), axis=1
            )

        # --- Live MAE (Batch) if ground-truth 'Price' available ---
        gt_price_col = next((c for c in results.columns if normalize_name(c) == "price"), None)
        live_mae = None
        if gt_price_col is not None:
            try:
                gt_vals = pd.to_numeric(results[gt_price_col], errors="coerce")
                live_mae = float(np.mean(np.abs(gt_vals - results["yhat"])))
            except Exception:
                live_mae = None

        lbl2, dot2 = classify_mae(live_mae, mae_ref)
        st.markdown("### Batch — Model Health")
        st.write(f"**Live MAE:** {live_mae:,.0f}" if live_mae is not None else "**Live MAE:** —")
        st.write(f"**Status:** {dot2} {lbl2}")

        # Persist history if we have Live MAE
        if live_mae is not None:
            log_mae(live_mae, lbl2, artifacts_dir)

        st.write("Preview:")
        st.dataframe(results.head(20))

        # Download CSV
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions (CSV)",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

st.caption("Note: Model Health uses tighter thresholds (OK ≤1.0×, Attention ≤1.2×, Retrain >1.2× of Reference MAE). Predictions and bands do not change with this status.")
