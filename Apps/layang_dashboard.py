# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FISHCAST — Zona Potensi Penangkapan Ikan                                  ║
# ║  Versi multi-spesies, multi-mode (harian/bulanan), multi-lokasi            ║
# ║  Apps/fishcast_dashboard.py                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Imports ───────────────────────────────────────────────────────────────────
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
import streamlit as st

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

import os
import time

# ── Stub untuk backward-compatibility pickle ──────────────────────────────────
# Jika model lama disimpan dengan kelas custom, stub ini mencegah error saat load.
# Setelah semua model dimigasi ke format bundle dict, stub ini bisa dihapus.
class RFDecisionModel:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI PROYEK
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps")

# ── Spesies yang didukung ─────────────────────────────────────────────────────
# CATATAN: "predictors" TIDAK lagi didefinisikan di sini.
# Predictor dibaca otomatis dari file .joblib masing-masing model.
# Format joblib yang didukung:
#   1. Bundle dict : {"model": <estimator>, "predictors": [...]}  ← DIANJURKAN
#   2. Raw sklearn  : estimator dengan atribut feature_names_in_  ← fallback
#   3. RFDecisionModel atau wrapper lain dengan atribut .model / .estimator
SPECIES_CONFIG: dict[str, dict] = {
    "Kembung": {
        "label":        "Ikan Kembung",
        "data_daily":   ROOT / "Data"   / "HSI_Kembug_Jateng-DIY.csv",
        "data_monthly": ROOT / "Data"   / "HSI_Kembug_Jateng-DIY.csv",
        "model":        ROOT / "Models" / "rf_kembung_model.joblib",
        "color_accent": "#2166ac",
    },
    "Layang": {
        "label":        "Ikan Layang",
        "data_daily":   ROOT / "Data"   / "HSI_layang_daily.csv",
        "data_monthly": ROOT / "Data"   / "HSI_layang_full_grid.csv",
        "model":        ROOT / "Models" / "rf_layang_model.joblib",
        "color_accent": "#1a9850",
    },
    "Tuna Albacore": {
        "label":        "Ikan Tuna Albacore",
        "data_daily":   ROOT / "Data"   / "HSI_albacore_daily.csv",
        "data_monthly": ROOT / "Data"   / "HSI_albacore_full_grid.csv",
        "model":        ROOT / "Models" / "rf_albacore_model.joblib",
        "color_accent": "#d73027",
    },
    "Tuna Skipjack": {
        "label":        "Ikan Tuna Skipjack",
        "data_daily":   ROOT / "Data"   / "HSI_skipjack_daily.csv",
        "data_monthly": ROOT / "Data"   / "HSI_skipjack_full_grid.csv",
        "model":        ROOT / "Models" / "rf_skipjack_model.joblib",
        "color_accent": "#f46d43",
    },
}

# ── Bounding box lokasi ───────────────────────────────────────────────────────
LOCATIONS: dict[str, dict] = {
    "DIY (Default)": {
        "lat_min": -9.3,  "lat_max": -7.5,
        "lon_min": 109.5, "lon_max": 111.5,
    },
    "Selatan Bantul": {
        "lat_min": -8.5,  "lat_max": -7.8,
        "lon_min": 110.0, "lon_max": 110.8,
    },
    "Selatan Gunungkidul": {
        "lat_min": -8.8,  "lat_max": -7.9,
        "lon_min": 110.5, "lon_max": 111.2,
    },
    "Kulon Progo": {
        "lat_min": -8.2,  "lat_max": -7.5,
        "lon_min": 110.0, "lon_max": 110.3,
    },
    "Jateng-DIY": {
        "lat_min": -14.5, "lat_max": -7.5,
        "lon_min": 108.5, "lon_max": 112.5,
    },
    "Custom": None,  # Bounding box diisi manual oleh user
}

MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

GRID_RES  = 0.125
CHUNKSIZE = 1_000_000

HSI_CMAP = LinearSegmentedColormap.from_list(
    "ZPPI",
    ["#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fishcast – ZPPI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #0f1b2d 100%);
    border-right: 1px solid #1f2a3a;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #8aa4c8 !important; font-size: 0.75rem !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span { color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: #1f2a3a !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important; font-weight: 600;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f2 100%) !important;
    border: 1px solid #c3e6cb !important; border-radius: 8px;
    padding: 14px 18px !important; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] p {
    color: #2d6a35 !important; font-weight: 600 !important;
    font-size: 0.78rem !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #0d3318 !important; font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.section-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #5a8a62;
    margin: 1.4rem 0 0.5rem 0; padding-bottom: 4px;
    border-bottom: 2px solid #c8e6c9;
}
.info-banner {
    background: #e8f4fd; border-left: 4px solid #2196F3;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 0.85rem; color: #0d47a1; margin-bottom: 1rem;
}
.warn-banner {
    background: #fff8e1; border-left: 4px solid #ffc107;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 0.85rem; color: #7b5800; margin-bottom: 1rem;
}
.predictor-badge {
    display: inline-block; background: #1f2a3a; color: #8ac8ff !important;
    border: 1px solid #2a3f5a; border-radius: 4px;
    padding: 2px 7px; font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace;
    margin: 2px 2px;
}

[data-testid="stSidebar"] div[data-baseweb="select"],
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important; color: #000000 !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div { color: #000000 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] svg,
[data-testid="stSidebar"] .stSelectbox svg { fill: #000000 !important; }
[data-testid="stSidebar"] .stDateInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox input {
    background-color: #ffffff !important; color: #000000 !important;
}
[data-testid="stSidebar"] input::placeholder { color: #555555 !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #ffffff !important; color: #000000 !important; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUNDLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_bundle(model_fp: str) -> tuple:
    """
    Load model joblib dan ekstrak predictor list-nya.

    Mendukung format:
      1. Bundle dict  : {"model": estimator, "predictors": [...]}   -- DIANJURKAN
      2. Raw sklearn  : estimator dengan feature_names_in_           -- fallback
      3. Wrapper kelas: dicari semua atribut yang punya predict_proba -- legacy fallback

    Returns:
        (model, predictors): tuple estimator sklearn + list nama kolom predictor

    Raises:
        ValueError: jika predictor tidak bisa ditemukan dari format apapun.
    """
    bundle = joblib.load(model_fp)

    # -- Format 1: Bundle dict ------------------------------------------------
    if isinstance(bundle, dict):
        model      = bundle.get("model")
        predictors = bundle.get("predictors")
        if model is None:
            raise ValueError(
                f"Bundle dict di '{model_fp}' tidak memiliki key 'model'.\n"
                f"Key yang ada: {list(bundle.keys())}"
            )
        if predictors is None:
            raise ValueError(
                f"Bundle dict di '{model_fp}' tidak memiliki key 'predictors'.\n"
                f"Key yang ada: {list(bundle.keys())}"
            )
        return model, list(predictors)

    # -- Format 2: Raw sklearn estimator --------------------------------------
    if hasattr(bundle, "predict_proba"):
        if hasattr(bundle, "feature_names_in_"):
            return bundle, list(bundle.feature_names_in_)
        raise ValueError(
            f"Model sklearn di '{model_fp}' tidak menyimpan feature_names_in_.\n"
            "Simpan ulang model sebagai bundle dict:\n"
            "  joblib.dump({'model': model, 'predictors': [...]}, path)"
        )

    # -- Format 3: Wrapper kelas (legacy, misal RFDecisionModel) --------------
    #
    # Strategi resolusi predictor (urutan prioritas):
    #   a) Atribut 'feature_names'    -- RFDecisionModel style
    #   b) Atribut 'predictors'       -- nama alternatif umum
    #   c) feature_names_in_ dari inner estimator -- sklearn style
    #
    # Strategi resolusi estimator:
    #   Scan semua atribut __dict__, ambil yang punya predict_proba

    bundle_attrs = vars(bundle) if hasattr(bundle, "__dict__") else {}

    # Cari predictor list dari atribut wrapper
    predictors = None
    for pred_attr in ("feature_names", "predictors", "feature_list", "features"):
        candidate = bundle_attrs.get(pred_attr)
        if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
            predictors = list(candidate)
            break

    # Cari inner estimator dari atribut wrapper
    inner      = None
    inner_attr = None
    for attr in ("base_model", "model", "estimator", "rf", "classifier",
                 "regressor", "pipeline"):
        candidate = bundle_attrs.get(attr)
        if candidate is not None and hasattr(candidate, "predict_proba"):
            inner      = candidate
            inner_attr = attr
            break

    # Jika belum ketemu, scan seluruh __dict__
    if inner is None:
        for attr, val in bundle_attrs.items():
            if hasattr(val, "predict_proba"):
                inner      = val
                inner_attr = attr
                break

    # Fallback predictor dari feature_names_in_ inner estimator
    if predictors is None and inner is not None:
        if hasattr(inner, "feature_names_in_"):
            predictors = list(inner.feature_names_in_)

    if inner is not None and predictors is not None:
        return inner, predictors

    # -- Tidak bisa dikenali — tampilkan info debug lengkap -------------------
    attrs = list(bundle_attrs.keys()) if bundle_attrs else "N/A"
    raise ValueError(
        f"Format model tidak dikenali di '{model_fp}'.\n"
        f"Tipe objek  : {type(bundle)}\n"
        f"Atribut     : {attrs}\n"
        f"Inner model : {'ditemukan di ' + str(inner_attr) if inner else 'tidak ditemukan'}\n"
        f"Predictor   : {'ditemukan' if predictors else 'tidak ditemukan'}\n"
        "Solusi: simpan ulang sebagai bundle dict:\n"
        "  joblib.dump({'model': <estimator>, 'predictors': [...]}, path)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — PANEL KONTROL
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# Fishcast")
    st.markdown("**Zona Potensi Penangkapan Ikan**")
    st.divider()

    # -- 1. Jenis Ikan ---------------------------------------------------------
    st.markdown('<p class="section-header">Jenis Ikan</p>', unsafe_allow_html=True)
    sel_species = st.radio(
        "Jenis Ikan",
        options=list(SPECIES_CONFIG.keys()),
        format_func=lambda k: SPECIES_CONFIG[k]["label"],
        label_visibility="collapsed",
    )
    cfg = SPECIES_CONFIG[sel_species]

    # ── Info predictor model yang aktif ──────────────────────────────────────
    # Ditampilkan di sidebar agar user tahu predictor apa yang dipakai model ini
    try:
        _model_preview, _preds_preview = load_model_bundle(str(cfg["model"]))
        badges = " ".join(
            f'<span class="predictor-badge">{p}</span>' for p in _preds_preview
        )
        st.markdown(
            f'<div style="margin-top:4px;margin-bottom:8px;">'
            f'<span style="font-size:0.68rem;color:#8aa4c8;">PREDICTOR MODEL:</span><br>{badges}'
            f'</div>',
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown(
            '<div style="font-size:0.68rem;color:#f4a261;">Predictor belum bisa dibaca</div>',
            unsafe_allow_html=True
        )

    # ── 2. Mode Prediksi ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Mode Prediksi</p>', unsafe_allow_html=True)
    sel_mode = st.radio(
        "Mode",
        options=["Harian", "Bulanan"],
        label_visibility="collapsed",
    )

    # ── 3. Pilih Bulan / Tanggal ──────────────────────────────────────────────
    if sel_mode == "Bulanan":
        st.markdown('<p class="section-header">Bulan</p>', unsafe_allow_html=True)
        sel_month = st.selectbox(
            "Bulan",
            options=list(range(1, 13)),
            format_func=lambda m: MONTHS[m - 1],
            label_visibility="collapsed",
        )
        show_all_months = st.checkbox("Tampilkan semua 12 bulan", value=False)
    else:
        sel_month      = None
        show_all_months = False
        st.markdown('<p class="section-header">Tanggal</p>', unsafe_allow_html=True)
        sel_date = st.date_input(
            "Tanggal",
            value=datetime.date.today(),
            label_visibility="collapsed",
        )

    # ── 4. Filter Lokasi ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Fokus Wilayah</p>', unsafe_allow_html=True)
    sel_location = st.selectbox(
        "Lokasi",
        options=list(LOCATIONS.keys()),
        index=4,
        label_visibility="collapsed",
    )

    if sel_location == "Custom":
        st.markdown("**Bounding Box Custom**")
        col_a, col_b = st.columns(2)
        with col_a:
            custom_lat_min = st.number_input("Lat Min", value=-9.3,  step=0.1, format="%.2f")
            custom_lon_min = st.number_input("Lon Min", value=109.5, step=0.1, format="%.2f")
        with col_b:
            custom_lat_max = st.number_input("Lat Max", value=-7.5,  step=0.1, format="%.2f")
            custom_lon_max = st.number_input("Lon Max", value=111.5, step=0.1, format="%.2f")
        bbox = {
            "lat_min": custom_lat_min, "lat_max": custom_lat_max,
            "lon_min": custom_lon_min, "lon_max": custom_lon_max,
        }
    else:
        bbox = LOCATIONS[sel_location]

    # ── 5. Pengaturan Visualisasi ─────────────────────────────────────────────
    st.markdown('<p class="section-header">Tampilan Peta</p>', unsafe_allow_html=True)
    vmin = st.slider("Skala warna min", 0.0, 1.0, 0.0, step=0.05)
    vmax = st.slider("Skala warna max", 0.0, 1.0, 1.0, step=0.05)

    st.divider()
    st.caption("Fishcast v2.1 · Fisheries Server")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNGSI UTILITAS
# ══════════════════════════════════════════════════════════════════════════════

def filter_bbox(df: pd.DataFrame, bb: dict) -> pd.DataFrame:
    """Filter DataFrame ke dalam bounding box lat/lon yang diberikan."""
    return df.loc[
        df["lat"].between(bb["lat_min"], bb["lat_max"]) &
        df["lon"].between(bb["lon_min"], bb["lon_max"])
    ].copy()


def add_spatial_bins(df: pd.DataFrame, lat_min: float, lon_min: float) -> pd.DataFrame:
    """Tambah kolom grid-cell centers untuk agregasi spasial."""
    df["lat_bin"] = np.floor((df["lat"] - lat_min) / GRID_RES).astype(int)
    df["lon_bin"] = np.floor((df["lon"] - lon_min) / GRID_RES).astype(int)
    df["lat_c"]   = lat_min + (df["lat_bin"] + 0.5) * GRID_RES
    df["lon_c"]   = lon_min + (df["lon_bin"] + 0.5) * GRID_RES
    return df


def compute_hsi(df: pd.DataFrame, model, predictors: list[str]) -> pd.DataFrame:
    """
    Hitung HSI menggunakan model dan predictor yang diberikan.
    Predictor list didapat dari load_model_bundle, bukan dari config hardcoded.
    """
    # Validasi: pastikan semua kolom predictor ada di DataFrame
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        available = df.columns.tolist()
        raise KeyError(
            f"Kolom predictor tidak ditemukan di CSV: {missing}\n"
            f"Kolom tersedia di file: {available}\n"
            f"Periksa nama kolom CSV atau update bundle model dengan predictor yang benar."
        )

    X    = df[predictors].copy()
    mask = ~X.isna().any(axis=1)
    df   = df.loc[mask].copy()

    if len(df) == 0:
        return df

    df["hsi"] = model.predict_proba(X.loc[mask])[:, 1]
    return df


def make_grid(df: pd.DataFrame):
    """Pivot DataFrame menjadi 2-D array untuk pcolormesh."""
    Z    = df.pivot(index="lat_c", columns="lon_c", values="hsi_mean")
    lats = Z.index.values
    lons = Z.columns.values
    return lats, lons, Z.values


def mesh_edges(lats, lons):
    """Hitung cell edges dari centers agar pcolormesh tidak misalign."""
    dlon      = GRID_RES / 2
    dlat      = GRID_RES / 2
    lon_edges = np.concatenate([lons - dlon, [lons[-1] + dlon]])
    lat_edges = np.concatenate([lats - dlat, [lats[-1] + dlat]])
    return lat_edges, lon_edges


def add_cartopy_features(ax, label_left=True, label_bottom=True,
                          lon_step=1, lat_step=1, lw_coast=0.8):
    """Tambahkan fitur geografis & gridlines ke axes Cartopy."""
    ax.add_feature(cfeature.LAND,      facecolor="#e8e0d0", zorder=2)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde7f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=lw_coast, edgecolor="#3a3a3a", zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="#666666",
                   linestyle="--", zorder=3)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.4, edgecolor="#7ab8d4",
                   alpha=0.6, zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = label_left
    gl.bottom_labels = label_bottom
    gl.xlocator      = mticker.FixedLocator(
        np.arange(np.floor(bbox["lon_min"]), np.ceil(bbox["lon_max"]) + 1, lon_step))
    gl.ylocator      = mticker.FixedLocator(
        np.arange(np.floor(bbox["lat_min"]), np.ceil(bbox["lat_max"]) + 1, lat_step))
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 8}
    gl.ylabel_style  = {"size": 8}
    ax.set_extent(
        [bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"]],
        crs=ccrs.PlateCarree()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FUNGSI LOAD DATA
#  Semua fungsi load menggunakan load_model_bundle() untuk mendapatkan
#  model + predictor secara dinamis — tidak ada predictor hardcoded.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_climatology(species_key: str) -> pd.DataFrame:
    """
    Load & bangun klimatologi bulanan untuk spesies tertentu.
    Predictor dibaca otomatis dari model bundle.
    """
    cfg_s   = SPECIES_CONFIG[species_key]
    data_fp = cfg_s["data_monthly"]

    # ← Predictor berasal dari model, bukan dari config
    model, predictors = load_model_bundle(str(cfg_s["model"]))

    parts = []
    for chunk in pd.read_csv(data_fp, chunksize=CHUNKSIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])

        chunk = compute_hsi(chunk, model, predictors)
        if chunk.empty or "hsi" not in chunk.columns:
            continue
        chunk["month"] = chunk["date"].dt.month
        chunk = add_spatial_bins(chunk, chunk["lat"].min(), chunk["lon"].min())

        agg = chunk.groupby(["month", "lat_c", "lon_c"]).agg(
            hsi_mean=("hsi", "mean")
        ).reset_index()
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["month", "lat_c", "lon_c", "hsi_mean"])

    combined = pd.concat(parts)
    return combined.groupby(["month", "lat_c", "lon_c"]).agg(
        hsi_mean=("hsi_mean", "mean")
    ).reset_index()


@st.cache_data(show_spinner=False)
def load_daily(species_key: str, date_str: str) -> pd.DataFrame:
    """
    Load data harian untuk spesies dan tanggal tertentu.
    Predictor dibaca otomatis dari model bundle.
    """
    cfg_s   = SPECIES_CONFIG[species_key]
    data_fp = cfg_s["data_daily"]

    # ← Predictor berasal dari model, bukan dari config
    model, predictors = load_model_bundle(str(cfg_s["model"]))

    parts = []
    for chunk in pd.read_csv(data_fp, chunksize=CHUNKSIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"].dt.date.astype(str) == date_str]
        if chunk.empty:
            continue

        chunk = compute_hsi(chunk, model, predictors)
        if chunk.empty or "hsi" not in chunk.columns:
            continue
        chunk = add_spatial_bins(chunk, chunk["lat"].min(), chunk["lon"].min())

        agg = chunk.groupby(["lat_c", "lon_c"]).agg(
            hsi_mean=("hsi", "mean")
        ).reset_index()
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["lat_c", "lon_c", "hsi_mean"])

    return pd.concat(parts).groupby(["lat_c", "lon_c"]).agg(
        hsi_mean=("hsi_mean", "mean")
    ).reset_index()


@st.cache_data(show_spinner=False)
def load_daily_climatology(species_key: str, target_date: datetime.date) -> pd.DataFrame:
    """
    Klimatologi harian berbasis day-of-year.
    Predictor dibaca otomatis dari model bundle.
    """
    cfg_s   = SPECIES_CONFIG[species_key]
    data_fp = cfg_s["data_daily"]

    # ← Predictor berasal dari model, bukan dari config
    model, predictors = load_model_bundle(str(cfg_s["model"]))

    parts      = []
    target_doy = pd.to_datetime(target_date).dayofyear

    for chunk in pd.read_csv(data_fp, chunksize=CHUNKSIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["doy"]  = chunk["date"].dt.dayofyear
        chunk         = chunk[chunk["doy"] == target_doy]

        if chunk.empty:
            continue

        chunk = compute_hsi(chunk, model, predictors)
        if chunk.empty or "hsi" not in chunk.columns:
            continue
        chunk = add_spatial_bins(chunk, chunk["lat"].min(), chunk["lon"].min())

        agg = chunk.groupby(["lat_c", "lon_c"]).agg(
            hsi_mean=("hsi", "mean")
        ).reset_index()
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["lat_c", "lon_c", "hsi_mean"])

    return pd.concat(parts).groupby(["lat_c", "lon_c"]).agg(
        hsi_mean=("hsi_mean", "mean")
    ).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
#  FUNGSI PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_single(df: pd.DataFrame, title: str) -> plt.Figure:
    """Buat peta HSI single-panel."""
    lats, lons, Z        = make_grid(df)
    lat_edges, lon_edges = mesh_edges(lats, lons)
    norm                 = Normalize(vmin=vmin, vmax=vmax)

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={"projection": proj})
        im = ax.pcolormesh(lon_edges, lat_edges, Z,
                           cmap=HSI_CMAP, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=1)
        add_cartopy_features(ax)
    else:
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.pcolormesh(lon_edges, lat_edges, Z, cmap=HSI_CMAP, norm=norm)
        ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
        ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.025, shrink=0.85)
    cbar.set_label("Tidak Potensial  ←────────→  Sangat Potensial", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor("#f7fbf7")
    fig.tight_layout()
    return fig


def plot_all_months(grid: pd.DataFrame) -> plt.Figure:
    """Buat grid peta 3×4 untuk 12 bulan sekaligus."""
    norm = Normalize(vmin=vmin, vmax=vmax)

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(3, 4, figsize=(26, 16),
                                 subplot_kw={"projection": proj})
    else:
        fig, axes = plt.subplots(3, 4, figsize=(26, 16))

    axes_flat = axes.flatten()

    for idx, month in enumerate(range(1, 13)):
        ax   = axes_flat[idx]
        df_i = grid[grid["month"] == month]

        if df_i.empty:
            ax.set_visible(False)
            continue

        lats, lons, Z        = make_grid(df_i)
        lat_edges, lon_edges = mesh_edges(lats, lons)
        row, col             = idx // 4, idx % 4

        if HAS_CARTOPY:
            im = ax.pcolormesh(lon_edges, lat_edges, Z,
                               cmap=HSI_CMAP, norm=norm,
                               transform=ccrs.PlateCarree(), zorder=1)
            add_cartopy_features(ax,
                                 label_left=(col == 0), label_bottom=(row == 2),
                                 lon_step=1, lat_step=1, lw_coast=0.5)
        else:
            im = ax.pcolormesh(lon_edges, lat_edges, Z, cmap=HSI_CMAP, norm=norm)
            ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
            ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
            ax.grid(True, linewidth=0.3, alpha=0.4, linestyle="--")

        ax.set_title(MONTHS[month - 1], fontsize=9, fontweight="bold")

    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm      = plt.cm.ScalarMappable(cmap=HSI_CMAP, norm=norm)
    sm.set_array([])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("HSI", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    species_label = SPECIES_CONFIG[sel_species]["label"]
    fig.suptitle(f"Klimatologi ZPPI {species_label} – 12 Bulan",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.patch.set_facecolor("#f7fbf7")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def show_metrics(df: pd.DataFrame) -> None:
    """Tampilkan 4 metric cards di atas peta."""
    hsi = df["hsi_mean"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata HSI", f"{hsi.mean():.3f}")
    c2.metric("HSI Tertinggi", f"{hsi.max():.3f}")
    c3.metric("HSI Terendah",  f"{hsi.min():.3f}")
    c4.metric("Jumlah Grid",   f"{len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — HEADER & VALIDASI
# ══════════════════════════════════════════════════════════════════════════════

species_label = cfg["label"]

st.markdown(f"## Peta Potensi Penangkapan Ikan — {species_label}")

loc_info = (
    f"{sel_location} | "
    f"Lat {bbox['lat_min']:.1f} - {bbox['lat_max']:.1f} | "
    f"Lon {bbox['lon_min']:.1f} - {bbox['lon_max']:.1f}"
)
st.markdown(f'<div class="info-banner">{loc_info}</div>', unsafe_allow_html=True)

if not HAS_CARTOPY:
    st.markdown(
        '<div class="warn-banner">Cartopy tidak terinstall — peta tampil '
        'tanpa fitur geografis. Install: <code>pip install cartopy</code></div>',
        unsafe_allow_html=True
    )

# ── Cek ketersediaan file ─────────────────────────────────────────────────────
mode_key  = "data_monthly" if sel_mode == "Bulanan" else "data_daily"
data_file = cfg[mode_key]
model_path = cfg["model"]

if not data_file.exists():
    st.error(f"File data tidak ditemukan: `{data_file}`")
    st.stop()
if not model_path.exists():
    st.error(f"Model tidak ditemukan: `{model_path}`")
    st.stop()

# ── Validasi model bisa di-load & predictor bisa dibaca ──────────────────────
try:
    _model_check, _preds_check = load_model_bundle(str(model_path))
    st.markdown(
        f'<div class="info-banner">Model aktif menggunakan '
        f'<b>{len(_preds_check)} predictor</b>: '
        f'{", ".join(_preds_check)}</div>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Gagal membaca model `{model_path.name}`:\n\n{e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA + FILTER LOKASI
# ══════════════════════════════════════════════════════════════════════════════

if sel_mode == "Bulanan":
    with st.spinner(f"Memuat klimatologi {species_label}…"):
        grid_full = load_climatology(sel_species)

    grid_bbox = grid_full[
        grid_full["lat_c"].between(bbox["lat_min"], bbox["lat_max"]) &
        grid_full["lon_c"].between(bbox["lon_min"], bbox["lon_max"])
    ]

    if show_all_months:
        show_metrics(grid_bbox)
        st.markdown("---")
        with st.spinner("Membuat peta 12 bulan…"):
            fig = plot_all_months(grid_bbox)
        st.pyplot(fig, use_container_width=True)

    else:
        df_month = grid_bbox[grid_bbox["month"] == sel_month]

        if df_month.empty:
            st.warning(f"Tidak ada data untuk {MONTHS[sel_month - 1]} di wilayah ini.")
            st.stop()

        show_metrics(df_month)
        st.markdown("---")

        title = f"Zona Penangkapan {species_label} – {MONTHS[sel_month - 1]} | {sel_location}"
        with st.spinner(f"Membuat peta {MONTHS[sel_month - 1]}…"):
            fig = plot_single(df_month, title=title)
        st.pyplot(fig, use_container_width=True)

else:  # Harian
    date_str = str(sel_date)

    with st.spinner(f"Memuat data harian {date_str}…"):
        df_daily = load_daily_climatology(sel_species, date_str)

    if df_daily.empty:
        st.warning(
            f"Tidak ada data untuk tanggal **{date_str}**. "
            "Coba tanggal lain atau pastikan file data tersedia."
        )
        st.stop()

    df_daily_bbox = df_daily[
        df_daily["lat_c"].between(bbox["lat_min"], bbox["lat_max"]) &
        df_daily["lon_c"].between(bbox["lon_min"], bbox["lon_max"])
    ]

    if df_daily_bbox.empty:
        st.warning("Tidak ada data di wilayah yang dipilih untuk tanggal ini.")
        st.stop()

    show_metrics(df_daily_bbox)
    st.markdown("---")

    title = f"Zona Penangkapan {species_label} – {date_str} | {sel_location}"
    with st.spinner(f"Membuat peta {date_str}…"):
        fig = plot_single(df_daily_bbox, title=title)
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "**Fishcast** · Sistem Informasi Zona Potensi Penangkapan Ikan · "
    "dashboard.fisheries-server.cloud · v2.1"
)