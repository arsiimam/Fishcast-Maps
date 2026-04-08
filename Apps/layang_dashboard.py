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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import streamlit as st

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI PROYEK
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps")

# ── Spesies yang didukung ─────────────────────────────────────────────────────
# Untuk menambah spesies baru, cukup tambahkan entri ke dict ini.
SPECIES_CONFIG: dict[str, dict] = {
    "Layang": {
        "label":      "Ikan Layang",
        "emoji":      "",
        "data_daily":    ROOT / "Data"   / "HSI_layang_daily.csv",
        "data_monthly":  ROOT / "Data"   / "HSI_layang_full_grid.csv",
        "model":         ROOT / "Models" / "rf_layang_model.joblib",
        "predictors":    ["SLA", "EKE", "SST", "CHL", "SSS"],
        "color_accent":  "#1a9850",
    },
    "Kembung": {
        "label":      "Ikan Kembung",
        "emoji":      "",
        "data_daily":    ROOT / "Data"   / "HSI_kembung_daily.csv",
        "data_monthly":  ROOT / "Data"   / "HSI_kembung_full_grid.csv",
        "model":         ROOT / "Models" / "rf_kembung_model.joblib",
        "predictors":    ["SLA", "EKE", "SST", "CHL", "SSS"],
        "color_accent":  "#2166ac",
    },
    "Tuna Albacore": {
        "label":      "Ikan Tuna Albacore",
        "emoji":      "",
        "data_daily":    ROOT / "Data"   / "HSI_albacore_daily.csv",
        "data_monthly":  ROOT / "Data"   / "HSI_albacore_full_grid.csv",
        "model":         ROOT / "Models" / "rf_albacore_model.joblib",
        "predictors":    ["SLA", "EKE", "SST", "CHL", "SSS"],
        "color_accent":  "#d73027",
    },
    "Tuna Skipjack Tuna": {
        "label":      "Ikan Tuna Skipjack Tuna",
        "emoji":      "",
        "data_daily":    ROOT / "Data"   / "HSI_albacore_daily.csv",
        "data_monthly":  ROOT / "Data"   / "HSI_albacore_full_grid.csv",
        "model":         ROOT / "Models" / "rf_albacore_model.joblib",
        "predictors":    ["SLA", "EKE", "SST", "CHL", "SSS"],
        "color_accent":  "#d73027",
    },
}

# ── Bounding box lokasi ───────────────────────────────────────────────────────
# Untuk menambah lokasi baru, tambahkan entri ke dict ini.
LOCATIONS: dict[str, dict] = {
    "DIY (Default)": {
        "lat_min": -9.3, "lat_max": -7.5,
        "lon_min": 109.5, "lon_max": 111.5,
    },
    "Selatan Bantul": {
        "lat_min": -8.5, "lat_max": -7.8,
        "lon_min": 110.0, "lon_max": 110.8,
    },
    "Selatan Gunungkidul": {
        "lat_min": -8.8, "lat_max": -7.9,
        "lon_min": 110.5, "lon_max": 111.2,
    },
    "Kulon Progo": {
        "lat_min": -8.2, "lat_max": -7.7,
        "lon_min": 109.7, "lon_max": 110.3,
    },
    "Seluruh Domain": {
        "lat_min": -9.0, "lat_max": 0.0,
        "lon_min": 114.4, "lon_max": 122.7,
    },
    "Custom": None,   # Bounding box diisi manual oleh user
}

MONTHS = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]

GRID_RES  = 0.125
CHUNKSIZE = 1_000_000

HSI_CMAP = LinearSegmentedColormap.from_list(
    "ZPPI",
    ["#1a9850","#a6d96a","#ffffbf","#fdae61","#d73027"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fishcast – ZPPI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Layout ── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1f0f;
    border-right: 1px solid #1e3d20;
}
[data-testid="stSidebar"] * {
    color: #d4e9d6 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #a3c9a8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    color: #e8f5e9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e3d20 !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f2 100%) !important;
    border: 1px solid #c3e6cb !important;
    border-radius: 8px;
    padding: 14px 18px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] p {
    color: #2d6a35 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #0d3318 !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Section dividers ── */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a8a62;
    margin: 1.4rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 2px solid #c8e6c9;
}

/* ── Info banner ── */
.info-banner {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #0d47a1;
    margin-bottom: 1rem;
}

/* ── Warning banner ── */
.warn-banner {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #7b5800;
    margin-bottom: 1rem;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — PANEL KONTROL
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# Fishcast")
    st.markdown("**Zona Potensi Penangkapan Ikan**")
    st.divider()

    # ── 1. Jenis Ikan ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Jenis Ikan</p>', unsafe_allow_html=True)
    sel_species = st.radio(
        "Jenis Ikan",
        options=list(SPECIES_CONFIG.keys()),
        format_func=lambda k: f"{SPECIES_CONFIG[k]['emoji']}  {SPECIES_CONFIG[k]['label']}",
        label_visibility="collapsed",
    )
    cfg = SPECIES_CONFIG[sel_species]

    # ── 2. Mode Prediksi ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Mode Prediksi</p>', unsafe_allow_html=True)
    sel_mode = st.radio(
        "Mode",
        options=["Bulanan", "Harian"],
        label_visibility="collapsed",
    )

    # ── 3a. Pilih Bulan (hanya jika mode Bulanan) ─────────────────────────────
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
        sel_month = None
        show_all_months = False
        # Mode Harian: pilih tanggal
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
        index=4,                       # Default: DIY
        label_visibility="collapsed",
    )

    # Jika Custom → tampilkan input bounding box
    if sel_location == "Custom":
        st.markdown("**Bounding Box Custom**")
        col_a, col_b = st.columns(2)
        with col_a:
            custom_lat_min = st.number_input("Lat Min", value=-9.3, step=0.1, format="%.2f")
            custom_lon_min = st.number_input("Lon Min", value=109.5, step=0.1, format="%.2f")
        with col_b:
            custom_lat_max = st.number_input("Lat Max", value=-7.5, step=0.1, format="%.2f")
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
    st.caption("Fishcast v2.0 · Fisheries Server")


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
    """Tambah kolom HSI dari model Random Forest."""
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
    ax.set_extent([bbox["lon_min"], bbox["lon_max"],
                   bbox["lat_min"], bbox["lat_max"]],
                  crs=ccrs.PlateCarree())


# ══════════════════════════════════════════════════════════════════════════════
#  FUNGSI LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_climatology(species_key: str) -> pd.DataFrame:
    """
    Load & bangun klimatologi bulanan untuk spesies tertentu.
    Hasil di-cache berdasarkan nama spesies.
    """
    cfg_s    = SPECIES_CONFIG[species_key]
    data_fp  = cfg_s["data_monthly"]
    model_fp = cfg_s["model"]
    preds    = cfg_s["predictors"]

    model = joblib.load(model_fp)
    parts = []

    for chunk in pd.read_csv(data_fp, chunksize=CHUNKSIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])

        # Filter domain global dulu (sebelum bbox lokasi)
        chunk = filter_bbox(chunk, {
            "lat_min": -9.0, "lat_max": 0.0,
            "lon_min": 114.4, "lon_max": 122.7,
        })

        chunk = compute_hsi(chunk, model, preds)
        chunk["month"] = chunk["date"].dt.month
        chunk = add_spatial_bins(chunk, -9.0, 114.4)

        agg = chunk.groupby(["month", "lat_c", "lon_c"]).agg(
            hsi_mean=("hsi", "mean")
        ).reset_index()
        parts.append(agg)

    combined = pd.concat(parts)
    return combined.groupby(["month", "lat_c", "lon_c"]).agg(
        hsi_mean=("hsi_mean", "mean")
    ).reset_index()


@st.cache_data(show_spinner=False)
def load_daily(species_key: str, date_str: str) -> pd.DataFrame:
    """
    Load data harian untuk spesies dan tanggal tertentu.
    Hasil di-cache berdasarkan kombinasi spesies + tanggal.
    """
    cfg_s    = SPECIES_CONFIG[species_key]
    data_fp  = cfg_s["data_daily"]
    model_fp = cfg_s["model"]
    preds    = cfg_s["predictors"]

    model = joblib.load(model_fp)
    parts = []

    for chunk in pd.read_csv(data_fp, chunksize=CHUNKSIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"].dt.date.astype(str) == date_str]
        if chunk.empty:
            continue

        chunk = compute_hsi(chunk, model, preds)
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
        ax.set_ylabel("Latitude", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.025, shrink=0.85)
    cbar.set_label("HSI", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor("#f7fbf7")
    fig.tight_layout()
    return fig


def plot_all_months(grid: pd.DataFrame) -> plt.Figure:
    """Buat grid peta 3×4 untuk 12 bulan sekaligus."""
    norm = Normalize(vmin=vmin, vmax=vmax)

    if HAS_CARTOPY:
        proj  = ccrs.PlateCarree()
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
    fig.suptitle(f"Klimatologi HSI {species_label} – 12 Bulan",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.patch.set_facecolor("#f7fbf7")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def show_metrics(df: pd.DataFrame, extra_label: str = "") -> None:
    """Tampilkan 4 metric cards di atas peta."""
    hsi = df["hsi_mean"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata HSI",  f"{hsi.mean():.3f}")
    c2.metric("HSI Tertinggi",  f"{hsi.max():.3f}")
    c3.metric("HSI Terendah",   f"{hsi.min():.3f}")
    c4.metric("Jumlah Grid",    f"{len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — HEADER & VALIDASI
# ══════════════════════════════════════════════════════════════════════════════

species_emoji = cfg["emoji"]
species_label = cfg["label"]

st.markdown(f"## {species_emoji} Peta Potensi Penangkapan — {species_label}")

# Tampilkan lokasi aktif
loc_info = (
    f"📍 **{sel_location}** | "
    f"Lat {bbox['lat_min']:.1f}°–{bbox['lat_max']:.1f}° | "
    f"Lon {bbox['lon_min']:.1f}°–{bbox['lon_max']:.1f}°"
)
st.markdown(f'<div class="info-banner">{loc_info}</div>', unsafe_allow_html=True)

if not HAS_CARTOPY:
    st.markdown(
        '<div class="warn-banner">⚠️ Cartopy tidak terinstall — peta tampil '
        'tanpa fitur geografis. Install: <code>pip install cartopy</code></div>',
        unsafe_allow_html=True
    )

# ── Cek ketersediaan file ─────────────────────────────────────────────────────
mode_key  = "data_monthly" if sel_mode == "Bulanan" else "data_daily"
data_file = cfg[mode_key]
model_fp  = cfg["model"]

if not data_file.exists():
    st.error(f"❌ File data tidak ditemukan: `{data_file}`")
    st.stop()
if not model_fp.exists():
    st.error(f"❌ Model tidak ditemukan: `{model_fp}`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA + FILTER LOKASI
# ══════════════════════════════════════════════════════════════════════════════

if sel_mode == "Bulanan":
    with st.spinner(f"Memuat klimatologi {species_label}…"):
        grid_full = load_climatology(sel_species)

    # Filter ke bounding box lokasi yang dipilih
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

        title = (
            f"HSI {species_label} – {MONTHS[sel_month - 1]} | {sel_location}"
        )
        with st.spinner(f"Membuat peta {MONTHS[sel_month - 1]}…"):
            fig = plot_single(df_month, title=title)
        st.pyplot(fig, use_container_width=True)

else:  # Harian
    date_str = str(sel_date)

    with st.spinner(f"Memuat data harian {date_str}…"):
        df_daily = load_daily(sel_species, date_str)

    if df_daily.empty:
        st.warning(
            f"Tidak ada data untuk tanggal **{date_str}**. "
            "Coba tanggal lain atau pastikan file data tersedia."
        )
        st.stop()

    # Filter ke bounding box lokasi
    df_daily_bbox = df_daily[
        df_daily["lat_c"].between(bbox["lat_min"], bbox["lat_max"]) &
        df_daily["lon_c"].between(bbox["lon_min"], bbox["lon_max"])
    ]

    if df_daily_bbox.empty:
        st.warning("Tidak ada data di wilayah yang dipilih untuk tanggal ini.")
        st.stop()

    show_metrics(df_daily_bbox)
    st.markdown("---")

    title = f"HSI {species_label} – {date_str} | {sel_location}"
    with st.spinner(f"Membuat peta {date_str}…"):
        fig = plot_single(df_daily_bbox, title=title)
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "**Fishcast** · Sistem Informasi Zona Potensi Penangkapan Ikan · "
    "dashboard.fisheries-server.cloud · v2.0"
)