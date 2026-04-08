# ── Imports ───────────────────────────────────────────────────────────────────
from pathlib import Path
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


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZPPI Kembung Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLE ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }

[data-testid="stMetric"] {
    background: #d4edda !important;
    border-radius: 10px;
    padding: 12px 16px;
}

[data-testid="stMetricLabel"] p {
    color: #1a4a2e !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #0d2b1a !important;
    font-weight: 700 !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── PATH SERVER ───────────────────────────────────────────────────────────────
ROOT = Path("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps")

DATA_FILE  = ROOT / "Data"   / "HSI_layang_full_grid.csv"
PATH_MODEL = ROOT / "Models" / "rf_layang_model.joblib"

# ── Konstanta ─────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -9.0,  0.0
LON_MIN, LON_MAX = 114.4, 122.7
GRID_RES          = 0.125
CHUNKSIZE         = 1_000_000

PREDICTORS = ["SLA", "EKE", "SST", "CHL", "SSS"]

MONTHS = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]

HSI_CMAP = LinearSegmentedColormap.from_list(
    "ZPPI",
    ["#1a9850","#a6d96a","#ffffbf","#fdae61","#d73027"]
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Zona Potensi Penangkapan Ikan Kembung")
st.sidebar.divider()

sel_month = st.sidebar.selectbox(
    "Bulan",
    options=list(range(1, 13)),
    format_func=lambda m: MONTHS[m - 1]
)

show_all = st.sidebar.checkbox("12 bulan sekaligus", value=False)
vmin = st.sidebar.slider("Skala warna min", 0.0, 1.0, 0.0)
vmax = st.sidebar.slider("Skala warna max", 0.0, 1.0, 1.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def filter_domain(df):
    return df.loc[
        df["lat"].between(LAT_MIN, LAT_MAX) &
        df["lon"].between(LON_MIN, LON_MAX)
    ].copy()

def add_spatial_bins(df):
    df["lat_bin"] = np.floor((df["lat"] - LAT_MIN) / GRID_RES).astype(int)
    df["lon_bin"] = np.floor((df["lon"] - LON_MIN) / GRID_RES).astype(int)
    df["lat_c"]   = LAT_MIN + (df["lat_bin"] + 0.5) * GRID_RES
    df["lon_c"]   = LON_MIN + (df["lon_bin"] + 0.5) * GRID_RES
    return df

def add_hsi(df, model):
    X    = df[PREDICTORS].copy()
    mask = ~X.isna().any(axis=1)

    df = df.loc[mask].copy()
    X  = X.loc[mask]

    if len(df) == 0:
        return df

    df["hsi"] = model.predict_proba(X)[:, 1]
    return df

# ── Load model + data ─────────────────────────────────────────────────────────
@st.cache_data
def build_climatology():

    model = joblib.load(PATH_MODEL)
    parts = []

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNKSIZE):

        chunk["date"] = pd.to_datetime(chunk["date"])

        chunk = filter_domain(chunk)
        chunk = add_hsi(chunk, model)

        chunk["month"] = chunk["date"].dt.month
        chunk = add_spatial_bins(chunk)

        agg = chunk.groupby(
            ["month", "lat_c", "lon_c"]
        ).agg(
            hsi_mean=("hsi", "mean")
        ).reset_index()

        parts.append(agg)

    combined = pd.concat(parts)

    # Final aggregation to remove cross-chunk duplicates
    return combined.groupby(
        ["month", "lat_c", "lon_c"]
    ).agg(
        hsi_mean=("hsi_mean", "mean")
    ).reset_index()

# ── Load data ─────────────────────────────────────────────────────────────────
if not DATA_FILE.exists():
    st.error(f"CSV tidak ditemukan: {DATA_FILE}")
    st.stop()

if not PATH_MODEL.exists():
    st.error(f"Model tidak ditemukan: {PATH_MODEL}")
    st.stop()

with st.spinner("Loading data..."):
    grid = build_climatology()

# ── Filter bulan ──────────────────────────────────────────────────────────────
df_m = grid[grid["month"] == sel_month]

# ── PLOT helpers ──────────────────────────────────────────────────────────────
def make_grid(df):
    """Pivot dataframe menjadi 2-D array untuk pcolormesh."""
    Z    = df.pivot(index="lat_c", columns="lon_c", values="hsi_mean")
    lats = Z.index.values
    lons = Z.columns.values
    return lats, lons, Z.values


def _mesh_edges(lats, lons):
    """Hitung edges grid dari centers agar pcolormesh tidak misalign."""
    dlon      = GRID_RES / 2
    dlat      = GRID_RES / 2
    lon_edges = np.concatenate([lons - dlon, [lons[-1] + dlon]])
    lat_edges = np.concatenate([lats - dlat, [lats[-1] + dlat]])
    return lat_edges, lon_edges


def _add_cartopy_features(ax, label_left=True, label_bottom=True,
                           lon_step=2, lat_step=2, lw_coast=0.8):
    """Tambahkan fitur peta & gridlines ke axes Cartopy."""
    ax.add_feature(cfeature.LAND,
                   facecolor="#e8e0d0", zorder=2)
    ax.add_feature(cfeature.OCEAN,
                   facecolor="#cde7f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=lw_coast, edgecolor="#3a3a3a", zorder=3)
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.5, edgecolor="#666666",
                   linestyle="--", zorder=3)
    ax.add_feature(cfeature.RIVERS,
                   linewidth=0.4, edgecolor="#7ab8d4",
                   alpha=0.6, zorder=3)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = label_left
    gl.bottom_labels = label_bottom
    gl.xlocator      = mticker.FixedLocator(
        np.arange(np.floor(LON_MIN), np.ceil(LON_MAX) + 1, lon_step))
    gl.ylocator      = mticker.FixedLocator(
        np.arange(np.floor(LAT_MIN), np.ceil(LAT_MAX) + 1, lat_step))
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 8}
    gl.ylabel_style  = {"size": 8}

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                  crs=ccrs.PlateCarree())


# ── Plot single bulan ─────────────────────────────────────────────────────────
def plot_single(df, title):
    lats, lons, Z        = make_grid(df)
    lat_edges, lon_edges = _mesh_edges(lats, lons)
    norm                 = Normalize(vmin=vmin, vmax=vmax)

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(11, 7),
            subplot_kw={"projection": proj},
        )

        im = ax.pcolormesh(
            lon_edges, lat_edges, Z,
            cmap=HSI_CMAP, norm=norm,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        _add_cartopy_features(ax)

    else:
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.pcolormesh(
            lon_edges, lat_edges, Z,
            cmap=HSI_CMAP, norm=norm,
        )
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.4, alpha=0.5, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.025, shrink=0.85)
    cbar.set_label("HSI", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ── Plot 12 bulan (grid 3×4) ──────────────────────────────────────────────────
def plot_all_months():
    norm = Normalize(vmin=vmin, vmax=vmax)

    if HAS_CARTOPY:
        proj  = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            3, 4,
            figsize=(24, 15),
            subplot_kw={"projection": proj},
        )
    else:
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))

    axes_flat = axes.flatten()

    for idx, month in enumerate(range(1, 13)):
        ax   = axes_flat[idx]
        df_i = grid[grid["month"] == month]

        if df_i.empty:
            ax.set_visible(False)
            continue

        lats, lons, Z        = make_grid(df_i)
        lat_edges, lon_edges = _mesh_edges(lats, lons)

        row = idx // 4
        col = idx  % 4

        if HAS_CARTOPY:
            im = ax.pcolormesh(
                lon_edges, lat_edges, Z,
                cmap=HSI_CMAP, norm=norm,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            _add_cartopy_features(
                ax,
                label_left   = (col == 0),
                label_bottom = (row == 2),
                lon_step=4,
                lat_step=3,
                lw_coast=0.6,
            )
        else:
            im = ax.pcolormesh(
                lon_edges, lat_edges, Z,
                cmap=HSI_CMAP, norm=norm,
            )
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.grid(True, linewidth=0.3, alpha=0.4, linestyle="--")

        ax.set_title(MONTHS[month - 1], fontsize=9, fontweight="bold")

    # Colorbar bersama
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm      = plt.cm.ScalarMappable(cmap=HSI_CMAP, norm=norm)
    sm.set_array([])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("HSI", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Klimatologi HSI Ikan Layang – 12 Bulan",
                 fontsize=15, fontweight="bold", y=0.99)
    return fig


# ── Metrics ───────────────────────────────────────────────────────────────────
def show_metrics(df):
    hsi = df["hsi_mean"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata HSI", f"{hsi.mean():.3f}")
    c2.metric("HSI Maks",      f"{hsi.max():.3f}")
    c3.metric("HSI Min",       f"{hsi.min():.3f}")
    c4.metric("Jumlah Grid",   f"{len(df):,}")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Zona Potensi Penangkapan Ikan Kembung")

if not HAS_CARTOPY:
    st.warning(
        "⚠️ Cartopy tidak terinstall — peta ditampilkan tanpa fitur geografis. "
        "Install dengan: `pip install cartopy`"
    )

if show_all:
    show_metrics(grid)
    with st.spinner("Membuat peta 12 bulan..."):
        fig = plot_all_months()
    st.pyplot(fig)
else:
    show_metrics(df_m)
    with st.spinner(f"Membuat peta {MONTHS[sel_month - 1]}..."):
        fig = plot_single(
            df_m,
            title=f"HSI Ikan Kembung – {MONTHS[sel_month - 1]}"
        )
    st.pyplot(fig)