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
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HSI Layang Dashboard",
    page_icon="🐟",
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

# ── PATH SERVER FIXED ─────────────────────────────────────────────────────────
ROOT = Path("/www/wwwroot/fisheries-server.cloud/Fishcast-Maps")

DATA_FILE = ROOT / "Data" / "HSI_layang_full_grid.csv"
PATH_MODEL = ROOT / "Models" / "rf_layang_model.joblib"

# debug sidebar
st.sidebar.markdown("### Debug Path")
st.sidebar.write("CSV:", DATA_FILE)
st.sidebar.write("MODEL:", PATH_MODEL)

# ── Konstanta ─────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -9.0,  0.0
LON_MIN, LON_MAX = 114.4, 122.7
GRID_RES         = 0.125
CHUNKSIZE        = 1_000_000

PREDICTORS = ["SLA", "EKE", "SST", "CHL", "SSS"]

MONTHS = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]

HSI_CMAP = LinearSegmentedColormap.from_list(
    "hsi",
    ["#1a9850","#a6d96a","#ffffbf","#fdae61","#d73027"]
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🐟 HSI Layang")
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
    df["lat_c"] = LAT_MIN + (df["lat_bin"] + 0.5) * GRID_RES
    df["lon_c"] = LON_MIN + (df["lon_bin"] + 0.5) * GRID_RES
    return df

def add_hsi(df, model):
    X = df[PREDICTORS].copy()
    mask = ~X.isna().any(axis=1)

    df = df.loc[mask].copy()
    X = X.loc[mask]

    if len(df) == 0:
        return df

    df["hsi"] = model.predict_proba(X)[:,1]
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
            ["month","lat_c","lon_c"]
        ).agg(
            hsi_mean=("hsi","mean")
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

# ── PLOT ──────────────────────────────────────────────────────────────────────
def make_grid(df):

    lats = np.sort(df["lat_c"].unique())
    lons = np.sort(df["lon_c"].unique())

    Z = df.pivot(
        index="lat_c",
        columns="lon_c",
        values="hsi_mean"
    ).values

    return lats, lons, Z

# ── plot single ───────────────────────────────────────────────────────────────
def plot_single():

    lats, lons, Z = make_grid(df_m)

    fig, ax = plt.subplots(figsize=(10,6))

    im = ax.pcolormesh(
        lons,
        lats,
        Z,
        cmap=HSI_CMAP,
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(im, ax=ax)

    ax.set_title(MONTHS[sel_month-1])

    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐟 HSI Layang Dashboard")

fig = plot_single()

st.pyplot(fig)