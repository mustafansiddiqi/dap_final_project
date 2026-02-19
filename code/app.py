from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import altair as alt
import streamlit as st

import ee
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent  # app.py in /code
DATA_DIR = REPO_ROOT / "data"

PANEL_PATH = DATA_DIR / "lahore_monthly_panel.csv"
PAQI_PATH = DATA_DIR / "PAQI_lahore_hourly_pm25_2019_2024.csv"

# Yearly gasoline table (wide format: year columns)
GASOLINE_PATH = DATA_DIR / "energy_institute_table.csv"

# Lahore bbox (match preprocessing.py)
LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]  # [xmin, ymin, xmax, ymax]

START = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)

# -----------------------------
# EE setup
# -----------------------------
@st.cache_resource
def ee_setup():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    return ee.Geometry.Rectangle(LAHORE_BBOX)

def month_list(start=START, end_excl=END_EXCL):
    out = []
    cur = date(start.year, start.month, 1)
    while cur < end_excl:
        out.append(cur)
        cur = (cur + relativedelta(months=1)).replace(day=1)
    return out

MONTHS = month_list()
MONTH_LABELS = [d.strftime("%b %Y") for d in MONTHS]

def month_range(d: date):
    s = ee.Date(d.isoformat())
    e = s.advance(1, "month")
    return s, e

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2024-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data
def load_paqi_station_monthly(path: Path) -> pd.DataFrame:
    """
    Hourly -> station-level monthly mean PM2.5.
    Output: date, station_name, latitude, longitude, pm25_mean
    """
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "pm25_ugm3", "latitude", "longitude"])
    df = df[(df["timestamp_utc"] >= "2019-01-01") & (df["timestamp_utc"] < "2024-01-01")].copy()
    df["date"] = df["timestamp_utc"].dt.to_period("M").dt.to_timestamp()

    out = (
        df.groupby(["date", "station_name", "latitude", "longitude"], as_index=False)
          .agg(pm25_mean=("pm25_ugm3", "mean"))
    )
    return out.sort_values(["date", "station_name"]).reset_index(drop=True)

@st.cache_data
def load_gasoline_yearly(path: Path) -> pd.DataFrame | None:
    """
    Reads energy_institute_table.csv (wide year columns) and returns:
      year, gasoline_usage
    """
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "Units" not in df.columns:
        return None

    gas_rows = df[df["Units"].astype(str).str.contains("Gasoline", case=False, na=False)]
    if gas_rows.empty:
        return None

    gas_row = gas_rows.iloc[0]
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    rows = []
    for y in years:
        col = str(y)
        if col not in df.columns:
            continue
        val = gas_row[col]

        # handle "410.01K" / "1.2M" style if it appears
        if isinstance(val, str):
            v = val.strip()
            mult = 1.0
            if v.endswith("K"):
                mult = 1_000.0
                v = v[:-1]
            elif v.endswith("M"):
                mult = 1_000_000.0
                v = v[:-1]
            val = float(v) * mult

        rows.append({"year": y, "gasoline_usage": float(val)})

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

# -----------------------------
# EE layers
# -----------------------------
def add_ee_tile_layer(fmap: folium.Map, ee_image: ee.Image, vis_params: dict, name: str, opacity: float = 0.8):
    map_id_dict = ee_image.getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        opacity=opacity,
    ).add_to(fmap)

def ee_nightlights(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    img = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
           .filterDate(s, e)
           .select("avg_rad")
           .mean()
           .clip(aoi))
    return img, {"min": 0, "max": 60}, "Nightlights (VIIRS)"

def ee_ndvi(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    img = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(s, e)
           .select("NDVI")
           .mean()
           .multiply(0.0001)
           .clip(aoi))
    return img, {"min": 0.0, "max": 0.8}, "NDVI (Greenness)"

def ee_wind_speed(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    uv = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
          .filterDate(s, e)
          .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
          .mean())
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed").clip(aoi)
    return speed, {"min": 0, "max": 8}, "Wind speed (10m)"

def ee_wind_dir(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    uv = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
          .filterDate(s, e)
          .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
          .mean())
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    dir_deg = u.atan2(v).multiply(180 / 3.141592653589793).add(360).mod(360).rename("wind_dir").clip(aoi)
    return dir_deg, {"min": 0, "max": 360}, "Wind direction (deg)"

def ee_fires(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    fires = ee.ImageCollection("FIRMS").filterDate(s, e)
    fire_sum = fires.select("T21").sum().clip(aoi)
    return fire_sum, {"min": 0, "max": 50}, "Fires (FIRMS T21 monthly sum)"

# -----------------------------
# Altair charts
# -----------------------------
def normalize_to_2019_index(df: pd.DataFrame, col: str) -> pd.Series:
    base = df[df["date"].dt.year == 2019][col].mean()
    if pd.isna(base) or base == 0:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return (df[col] / base) * 100.0

def comparison_chart(df: pd.DataFrame, driver_col: str, driver_label: str):
    tmp = df[["date", "pm25_mean", driver_col]].dropna().copy()
    tmp["pm25_index"] = normalize_to_2019_index(tmp, "pm25_mean")
    tmp["driver_index"] = normalize_to_2019_index(tmp, driver_col)

    long_df = tmp.melt(
        id_vars="date",
        value_vars=["pm25_index", "driver_index"],
        var_name="series",
        value_name="index_2019_100"
    )

    series_map = {
        "pm25_index": "PM2.5 (index, 2019=100)",
        "driver_index": f"{driver_label} (index, 2019=100)"
    }
    long_df["series"] = long_df["series"].map(series_map)

    return (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y("index_2019_100:Q", title="Index (2019=100)"),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("date:T", title="Month"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("index_2019_100:Q", title="Index", format=".1f")
            ],
        )
        .properties(height=380, title=f"PM2.5 vs {driver_label} (indexed to 2019=100)")
    )

def annual_pm25(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["year"] = out["date"].dt.year
    return out.groupby("year", as_index=False).agg(pm25_mean_annual=("pm25_mean", "mean"))

# ---- AQI conversion (US EPA PM2.5 breakpoints) ----
def pm25_to_aqi(pm: float) -> float:
    """
    Converts PM2.5 (µg/m³) to AQI using US EPA breakpoints.
    This is an approximation commonly used for AQI reporting.
    """
    if pm is None or pd.isna(pm):
        return float("nan")

    bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    # clamp above max breakpoint
    if pm > 500.4:
        pm = 500.4

    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= pm <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (pm - c_lo) + i_lo

    # if pm < 0
    return 0.0

@st.cache_data
def annual_aqi_from_paqi_hourly(path: Path) -> pd.DataFrame | None:
    """
    Uses PAQI hourly PM2.5 -> monthly station means -> city monthly mean -> annual mean AQI.
    Returns: year, aqi_mean_annual
    """
    if not path.exists():
        return None

    station_monthly = load_paqi_station_monthly(path)

    # city-wide monthly mean (average across stations)
    city_monthly = (
        station_monthly.groupby("date", as_index=False)
        .agg(pm25_city_mean=("pm25_mean", "mean"))
    )
    city_monthly["aqi_city"] = city_monthly["pm25_city_mean"].apply(pm25_to_aqi)
    city_monthly["year"] = city_monthly["date"].dt.year

    annual = city_monthly.groupby("year", as_index=False).agg(aqi_mean_annual=("aqi_city", "mean"))
    annual = annual[(annual["year"] >= 2019) & (annual["year"] <= 2024)].copy()
    return annual.sort_values("year").reset_index(drop=True)

def gasoline_bar_vs_aqi_line(panel: pd.DataFrame, gas_yearly: pd.DataFrame, annual_aqi: pd.DataFrame | None):
    """
    Bars: gasoline_usage (as provided)
    Line: annual AQI (preferred) or annual PM2.5 fallback
    """
    base = gas_yearly.copy()

    if annual_aqi is not None and not annual_aqi.empty:
        df = base.merge(annual_aqi, on="year", how="inner")
        line_field = "aqi_mean_annual"
        line_title = "Avg AQI (annual)"
        line_fmt = ".1f"
    else:
        pm25_y = annual_pm25(panel)
        df = base.merge(pm25_y, on="year", how="inner")
        line_field = "pm25_mean_annual"
        line_title = "Avg PM2.5 (annual, µg/m³)"
        line_fmt = ".2f"

    bars = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("gasoline_usage:Q", title="Gasoline consumption (kb/d)"),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("gasoline_usage:Q", title="Gasoline (kb/d)", format=",.1f"),
            ],
        )
    )



    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(f"{line_field}:Q", title=line_title),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip(f"{line_field}:Q", title=line_title, format=line_fmt),
            ],
        )
    )

    layered = (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(height=420, title="Gasoline consumption (bar) vs Lahore air quality (line), 2019–2024")
    )
    return layered

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Lahore Air Quality Decomposition", layout="wide")
st.title("Lahore Air Quality Decomposition (2019–2024)")

if not PANEL_PATH.exists():
    st.error(f"Missing {PANEL_PATH}. Run preprocessing.py first to generate it.")
    st.stop()

panel = load_panel(PANEL_PATH)

static_tab, interactive_tab = st.tabs(["Static visuals (Altair)", "Interactive maps"])

# -----------------------------
# TAB 1: Static
# -----------------------------
with static_tab:
    view_static = st.radio(
        "Choose a static view",
        [
            "PM2.5 vs drivers (monthly, indexed to 2019=100)",
            "Gasoline vs air quality (annual, 2019–2024)",
        ],
        horizontal=True,
    )

    st.divider()

    if view_static == "PM2.5 vs drivers (monthly, indexed to 2019=100)":
        st.subheader("Monthly comparisons (PM2.5 overlaid with each driver; indexed to 2019=100)")

        drivers = [
            ("nightlights_avg_rad_mean", "Nightlights (VIIRS)"),
            ("ndvi_mean", "Greenness (NDVI)"),
            ("wind_speed_mean", "Wind speed"),
            ("wind_dir_deg_mean", "Wind direction"),
            ("fire_detections_count", "Fire intensity proxy (FIRMS T21 sum)"),
        ]

        available = [(c, label) for (c, label) in drivers if c in panel.columns]
        if not available:
            st.error("No driver columns found in lahore_monthly_panel.csv.")
            st.stop()

        driver_choice = st.selectbox(
            "Choose driver to compare against PM2.5",
            options=available,
            format_func=lambda x: x[1],
        )
        driver_col, driver_label = driver_choice

        st.altair_chart(comparison_chart(panel, driver_col, driver_label), use_container_width=True)
        st.caption("Indexing uses the 2019 mean as 100 to compare series on the same scale.")

    else:
        st.subheader("Annual gasoline consumption (bar) vs Lahore air quality (line)")

        gas_yearly = load_gasoline_yearly(GASOLINE_PATH)
        if gas_yearly is None:
            st.error(
                f"Could not load gasoline from {GASOLINE_PATH}. "
                "Check that it has a 'Units' column and a row containing 'Gasoline' plus year columns 2019–2024."
            )
            st.stop()

        annual_aqi = annual_aqi_from_paqi_hourly(PAQI_PATH) if PAQI_PATH.exists() else None
        st.altair_chart(gasoline_bar_vs_aqi_line(panel, gas_yearly, annual_aqi), use_container_width=True)

        if annual_aqi is not None:
            st.caption("AQI is computed from PAQI PM2.5 using US EPA PM2.5 AQI breakpoints, then averaged annually.")
        else:
            st.caption("PAQI file not found; line shows annual mean PM2.5 instead of AQI.")

# -----------------------------
# TAB 2: Interactive
# -----------------------------
with interactive_tab:
    view_interactive = st.radio(
        "Choose an interactive map",
        [
            "Earth Engine layer (month slider)",
            "PAQI stations (month slider)",
        ],
        horizontal=True,
    )

    st.divider()

    if view_interactive == "Earth Engine layer (month slider)":
        st.subheader("Interactive Earth Engine layer (month slider updates imagery)")

        aoi = ee_setup()
        layer_name = st.selectbox("Layer", ["Nightlights", "NDVI", "Wind speed", "Wind direction", "Fires (FIRMS)"])

        selected_label = st.select_slider("Month", options=MONTH_LABELS, value=MONTH_LABELS[-1])
        month_date = MONTHS[MONTH_LABELS.index(selected_label)]

        opacity = st.slider("Layer opacity", 0.0, 1.0, 0.8, 0.05)
        fmap = folium.Map(location=[31.52, 74.35], zoom_start=10, tiles="cartodbpositron")

        if layer_name == "Nightlights":
            img, vis, name = ee_nightlights(month_date, aoi)
        elif layer_name == "NDVI":
            img, vis, name = ee_ndvi(month_date, aoi)
        elif layer_name == "Wind speed":
            img, vis, name = ee_wind_speed(month_date, aoi)
        elif layer_name == "Wind direction":
            img, vis, name = ee_wind_dir(month_date, aoi)
        else:
            img, vis, name = ee_fires(month_date, aoi)

        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

        xmin, ymin, xmax, ymax = LAHORE_BBOX
        folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)

        st_folium(
            fmap,
            width=1100,
            height=650,
            key=f"ee_map_{layer_name}_{selected_label}_{opacity}",
        )
        st.caption(f"Selected month: {selected_label}")

    else:
        st.subheader("PAQI station map (monthly mean PM2.5 by station)")

        if not PAQI_PATH.exists():
            st.error(f"Missing {PAQI_PATH}")
            st.stop()

        paqi = load_paqi_station_monthly(PAQI_PATH)

        months = sorted(paqi["date"].unique())
        month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in months]

        selected_label = st.select_slider("Month", options=month_labels, value=month_labels[-1])
        selected_month = months[month_labels.index(selected_label)]

        month_df = paqi[paqi["date"] == selected_month].copy()
        if month_df.empty:
            st.warning("No data for selected month.")
            st.stop()

        vmin = float(month_df["pm25_mean"].min())
        vmax = float(month_df["pm25_mean"].max())
        if vmin == vmax:
            vmax = vmin + 1.0

        colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
        colormap.caption = "Monthly mean PM2.5 (µg/m³)"

        fmap = folium.Map(location=[31.52, 74.35], zoom_start=11, tiles="cartodbpositron")

        for _, r in month_df.iterrows():
            color = colormap(float(r["pm25_mean"]))
            folium.CircleMarker(
                location=[float(r["latitude"]), float(r["longitude"])],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=f"{r['station_name']}<br>PM2.5: {r['pm25_mean']:.1f}",
            ).add_to(fmap)

        colormap.add_to(fmap)

        xmin, ymin, xmax, ymax = LAHORE_BBOX
        folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)

        st_folium(
            fmap,
            width=1100,
            height=650,
            key=f"paqi_map_{selected_label}",
        )

        st.caption(f"Selected month: {selected_label}  |  Range: {vmin:.1f}–{vmax:.1f} µg/m³")
