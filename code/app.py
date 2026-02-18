import os
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

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"

PANEL_PATH = DATA_DIR / "lahore_monthly_panel.csv"
PAQI_PATH = DATA_DIR / "PAQI_lahore_hourly_pm25_2019_2024.csv"

GASOLINE_PATH_CANDIDATES = [DATA_DIR / "energy_institute_table.csv"]

LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]

START = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)

# EE setup

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


def month_range(d: date):
    s = ee.Date(d.isoformat())
    e = s.advance(1, "month")
    return s, e



# Data loading

@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2024-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_gasoline_series() -> pd.DataFrame | None:
    """
    Tries to load a gasoline usage series from one of the candidate files.
    Returns monthly dataframe with columns: date, gasoline_usage
    If none found, returns None.
    """
    found = None
    for p in GASOLINE_PATH_CANDIDATES:
        if p.exists():
            found = p
            break
    if found is None:
        return None

    df = pd.read_csv(found)

    # Try common date columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        # Try to find a gasoline column
        gas_col = next((c for c in df.columns if "gas" in c.lower() or "petrol" in c.lower()), None)
        if gas_col is None:
            return None
        out = df[["date", gas_col]].rename(columns={gas_col: "gasoline_usage"}).copy()
        out = out[(out["date"] >= "2019-01-01") & (out["date"] < "2024-01-01")]
        # If daily/weekly, aggregate to monthly
        out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
        out = out.groupby("date", as_index=False)["gasoline_usage"].mean()
        return out.sort_values("date").reset_index(drop=True)

    # Annual pattern
    if "year" in df.columns:
        gas_col = next((c for c in df.columns if "gas" in c.lower() or "petrol" in c.lower()), None)
        if gas_col is None:
            return None
        tmp = df[["year", gas_col]].rename(columns={gas_col: "gasoline_usage"}).copy()
        tmp = tmp[(tmp["year"] >= 2019) & (tmp["year"] <= 2024)]
        # Expand annual -> monthly by repeating
        rows = []
        for _, r in tmp.iterrows():
            y = int(r["year"])
            for m in range(1, 13):
                rows.append({"date": pd.Timestamp(f"{y}-{m:02d}-01"), "gasoline_usage": r["gasoline_usage"]})
        out = pd.DataFrame(rows)
        return out.sort_values("date").reset_index(drop=True)

    return None


@st.cache_data
def load_paqi_station_monthly(path: Path) -> pd.DataFrame:
    """
    Converts the hourly PAQI file into station-level monthly mean PM2.5.
    Output columns: date (month start), station_name, latitude, longitude, pm25_mean
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



# EE layer builders

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
    # Matches your preprocessing approach (FIRMS + T21 monthly sum)
    s, e = month_range(month_date)
    fires = ee.ImageCollection("FIRMS").filterDate(s, e)
    fire_sum = fires.select("T21").sum().clip(aoi)
    return fire_sum, {"min": 0, "max": 50}, "Fires (FIRMS T21 monthly sum)"



# Altair: comparison chart

def normalize_to_2019_index(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert a series to an index where the mean over 2019 months = 100.
    """
    base = df[df["date"].dt.year == 2019][col].mean()
    if pd.isna(base) or base == 0:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return (df[col] / base) * 100.0


def comparison_chart(df: pd.DataFrame, driver_col: str, driver_label: str):
    """
    Returns a single Altair chart comparing PM2.5 vs driver metric using normalized index.
    """
    tmp = df[["date", "pm25_mean", driver_col]].dropna().copy()
    tmp["pm25_index"] = normalize_to_2019_index(tmp, "pm25_mean")
    tmp["driver_index"] = normalize_to_2019_index(tmp, driver_col)

    long_df = tmp.melt(id_vars="date", value_vars=["pm25_index", "driver_index"],
                       var_name="series", value_name="index_2019_100")

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
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"), alt.Tooltip("index_2019_100:Q", format=".1f")],
        )
        .properties(height=380, title=f"PM2.5 vs {driver_label} (indexed to 2019=100)")
    )



# UI

st.set_page_config(page_title="Lahore Air Quality Decomposition", layout="wide")
st.title("Lahore Air Quality Decomposition (2019–2024)")

if not PANEL_PATH.exists():
    st.error(f"Missing {PANEL_PATH}. Your preprocessing wrote to an absolute path; ensure this file exists in /data.")
    st.stop()

panel = load_panel(PANEL_PATH)

# Optional gasoline merge
gas_df = load_gasoline_series()
if gas_df is not None:
    panel = panel.merge(gas_df, on="date", how="left")

view = st.selectbox(
    "Select what to display",
    [
        "Static comparisons (Altair): PM2.5 vs drivers",
        "Interactive EE map: choose layer + month slider",
        "Interactive PAQI station map: PM2.5 by station + month slider",
    ],
)

st.divider()


# View 1: Static comparisons

if view == "Static comparisons (Altair): PM2.5 vs drivers":
    st.subheader("Static comparisons (each chart overlays PM2.5 with one driver)")

    drivers = [
        ("nightlights_avg_rad_mean", "Nightlights (VIIRS)"),
        ("ndvi_mean", "Greenness (NDVI)"),
        ("wind_speed_mean", "Wind speed"),
        ("wind_dir_deg_mean", "Wind direction"),
        ("fire_detections_count", "Fire intensity proxy (FIRMS T21 sum)"),
    ]
    if "gasoline_usage" in panel.columns:
        drivers.append(("gasoline_usage", "Gasoline usage"))

    available = [(c, label) for (c, label) in drivers if c in panel.columns]
    if not available:
        st.error("No driver columns found in lahore_monthly_panel.csv.")
        st.stop()

    driver_choice = st.selectbox("Choose driver to compare against PM2.5", options=available, format_func=lambda x: x[1])
    driver_col, driver_label = driver_choice

    st.altair_chart(comparison_chart(panel, driver_col, driver_label), use_container_width=True)

    st.caption("These comparisons use an index (2019 mean = 100) so PM2.5 and the driver can be read on the same scale.")


# View 2: Interactive EE maps

elif view == "Interactive EE map":
    st.subheader("Interactive Earth Engine layer")

    aoi = ee_setup()

    layer_name = st.selectbox(
        "Layer",
        ["Nightlights", "NDVI", "Wind speed", "Wind direction", "Fires (FIRMS)"]
    )

    month_idx = st.slider("Month", 0, len(MONTHS) - 1, len(MONTHS) - 1)
    month_date = MONTHS[month_idx]
    
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

    # AOI rectangle outline
    xmin, ymin, xmax, ymax = LAHORE_BBOX
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    # IMPORTANT: unique key so slider actually forces a re-render of the widget + tiles
    st_folium(
        fmap,
        width=1100,
        height=650,
        key=f"ee_map_{layer_name}_{month_idx}_{opacity}",
    )

    st.caption(f"Selected month: {month_date.strftime('%Y-%m')}")


# View 3: PAQI station map

else:
    st.subheader("PAQI station map (monthly mean PM2.5 by station)")

    if not PAQI_PATH.exists():
        st.error(f"Missing {PAQI_PATH}")
        st.stop()

    st.write("This map aggregates hourly PAQI readings to **monthly mean PM2.5** for each station, then maps station intensity.")

    paqi = load_paqi_station_monthly(PAQI_PATH)

    months = sorted(paqi["date"].unique())
    month_idx = st.slider("Month", 0, len(months) - 1, len(months) - 1)
    selected_month = months[month_idx]

    month_df = paqi[paqi["date"] == selected_month].copy()
    if month_df.empty:
        st.warning("No data for selected month.")
        st.stop()

    # Color scale
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

    # outline
    xmin, ymin, xmax, ymax = LAHORE_BBOX
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)

    st_folium(
        fmap,
        width=1100,
        height=650,
        key=f"paqi_map_{month_idx}",
    )

    st.caption(f"Selected month: {pd.Timestamp(selected_month).strftime('%Y-%m')}  |  Range: {vmin:.1f}–{vmax:.1f} µg/m³")
