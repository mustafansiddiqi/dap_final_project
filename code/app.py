import os
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import altair as alt
import streamlit as st

import ee
import folium
from streamlit_folium import st_folium
from pathlib import Path

# Config

st.set_page_config(page_title="Lahore Air Quality Decomposition", layout="wide")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent          # if app.py is in /code
DATA_DIR = REPO_ROOT / "data"
PANEL_PATH = DATA_DIR / "lahore_monthly_panel.csv"

# Lahore bbox placeholder (should match preprocessing.py)
LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]

START = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)


# Helpers

@st.cache_resource
def ee_setup():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # Create AOI only after initialization
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


@st.cache_data
def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2024-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)


def kpi_row(df: pd.DataFrame, dt: pd.Timestamp):
    row = df.loc[df["date"] == dt]
    return None if row.empty else row.iloc[0].to_dict()


def add_ee_tile_layer(
    fmap: folium.Map,
    ee_image: ee.Image,
    vis_params: dict,
    name: str,
    opacity: float = 0.8,
):
    map_id_dict = ee_image.getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        opacity=opacity,
    ).add_to(fmap)

# EE layer builders (AOI passed in)

def ee_nightlights(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    img = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(s, e)
        .select("avg_rad")
        .mean()
        .clip(aoi)
    )
    vis = {"min": 0, "max": 60}
    return img, vis, "Nightlights (VIIRS)"


def ee_ndvi(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    img = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(s, e)
        .select("NDVI")
        .mean()
        .multiply(0.0001)
        .clip(aoi)
    )
    vis = {"min": 0.0, "max": 0.8}
    return img, vis, "NDVI (Greenness)"


def ee_wind_speed(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    uv = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(s, e)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
    )
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed").clip(aoi)
    vis = {"min": 0, "max": 8}
    return speed, vis, "Wind speed (10m)"


def ee_wind_dir(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    uv = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(s, e)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
    )
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    dir_deg = (
        u.atan2(v)
        .multiply(180 / 3.141592653589793)
        .add(360)
        .mod(360)
        .rename("wind_dir")
        .clip(aoi)
    )
    vis = {"min": 0, "max": 360}
    return dir_deg, vis, "Wind direction (deg)"


def ee_fires(month_date: date, aoi: ee.Geometry):
    s, e = month_range(month_date)
    fires = ee.ImageCollection("FIRMS").filterDate(s, e)
    fire_sum = fires.select("T21").sum().clip(aoi)
    vis = {"min": 0, "max": 50}  # adjust if washed out/saturated
    return fire_sum, vis, "Fires (FIRMS T21 monthly sum)"


# Static chart builders

def chart_line(df, ycol, ytitle, title):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y(f"{ycol}:Q", title=ytitle),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip(ycol)],
        )
        .properties(height=330, title=title)
    )


def chart_bar(df, ycol, ytitle, title):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y(f"{ycol}:Q", title=ytitle),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip(ycol)],
        )
        .properties(height=330, title=title)
    )


# UI

st.title("Lahore Air Quality Decomposition (2019–2023)")

if not os.path.exists(PANEL_PATH):
    st.error(f"Missing {PANEL_PATH}. Run preprocessing.py first to generate it.")
    st.stop()

df = load_panel(PANEL_PATH)

view = st.selectbox(
    "Select view",
    [
        "Static: AQI & Traffic (Nightlights)",
        "Static: Greenness (NDVI) & Fires",
        "Static: Wind & PM2.5",
        "Interactive: Nightlights map",
        "Interactive: Greenness (NDVI) map",
        "Interactive: Wind speed map",
        "Interactive: Wind direction map",
        "Interactive: Fires map",
    ],
)

st.divider()


# Static views

if view.startswith("Static:"):
    c1, c2 = st.columns(2)

    if view == "Static: AQI & Traffic (Nightlights)":
        with c1:
            st.altair_chart(
                chart_line(df, "pm25_mean", "PM2.5 (µg/m³)", "Monthly PM2.5 (Lahore)"),
                use_container_width=True,
            )
        with c2:
            st.altair_chart(
                chart_line(
                    df,
                    "nightlights_avg_rad_mean",
                    "VIIRS avg radiance",
                    "Traffic/activity proxy (Nightlights)",
                ),
                use_container_width=True,
            )

    elif view == "Static: Greenness (NDVI) & Fires":
        with c1:
            st.altair_chart(
                chart_line(df, "ndvi_mean", "NDVI", "Urban greenness (NDVI)"),
                use_container_width=True,
            )
        with c2:
            if "fire_detections_count" in df.columns:
                st.altair_chart(
                    chart_bar(df, "fire_detections_count", "T21 sum", "Fire intensity proxy (monthly)"),
                    use_container_width=True,
                )
            else:
                st.warning("fire_detections_count not found in lahore_monthly_panel.csv")

    elif view == "Static: Wind & PM2.5":
        with c1:
            if "wind_speed_mean" in df.columns:
                st.altair_chart(
                    chart_line(df, "wind_speed_mean", "m/s", "Mean wind speed (monthly)"),
                    use_container_width=True,
                )
            else:
                st.warning("wind_speed_mean not found in lahore_monthly_panel.csv")
        with c2:
            st.altair_chart(
                chart_line(df, "pm25_mean", "PM2.5 (µg/m³)", "Monthly PM2.5 (Lahore)"),
                use_container_width=True,
            )


# Interactive views (folium + EE tiles)

else:
    # Initialize EE and get AOI only here (avoids EE init errors on import)
    LAHORE_AOI = ee_setup()

    month_idx = st.slider("Month", 0, len(MONTHS) - 1, len(MONTHS) - 1)
    month_date = MONTHS[month_idx]
    month_ts = pd.Timestamp(month_date.isoformat())

    opacity = st.slider("Layer opacity", 0.0, 1.0, 0.8, 0.05)

    fmap = folium.Map(location=[31.52, 74.35], zoom_start=10, tiles="cartodbpositron")

    if view == "Interactive: Nightlights map":
        img, vis, name = ee_nightlights(month_date, LAHORE_AOI)
        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

    elif view == "Interactive: Greenness (NDVI) map":
        img, vis, name = ee_ndvi(month_date, LAHORE_AOI)
        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

    elif view == "Interactive: Wind speed map":
        img, vis, name = ee_wind_speed(month_date, LAHORE_AOI)
        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

    elif view == "Interactive: Wind direction map":
        img, vis, name = ee_wind_dir(month_date, LAHORE_AOI)
        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

    elif view == "Interactive: Fires map":
        img, vis, name = ee_fires(month_date, LAHORE_AOI)
        add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

    # AOI outline rectangle
    xmin, ymin, xmax, ymax = LAHORE_BBOX
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    left, right = st.columns([2, 1])

    with left:
        st_folium(fmap, width=1000, height=650)

    with right:
        st.subheader("Selected month summary")
        kpi = kpi_row(df, month_ts)

        if kpi is None:
            st.write("No row found for this month in the panel dataset.")
        else:
            st.metric("PM2.5 (monthly mean)", f"{kpi.get('pm25_mean', float('nan')):.2f}")
            st.metric("Nightlights (avg radiance)", f"{kpi.get('nightlights_avg_rad_mean', float('nan')):.2f}")
            st.metric("NDVI (mean)", f"{kpi.get('ndvi_mean', float('nan')):.3f}")
            if "fire_detections_count" in kpi:
                st.metric("Fire intensity proxy", f"{kpi.get('fire_detections_count', float('nan')):.2f}")
            st.metric("Wind speed (mean)", f"{kpi.get('wind_speed_mean', float('nan')):.2f}")
            st.metric("Wind dir (mean)", f"{kpi.get('wind_dir_deg_mean', float('nan')):.1f}")

        st.caption(f"Selected month: {month_date.strftime('%Y-%m')}")