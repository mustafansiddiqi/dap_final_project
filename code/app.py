import os
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import altair as alt
import streamlit as st

import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium



# App config

st.set_page_config(page_title="Lahore Air Quality Decomposition", layout="wide")

DATA_DIR = "data"
PANEL_PATH = os.path.join(DATA_DIR, "lahore_monthly_panel.csv")

# Lahore AOI
LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]
LAHORE = ee.Geometry.Rectangle(LAHORE_BBOX)

START = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)



# Helpers

@st.cache_resource
def ee_init():
    ee.Initialize()

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
    # Keep only 2019–2023
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2024-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)

def kpi_row(df: pd.DataFrame, dt: pd.Timestamp):
    row = df.loc[df["date"] == dt]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return r



# EE layer builders

def ee_nightlights(month_date: date):
    s, e = month_range(month_date)
    img = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
           .filterDate(s, e)
           .select("avg_rad")
           .mean()
           .clip(LAHORE))
    vis = {"min": 0, "max": 60}
    return img, vis, "Nightlights (VIIRS)"

def ee_ndvi(month_date: date):
    s, e = month_range(month_date)
    img = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(s, e)
           .select("NDVI")
           .mean()
           .multiply(0.0001)
           .clip(LAHORE))
    vis = {"min": 0.0, "max": 0.8}
    return img, vis, "NDVI (Greenness)"

def ee_wind_speed(month_date: date):
    s, e = month_range(month_date)
    uv = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
          .filterDate(s, e)
          .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
          .mean())
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed").clip(LAHORE)
    vis = {"min": 0, "max": 8}
    return speed, vis, "Wind speed (10m)"

def ee_wind_dir(month_date: date):
    s, e = month_range(month_date)
    uv = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
          .filterDate(s, e)
          .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
          .mean())
    u = uv.select("u_component_of_wind_10m")
    v = uv.select("v_component_of_wind_10m")
    # direction = atan2(u, v) degrees normalized
    dir_deg = u.atan2(v).multiply(180 / 3.141592653589793).add(360).mod(360).rename("wind_dir").clip(LAHORE)
    vis = {"min": 0, "max": 360}
    return dir_deg, vis, "Wind direction (deg)"

def ee_fires(month_date: date):
    s, e = month_range(month_date)
    fires_fc = (ee.FeatureCollection("MODIS/061/MCD14ML")
                .filterDate(s, e)
                .filterBounds(LAHORE))
    painted = ee.Image().byte().paint(fires_fc, 1).clip(LAHORE)
    vis = {"min": 0, "max": 1}
    return painted, vis, "Fire detections"



# UI

st.title("Lahore Air Quality Decomposition (2019–2023)")

if not os.path.exists(PANEL_PATH):
    st.error(f"Missing {PANEL_PATH}. Run preprocessing.py first to generate it.")
    st.stop()

df = load_panel(PANEL_PATH)

# Main dropdown determines what to show
view = st.selectbox(
    "Choose what to display",
    [
        "Static: AQI & Traffic (Nightlights)",
        "Static: Greenness (NDVI) & Fires",
        "Static: Wind & Fires vs PM2.5",
        "Interactive: Nightlights map (monthly slider)",
        "Interactive: Greenness map (NDVI) (monthly slider)",
        "Interactive: Wind speed map (monthly slider)",
        "Interactive: Wind direction map (monthly slider)",
        "Interactive: Fires map (monthly slider)",
    ],
)

st.divider()


def chart_line(df, ycol, ytitle, title):
    return (alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Month"),
                y=alt.Y(f"{ycol}:Q", title=ytitle),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip(ycol)]
            )
            .properties(height=330, title=title))

def chart_bar(df, ycol, ytitle, title):
    return (alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Month"),
                y=alt.Y(f"{ycol}:Q", title=ytitle),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip(ycol)]
            )
            .properties(height=330, title=title))

if view.startswith("Static:"):
    c1, c2 = st.columns(2)

    if view == "Static: AQI & Traffic (Nightlights)":
        with c1:
            st.altair_chart(
                chart_line(df, "pm25_mean", "PM2.5 (µg/m³)", "Monthly PM2.5 (Lahore)"),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                chart_line(df, "nightlights_avg_rad_mean", "VIIRS avg radiance", "Traffic/activity proxy (Nightlights)"),
                use_container_width=True
            )

    elif view == "Static: Greenness (NDVI) & Fires":
        with c1:
            st.altair_chart(
                chart_line(df, "ndvi_mean", "NDVI", "Urban greenness (NDVI)"),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                chart_bar(df, "fire_detections_count", "Detections", "Fire detections (monthly)"),
                use_container_width=True
            )

    elif view == "Static: Wind & Fires vs PM2.5":
        # Keep it simple: show wind speed, wind direction, PM2.5, and fires
        with c1:
            st.altair_chart(
                chart_line(df, "wind_speed_mean", "m/s", "Mean wind speed (monthly)"),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                chart_line(df, "wind_dir_deg_mean", "degrees", "Mean wind direction (monthly)"),
                use_container_width=True
            )

        c3, c4 = st.columns(2)
        with c3:
            st.altair_chart(
                chart_line(df, "pm25_mean", "PM2.5 (µg/m³)", "Monthly PM2.5 (Lahore)"),
                use_container_width=True
            )
        with c4:
            st.altair_chart(
                chart_bar(df, "fire_detections_count", "Detections", "Fire detections (monthly)"),
                use_container_width=True
            )



# Interactive views (EE maps)

else:
    ee_init()

    # month slider
    month_idx = st.slider("Month", 0, len(MONTHS) - 1, len(MONTHS) - 1)
    month_date = MONTHS[month_idx]
    month_ts = pd.Timestamp(month_date.isoformat())

    # KPIs for selected month from panel
    kpi = kpi_row(df, month_ts)

    left, right = st.columns([2, 1])

    with left:
        m = geemap.Map(center=[31.52, 74.35], zoom=10)

        if view == "Interactive: Nightlights map (monthly slider)":
            img, vis, name = ee_nightlights(month_date)
        elif view == "Interactive: Greenness map (NDVI) (monthly slider)":
            img, vis, name = ee_ndvi(month_date)
        elif view == "Interactive: Wind speed map (monthly slider)":
            img, vis, name = ee_wind_speed(month_date)
        elif view == "Interactive: Wind direction map (monthly slider)":
            img, vis, name = ee_wind_dir(month_date)
        elif view == "Interactive: Fires map (monthly slider)":
            img, vis, name = ee_fires(month_date)
        else:
            img, vis, name = ee_nightlights(month_date)

        opacity = st.slider("Layer opacity", 0.0, 1.0, 0.8, 0.05)

        m.addLayer(img, vis, name, shown=True, opacity=opacity)
        m.addLayer(LAHORE, {}, "Lahore AOI", shown=True)

        st_folium(m, width=1000, height=650)

        st.caption(f"Selected month: {month_date.strftime('%Y-%m')}")

    with right:
        st.subheader("Selected month summary")

        if kpi is None:
            st.write("No panel row found for this month (check your merge).")
        else:
            st.metric("PM2.5 (monthly mean)", f"{kpi.get('pm25_mean', float('nan')):.2f}")
            st.metric("Nightlights (avg radiance)", f"{kpi.get('nightlights_avg_rad_mean', float('nan')):.2f}")
            st.metric("NDVI (mean)", f"{kpi.get('ndvi_mean', float('nan')):.3f}")
            st.metric("Fire detections", f"{int(kpi.get('fire_detections_count', 0))}")
            st.metric("Wind speed (mean)", f"{kpi.get('wind_speed_mean', float('nan')):.2f}")
            st.metric("Wind dir (mean)", f"{kpi.get('wind_dir_deg_mean', float('nan')):.1f}")

        st.divider()
        st.subheader("Quick context")

        # Tiny trend snippet around the selected month
        window = df[(df["date"] >= month_ts - pd.offsets.MonthBegin(6)) &
                    (df["date"] <= month_ts + pd.offsets.MonthBegin(6))].copy()

        if not window.empty:
            mini = alt.Chart(window).mark_line().encode(
                x="date:T",
                y=alt.Y("pm25_mean:Q", title="PM2.5"),
                tooltip=["date:T", "pm25_mean:Q"]
            ).properties(height=220, title="PM2.5 around selected month")
            st.altair_chart(mini, use_container_width=True)
