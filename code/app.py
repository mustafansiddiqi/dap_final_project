from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
import calendar
import requests
import base64

import pandas as pd
import altair as alt
import streamlit as st

import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from geopy.geocoders import Nominatim


# Paths

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"

PANEL_PATH = DATA_DIR / "lahore_monthly_panel.csv"
PAQI_PATH = DATA_DIR / "PAQI_lahore_hourly_pm25_2019_2024.csv"
MOTOR_TIDY_PATH = DATA_DIR / "motor_vehicles_subset_tidy.csv"
MOTOR_RAW_PATH = DATA_DIR / "motor-vehicles-registered-by-type-division-and-district-the-punjab-uptil-2021.csv"
ENERGY_PATH = DATA_DIR / "energy_institute_table.csv"

# NEW: satellite image folders (produced by preprocessing.py)
SAT_IMG_DIR = DATA_DIR / "satellite_images"
VIIRS_IMG_DIR = SAT_IMG_DIR / "viirs"
NDVI_IMG_DIR = SAT_IMG_DIR / "ndvi"

LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]

START = date(2019, 1, 1)
END_EXCL = date(2025, 1, 1)

API_KEY = "48b8cf776845b1b3b76e183c60826568"


# Manual emissions per km (EDIT THESE)
# Units: g/km (or consistent unit you choose)
# Keys must match vehicle_type in motor_vehicles_subset_tidy.csv

EMISSIONS_G_PER_KM = {
    "Motor Cars, Jeeps and Station Wagons": 240,
    "Motor Cycles and Scooters": 148,
    "Trucks": 1800,
    "Pick-ups / Delivery Vans": 1350,
    "Mini Buses/ Buses/ Flying/ Luxury Coaches": 2127,
    "Taxis": 736,
    "Auto Rickshaws": 530,
    "Tractors": 1146,
    "Other Vehicles": 240,
}


def month_list():
    months = []
    cur = START
    while cur < END_EXCL:
        months.append(cur)
        cur = (cur.replace(day=1) + relativedelta(months=1))
    return months


MONTHS = month_list()


LAYER_EXPLANATION_MAP = {
    "Urban greenness (NDVI)": "Map layer shows monthly NDVI (greenness) for the Lahore bounding box (pre-saved PNGs from MODIS NDVI ×0.0001).",
    "Nightlights (VIIRS)": "Map layer shows monthly nighttime radiance for the Lahore bounding box (pre-saved PNGs from VIIRS avg_rad).",
}

LAYER_EXPLANATION_TREND = {
    "Urban greenness (NDVI)": "Trend line shows monthly mean NDVI (greenness index) averaged over the Lahore bounding box.",
    "Nightlights (VIIRS)": "Trend line shows monthly mean radiance (avg_rad) averaged over the Lahore bounding box.",
}


# Data loading

@st.cache_data
def load_panel(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2025-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_motor_tidy(path: Path):
    df = pd.read_csv(path)
    if not {"region", "vehicle_type", "count"}.issubset(df.columns):
        raise ValueError(f"Unexpected motor vehicles tidy columns: {df.columns.tolist()}")
    return df


@st.cache_data
def load_paqi_station_monthly(path: Path):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "pm25_ugm3", "latitude", "longitude"])
    df = df[(df["timestamp_utc"] >= "2019-01-01") & (df["timestamp_utc"] < "2025-01-01")].copy()
    df["date"] = df["timestamp_utc"].dt.to_period("M").dt.to_timestamp()

    out = (
        df.groupby(["date", "station_name", "latitude", "longitude"], as_index=False)
        .agg(pm25_mean=("pm25_ugm3", "mean"))
    )
    return out.sort_values(["date", "station_name"]).reset_index(drop=True)


def clean_num(x):
    if pd.isna(x):
        return float("nan")
    s = str(x).replace(",", "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


@st.cache_data
def compute_region_shares_from_motor_raw(motor_raw_path: Path):
    mv = pd.read_csv(motor_raw_path)
    if "Division/ District" not in mv.columns or "Total" not in mv.columns:
        raise ValueError("Motor vehicle raw file must contain 'Division/ District' and 'Total'.")

    mv["Total_num"] = mv["Total"].map(clean_num)

    total_punjab = float(mv.loc[mv["Division/ District"] == "The Punjab", "Total_num"].iloc[0])
    lahore_total = mv.loc[mv["Division/ District"].isin(["Lahore", "Lahore Divn."]), "Total_num"].sum()
    sheikh_total = float(mv.loc[mv["Division/ District"] == "Sheikhupura", "Total_num"].iloc[0])

    return {
        "total_punjab": total_punjab,
        "lahore_total": float(lahore_total),
        "sheikhupura_total": float(sheikh_total),
        "lahore_share": float(lahore_total / total_punjab) if total_punjab else float("nan"),
        "sheikhupura_share": float(sheikh_total / total_punjab) if total_punjab else float("nan"),
    }


@st.cache_data
def load_pakistan_gasoline_kbd(energy_path: Path):
    ei = pd.read_csv(energy_path)

    row = ei[
        (ei["Region / Grouping"].str.strip() == "Pakistan")
        & (ei["Units"].str.contains("Gasoline Consumption", na=False))
    ].copy()

    if row.empty:
        raise ValueError("Could not find Pakistan Gasoline Consumption row in Energy Institute file.")

    year_cols = [c for c in row.columns if c.isdigit()]
    long = row.melt(id_vars=["Region / Grouping", "Units"], value_vars=year_cols, var_name="year", value_name="kbd")
    long["year"] = long["year"].astype(int)
    long["kbd"] = pd.to_numeric(long["kbd"], errors="coerce")
    return long[["year", "kbd"]].sort_values("year").reset_index(drop=True)


def estimate_lahore_monthly_barrels(panel: pd.DataFrame, energy_yearly: pd.DataFrame, lahore_share: float):
    yearly = energy_yearly.set_index("year")["kbd"].to_dict()

    barrels = []
    for ts in panel["date"]:
        y = int(ts.year)
        m = int(ts.month)
        kbd = yearly.get(y, float("nan"))
        days = calendar.monthrange(y, m)[1]
        pakistan_monthly_bbl = kbd * 1000.0 * days if pd.notna(kbd) else float("nan")
        lahore_monthly_bbl = pakistan_monthly_bbl * lahore_share if pd.notna(pakistan_monthly_bbl) else float("nan")
        barrels.append(lahore_monthly_bbl)

    return pd.Series(barrels, index=panel.index, name="lahore_gasoline_barrels_est")


# Vehicle aggregation + metrics

def vehicle_summary(mv_tidy: pd.DataFrame, selected_regions: list[str]):
    df = mv_tidy[mv_tidy["region"].isin(selected_regions)].copy()
    df = df.groupby("vehicle_type", as_index=False)["count"].sum()
    df["emissions_g_per_km"] = df["vehicle_type"].map(EMISSIONS_G_PER_KM)
    df["total_emissions"] = df["count"] * df["emissions_g_per_km"]
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def compute_vehicle_metrics(vsum: pd.DataFrame):
    total_vehicles = float(vsum["count"].sum())
    valid = vsum.dropna(subset=["emissions_g_per_km"]).copy()
    weighted_avg = float((valid["count"] * valid["emissions_g_per_km"]).sum() / valid["count"].sum()) if valid["count"].sum() else float("nan")
    total_emissions = float(valid["total_emissions"].sum()) if not valid.empty else float("nan")
    return {"total_vehicles": total_vehicles, "weighted_avg_emissions": weighted_avg, "total_emissions": total_emissions}


# Charts

def pm25_with_fuel_bars(panel: pd.DataFrame):
    df = panel.copy()
    df["bar_series"] = "Estimated gasoline barrels (Lahore)"

    base = alt.Chart(df).encode(x=alt.X("date:T", title="Month"))

    bars = base.mark_bar().encode(
        y=alt.Y(
            "lahore_gasoline_barrels_est:Q",
            title="Estimated Lahore gasoline barrels (monthly)",
            axis=alt.Axis(format="~s"),
        ),
        color=alt.Color(
            "bar_series:N",
            title="",
            scale=alt.Scale(domain=["Estimated gasoline barrels (Lahore)"], range=["#4C78A8"]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Month"),
            alt.Tooltip("lahore_gasoline_barrels_est:Q", title="Estimated Lahore barrels", format=",.0f"),
            alt.Tooltip("pm25_mean:Q", title="PM2.5 (µg/m³)", format=".1f"),
        ],
    )

    line = base.mark_area().encode(
        y=alt.Y("pm25_mean:Q", title="PM2.5 (µg/m³)"),
        tooltip=[alt.Tooltip("date:T", title="Month"), alt.Tooltip("pm25_mean:Q", title="PM2.5", format=".1f")],
        opacity=alt.value(0.6), color=alt.value("#A87272")
    )

    return (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(height=340, title="PM2.5 in Lahore over time + estimated Lahore gasoline use")
    )


def vehicle_breakdown_chart(vsum: pd.DataFrame):
    vsum = vsum.copy()
    vsum["label"] = vsum.apply(
        lambda r: f"{r['count']:,.0f} | {int(r['emissions_g_per_km'])} g/km" if pd.notna(r["emissions_g_per_km"]) else f"{r['count']:,.0f} | N/A",
        axis=1,
    )

    base = (
        alt.Chart(vsum)
        .encode(
            y=alt.Y("vehicle_type:N", sort="-x", title="Vehicle type"),
            x=alt.X("count:Q", title="Registered vehicles"),
            tooltip=[
                alt.Tooltip("vehicle_type:N", title="Type"),
                alt.Tooltip("count:Q", format=",.0f"),
                alt.Tooltip("emissions_g_per_km:Q", title="Emissions (g/km)", format=",.0f"),
                alt.Tooltip("total_emissions:Q", title="Total emissions (count×g/km)", format=",.0f"),
            ],
        )
        .properties(height=420, title="Vehicle breakdown with emissions labels")
    )

    bars = base.mark_bar()
    labels = base.mark_text(align="left", dx=6).encode(text="label:N")
    return bars + labels


def satellite_trend_single(panel: pd.DataFrame, choice: str):
    df = panel.copy()

    if choice == "Urban greenness":
        col = "ndvi_mean"
        ytitle = "NDVI"
        title = "Urban greenness over time"
        fmt = ".3f"
    else:
        col = "nightlights_avg_rad_mean"
        ytitle = "Radiance avg_rad (mean over bbox)"
        title = "Nightlights over time"
        fmt = ".3f"

    return (
        alt.Chart(df.dropna(subset=[col]))
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y(f"{col}:Q", title=ytitle),
            tooltip=[
                alt.Tooltip("date:T", title="Month"),
                alt.Tooltip(f"{col}:Q", title=ytitle, format=fmt),
            ],
        )
        .properties(height=240, title=title)
    )

def img_to_data_uri(p: Path):
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def sat_image_path(mode: str, month_date: date) -> Path:
    yyyymm = f"{month_date.year}{month_date.month:02d}"
    if mode == "Nightlights (VIIRS)":
        return VIIRS_IMG_DIR / f"viirs_{yyyymm}.png"
    return NDVI_IMG_DIR / f"ndvi_{yyyymm}.png"


# App UI

st.set_page_config(page_title="Decomposing Lahore Air Quality", layout="wide")
st.title("Decomposing Lahore Air Quality from 2019–2024")

if not PANEL_PATH.exists():
    st.error(f"Missing {PANEL_PATH}. Run preprocessing.py to generate it in /data.")
    st.stop()

panel = load_panel(PANEL_PATH)

mv_tidy = load_motor_tidy(MOTOR_TIDY_PATH) if MOTOR_TIDY_PATH.exists() else None
paqi = load_paqi_station_monthly(PAQI_PATH) if PAQI_PATH.exists() else None

# Compute shares + estimated Lahore gasoline barrels and merge into panel
if MOTOR_RAW_PATH.exists() and ENERGY_PATH.exists():
    shares = compute_region_shares_from_motor_raw(MOTOR_RAW_PATH)
    ei_yearly = load_pakistan_gasoline_kbd(ENERGY_PATH)
    panel["lahore_gasoline_barrels_est"] = estimate_lahore_monthly_barrels(panel, ei_yearly, shares["lahore_share"])
else:
    shares = None
    panel["lahore_gasoline_barrels_est"] = float("nan")


# Static charts section

st.subheader("Static charts")

regions_available = ["Lahore (incl. Divn.)", "Sheikhupura"]
selected_regions = st.multiselect(
    "Regions (motor vehicle table)",
    options=regions_available,
    default=["Lahore (incl. Divn.)"],
)

if mv_tidy is not None and selected_regions:
    vsum = vehicle_summary(mv_tidy, selected_regions)
    metrics = compute_vehicle_metrics(vsum)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total registered vehicles (selected regions)", f"{metrics['total_vehicles']:,.0f}")
    c2.metric("Weighted avg emissions per km", "N/A" if pd.isna(metrics["weighted_avg_emissions"]) else f"{metrics['weighted_avg_emissions']:,.1f} g/km")
    c3.metric("Total emissions (sum count×avg)", "N/A" if pd.isna(metrics["total_emissions"]) else f"{metrics['total_emissions']:,.0f} (g/km×vehicles)")

    if shares is not None:
        st.caption(
            f"Vehicle shares (of Punjab total): Lahore (incl. Divn.) = {shares['lahore_share']:.2%}, "
            f"Sheikhupura = {shares['sheikhupura_share']:.2%}. "
            f"Punjab total vehicles = {shares['total_punjab']:,.0f}."
        )
    
    st.altair_chart(pm25_with_fuel_bars(panel), use_container_width=True)
    colA, colB = st.columns([1.0, 1.5], vertical_alignment="bottom")

    with colA:
        sat_choice = st.radio(
            "Satellite trend to display",
            options=["Urban greenness (NDVI)", "Nightlights (VIIRS)"],
            index=0,
            horizontal=True,
            key="sat_trend_choice",
        )
        st.caption(LAYER_EXPLANATION_TREND[sat_choice])
        st.altair_chart(satellite_trend_single(panel, sat_choice), use_container_width=True)

    with colB:
        st.altair_chart(vehicle_breakdown_chart(vsum), use_container_width=True)
        st.caption("Bar labels show: registered count | emissions factor.")
     
else:
    st.info("Vehicle data not available (or no regions selected).")
    sat_choice = st.radio(
        "Satellite trend to display",
        options=["Urban greenness (NDVI)", "Nightlights (VIIRS)"],
        index=0,
        horizontal=True,
        key="sat_trend_choice_no_vehicle",
    )
    st.caption(LAYER_EXPLANATION_TREND[sat_choice])
    st.altair_chart(satellite_trend_single(panel, sat_choice), use_container_width=True)

st.divider()


def blue_to_red_colormap(vmin: float, vmax: float):
    return cm.LinearColormap(
        colors=["#081d58", "#225ea8", "#41b6c4", "#ffffbf", "#fdae61", "#f46d43", "#a50026"],
        vmin=vmin,
        vmax=vmax,
    )

# Interactive map section

st.subheader("Interactive map: Air quality vs Nightlights vs Urban greenness")

map_mode = st.radio(
    "Map mode",
    options=["Air quality (PAQI monitors)", "Nightlights (VIIRS)", "Urban greenness (NDVI)"],
    index=2,
    horizontal=True,
)

selected_month_date = st.select_slider(
    "Month",
    options=MONTHS,
    value=MONTHS[-1],
    format_func=lambda d: d.strftime("%Y-%m"),
)

opacity = 0.85

fmap = folium.Map(location=[31.52, 74.35], zoom_start=10, tiles="cartodbpositron")

xmin, ymin, xmax, ymax = LAHORE_BBOX
bbox_bounds = [[ymin, xmin], [ymax, xmax]]
folium.Rectangle(bounds=bbox_bounds, color="black", weight=2, fill=False).add_to(fmap)


@st.cache_data
def paqi_global_scale(paqi_df: pd.DataFrame):
    q05 = float(paqi_df["pm25_mean"].quantile(0.05))
    q95 = float(paqi_df["pm25_mean"].quantile(0.95))
    if q05 == q95:
        q95 = q05 + 1.0
    return q05, q95


# render the selected mode
if map_mode == "Nightlights (VIIRS)":
    st.caption("Map shows **monthly nighttime radiance (VIIRS avg_rad)** for the selected month (pre-saved PNG overlay).")
    img_path = sat_image_path("Nightlights (VIIRS)", selected_month_date)

    if not img_path.exists():
        st.info(f"Missing satellite image: {img_path}. Run preprocessing.py and commit /data/satellite_images.")
    else:
        folium.raster_layers.ImageOverlay(
            image=img_to_data_uri(img_path),
            bounds=bbox_bounds,
            opacity=opacity,
            name="Nightlights (VIIRS)",
            interactive=True,
            cross_origin=False,
        ).add_to(fmap)

elif map_mode == "Urban greenness (NDVI)":
    st.caption("Map shows **monthly NDVI greenness (MODIS NDVI ×0.0001)** for the selected month (pre-saved PNG overlay).")
    img_path = sat_image_path("Urban greenness (NDVI)", selected_month_date)

    if not img_path.exists():
        st.info(f"Missing satellite image: {img_path}. Run preprocessing.py and commit /data/satellite_images.")
    else:
        folium.raster_layers.ImageOverlay(
            image=img_to_data_uri(img_path),
            bounds=bbox_bounds,
            opacity=opacity,
            name="Urban greenness (NDVI)",
            interactive=True,
            cross_origin=False,
        ).add_to(fmap)

else:
    st.caption("Map shows **monthly mean PM2.5 at PAQI monitors**. Colors go **dark blue (better) → red (worse)** and update as you change months.")

    if paqi is None:
        st.info("PAQI file not found, so monitor points can’t be shown.")
    else:
        sel_month_ts = pd.Timestamp(f"{selected_month_date.year}-{selected_month_date.month:02d}-01")
        month_df = paqi[paqi["date"] == sel_month_ts].copy()

        if month_df.empty:
            st.caption("No PAQI monitor data found for this month.")
        else:
            vmin, vmax = paqi_global_scale(paqi)
            colormap = blue_to_red_colormap(vmin, vmax)
            colormap.caption = "Monthly mean PM2.5 (µg/m³) — dark blue (better) → red (worse)"

            for _, r in month_df.iterrows():
                pm = float(r["pm25_mean"])
                color = colormap(pm)

                folium.CircleMarker(
                    location=[float(r["latitude"]), float(r["longitude"])],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.95,
                    popup=folium.Popup(
                        html=(
                            f"<b>{r['station_name']}</b><br>"
                            f"Month: {sel_month_ts.strftime('%Y-%m')}<br>"
                            f"<b>PM2.5:</b> {pm:.1f} µg/m³"
                        ),
                        max_width=280,
                    ),
                    tooltip=f"{r['station_name']} | PM2.5: {pm:.1f}",
                ).add_to(fmap)

            colormap.add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)

st_folium(
    fmap,
    width=1100,
    height=650,
    key=f"map_{map_mode}_{selected_month_date.strftime('%Y%m')}_{opacity}",
)

pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
selected_pollutants = st.multiselect("Select Pollutants", pollutants, default=["pm25"])

def aqi_description(aqi):
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }.get(aqi, "Unknown")

def extract_values(aqi_data, pollutant, mode):
    #key = pollutant_key_map(pollutant)
    if mode == "current":
        if aqi_data:
            return aqi_data[0]["components"].get(key)
    else:
        vals = [x["components"].get(key) for x in aqi_data if key in x["components"]]
        vals = [v for v in vals if v is not None]
        return sum(vals)/len(vals) if vals else None
    return None

@st.cache_data(ttl=900)
def fetch_aqi(lat, lon, mode="current", start=None, end=None):
    if mode == "forecast":
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    elif mode == "historic" and start and end:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
    else:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json().get("list", [])
    return []


st.subheader("My Location")

address = st.text_input("Enter your address or zip code:")

if address:
    geolocator = Nominatim(user_agent="aqi_chicago")
    location = geolocator.geocode(address)

    if location:
        lat, lon = location.latitude, location.longitude
        personal_data = fetch_aqi(lat, lon, "current")

        if personal_data:
            comp = personal_data[0]["components"]
            aqi_index = personal_data[0]["main"]["aqi"]
            aqi_label = aqi_description(aqi_index)

            st.markdown(
                f"### The air quality in your area is **{aqi_label}** (AQI Index: {aqi_index})."
            )
        else:
            st.warning("No AQI data available for this location.")
    else:
        st.warning("Address not found.")