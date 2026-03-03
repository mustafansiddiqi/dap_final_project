from __future__ import annotations

from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
import calendar

import pandas as pd
import altair as alt
import streamlit as st

import ee
import folium
from streamlit_folium import st_folium
import branca.colormap as cm



# Paths

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"

PANEL_PATH = DATA_DIR / "lahore_monthly_panel.csv"
PAQI_PATH = DATA_DIR / "PAQI_lahore_hourly_pm25_2019_2024.csv"
MOTOR_TIDY_PATH = DATA_DIR / "motor_vehicles_subset_tidy.csv"
MOTOR_RAW_PATH = DATA_DIR / "motor-vehicles-registered-by-type-division-and-district-the-punjab-uptil-2021.csv"
ENERGY_PATH = DATA_DIR / "energy_institute_table.csv"  # Pakistan gasoline consumption kb/d

LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]  # [xmin, ymin, xmax, ymax]

START = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)



# Manual emissions per km (EDIT THESE)
# Units: g/km (or consistent unit you choose)
# Keys must match vehicle_type in motor_vehicles_subset_tidy.csv

EMISSIONS_G_PER_KM = {
    "Motor Cars, Jeeps and Station Wagons": 180,
    "Motor Cycles and Scooters": 70,
    "Trucks": 900,
    "Pick-ups / Delivery Vans": 400,
    "Mini Buses/ Buses/ Flying/ Luxury Coaches": 1100,
    "Taxis": 200,
    "Auto Rickshaws": 250,
    "Tractors": 1200,
    "Other Vehicles": 300,
}

# Earth Engine setup

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
    img = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(s, e)
        .select("avg_rad")
        .mean()
        .clip(aoi)
    )
    return img, {"min": 0, "max": 60}, "Nightlights (VIIRS)"


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
    return img, {"min": 0.0, "max": 0.8}, "Urban greenness (NDVI)"


LAYER_EXPLANATION_MAP = {
    "Urban greenness (NDVI)": "Map layer shows monthly mean NDVI (greenness index) averaged over the Lahore bounding box (MODIS NDVI, rescaled ×0.0001).",
    "Nightlights (VIIRS)": "Map layer shows monthly mean nighttime radiance averaged over the Lahore bounding box (VIIRS DNB avg_rad).",
}

LAYER_EXPLANATION_TREND = {
    "Urban greenness (NDVI)": "Trend line shows monthly mean NDVI (greenness index) averaged over the Lahore bounding box.",
    "Nightlights (VIIRS)": "Trend line shows monthly mean radiance (avg_rad) averaged over the Lahore bounding box.",
}


# Data loading

@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2019-01-01") & (df["date"] < "2024-01-01")].copy()
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_motor_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"region", "vehicle_type", "count"}.issubset(df.columns):
        raise ValueError(f"Unexpected motor vehicles tidy columns: {df.columns.tolist()}")
    return df


@st.cache_data
def load_paqi_station_monthly(path: Path) -> pd.DataFrame:
    
    # Hourly PAQI -> station-level monthly mean PM2.5.
    # Output: date (month start), station_name, latitude, longitude, pm25_mean
    
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


def clean_num(x) -> float:
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
def compute_region_shares_from_motor_raw(motor_raw_path: Path) -> dict:
    
    # Uses Total column to compute shares of total Punjab registered vehicles:
    #   - Lahore (incl. Divn.) share
    #   - Sheikhupura share

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
def load_pakistan_gasoline_kbd(energy_path: Path) -> pd.DataFrame:
    
    # energy_institute_table.csv format:
    #   Region / Grouping, Units, 1980..2024
    # We extract Pakistan Gasoline Consumption (kb/d) and return yearly series.
    
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


def estimate_lahore_monthly_barrels(panel: pd.DataFrame, energy_yearly: pd.DataFrame, lahore_share: float) -> pd.Series:

    # Convert annual kb/d to monthly barrels:
    #   kbd * 1000 (bbl/day) * days_in_month
    # Then multiply by lahore_share to estimate Lahore barrels.

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

def vehicle_summary(mv_tidy: pd.DataFrame, selected_regions: list[str]) -> pd.DataFrame:
    df = mv_tidy[mv_tidy["region"].isin(selected_regions)].copy()
    df = df.groupby("vehicle_type", as_index=False)["count"].sum()
    df["emissions_g_per_km"] = df["vehicle_type"].map(EMISSIONS_G_PER_KM)
    df["total_emissions"] = df["count"] * df["emissions_g_per_km"]
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def compute_vehicle_metrics(vsum: pd.DataFrame) -> dict:
    total_vehicles = float(vsum["count"].sum())
    valid = vsum.dropna(subset=["emissions_g_per_km"]).copy()
    weighted_avg = float((valid["count"] * valid["emissions_g_per_km"]).sum() / valid["count"].sum()) if valid["count"].sum() else float("nan")
    total_emissions = float(valid["total_emissions"].sum()) if not valid.empty else float("nan")
    return {"total_vehicles": total_vehicles, "weighted_avg_emissions": weighted_avg, "total_emissions": total_emissions}



# Charts

def pm25_with_fuel_bars(panel: pd.DataFrame) -> alt.Chart:

    # Bars: estimated Lahore gasoline barrels (legend + custom color + include AQI in tooltip)
    # Line: PM2.5
    
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

    line = base.mark_line().encode(
        y=alt.Y("pm25_mean:Q", title="PM2.5 (µg/m³)"),
        tooltip=[alt.Tooltip("date:T", title="Month"), alt.Tooltip("pm25_mean:Q", title="PM2.5", format=".1f")],
    )

    return (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(height=340, title="PM2.5 in Lahore over time + estimated Lahore gasoline use (bars)")
    )


def vehicle_breakdown_chart(vsum: pd.DataFrame) -> alt.Chart:
    vsum = vsum.copy()
    vsum["label"] = vsum.apply(
        lambda r: f"{r['count']:,.0f} | {int(r['emissions_g_per_km'])} g/km" if pd.notna(r["emissions_g_per_km"]) else f"{r['count']:,.0f} | N/A",
        axis=1,
    )

    base = (
        alt.Chart(vsum)
        .encode(
            y=alt.Y("vehicle_type:N", sort="-x", title="Vehicle type"),
            x=alt.X("count:Q", title="Registered vehicles (up to 2021)"),
            tooltip=[
                alt.Tooltip("vehicle_type:N", title="Type"),
                alt.Tooltip("count:Q", format=",.0f"),
                alt.Tooltip("emissions_g_per_km:Q", title="Emissions (g/km)", format=",.0f"),
                alt.Tooltip("total_emissions:Q", title="Total emissions (count×g/km)", format=",.0f"),
            ],
        )
        .properties(height=420, title="Vehicle breakdown (counts) with emissions labels")
    )

    bars = base.mark_bar()
    labels = base.mark_text(align="left", dx=6).encode(text="label:N")
    return bars + labels


def satellite_trend_single(panel: pd.DataFrame, choice: str) -> alt.Chart:
    
    # Single-series trend depending on choice:
    #   - NDVI: ndvi_mean
    #   - VIIRS: nightlights_avg_rad_mean

    df = panel.copy()

    if choice == "Urban greenness (NDVI)":
        col = "ndvi_mean"
        ytitle = "NDVI (mean over bbox)"
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



# App UI

st.set_page_config(page_title="Lahore Air Quality + Vehicles + Satellite", layout="wide")
st.title("Lahore: Air Quality, Vehicle Mix, Fuel Use (Estimated), and Satellite Indicators (2019–2024)")

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

    colA, colB = st.columns([1.15, 1.0], vertical_alignment="top")

    with colA:
        st.altair_chart(pm25_with_fuel_bars(panel), use_container_width=True)

        # satellite trend radio button + explainer (like earlier)
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
        st.caption("Bar labels show: registered count | emissions factor (manual). Edit EMISSIONS_G_PER_KM in code to change.")

else:
    st.info("Vehicle data not available (or no regions selected).")
    # still show satellite trend control even if vehicle missing
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

def blue_to_red_colormap(vmin: float, vmax: float) -> cm.LinearColormap:
    # dark blue -> light blue -> yellow -> orange -> red
    return cm.LinearColormap(
        colors=["#081d58", "#225ea8", "#41b6c4", "#ffffb2", "#fe9929", "#cc4c02", "#b10026"],
        vmin=vmin,
        vmax=vmax,
    )


# Interactive map section


st.subheader("Interactive map: Air quality vs Nightlights vs Urban greenness")

aoi = ee_setup()

# 3-way selector (NOT overlay)
map_mode = st.radio(
    "Map mode",
    options=["Air quality (PAQI monitors)", "Nightlights (VIIRS)", "Urban greenness (NDVI)"],
    index=2,  # default NDVI on load
    horizontal=True,
)

# Month slider (drives all three modes)
selected_month_date = st.select_slider(
    "Month",
    options=MONTHS,
    value=MONTHS[-1],
    format_func=lambda d: d.strftime("%Y-%m"),
)

opacity = st.slider("Layer opacity (satellite only)", 0.0, 1.0, 0.85, 0.05)

# Base map
fmap = folium.Map(location=[31.52, 74.35], zoom_start=10, tiles="cartodbpositron")

# Outline bbox
xmin, ymin, xmax, ymax = LAHORE_BBOX
folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False).add_to(fmap)

# fixed color scale for PAQI so months are comparable
@st.cache_data
def paqi_global_scale(paqi_df: pd.DataFrame) -> tuple[float, float]:
    # use robust bounds so outliers don't wreck the scale
    q05 = float(paqi_df["pm25_mean"].quantile(0.05))
    q95 = float(paqi_df["pm25_mean"].quantile(0.95))
    if q05 == q95:
        q95 = q05 + 1.0
    return q05, q95

def blue_to_red_colormap(vmin: float, vmax: float) -> cm.LinearColormap:
    # dark blue -> cyan -> yellow -> orange -> red
    return cm.LinearColormap(
        colors=["#081d58", "#225ea8", "#41b6c4", "#ffffbf", "#fdae61", "#f46d43", "#a50026"],
        vmin=vmin,
        vmax=vmax,
    )

# render the selected mode
if map_mode == "Nightlights (VIIRS)":
    st.caption("Map shows **monthly mean nighttime radiance (VIIRS avg_rad)** averaged within each pixel for the selected month.")
    img, vis, name = ee_nightlights(selected_month_date, aoi)
    add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

elif map_mode == "Urban greenness (NDVI)":
    st.caption("Map shows **monthly mean NDVI greenness index (MODIS NDVI ×0.0001)** for the selected month.")
    img, vis, name = ee_ndvi(selected_month_date, aoi)
    add_ee_tile_layer(fmap, img, vis, name, opacity=opacity)

else:
    # Air quality (PAQI monitors)
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