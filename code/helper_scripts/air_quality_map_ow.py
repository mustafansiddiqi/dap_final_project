import geopandas as gpd
import folium
import requests
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import streamlit as st
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

# CONFIG
st.set_page_config(page_title="Chicago AQI Map", layout="wide")
SHAPEFILE_PATH = "neighborhoods_shapefile.shp"
API_KEY = "48b8cf776845b1b3b76e183c60826568.rm698"

# Load shapefile
@st.cache_resource
def load_neighborhoods():
    gdf = gpd.read_file(SHAPEFILE_PATH).rename(columns={'neighborho': 'neighborhood'})
    gdf["neighborhood"] = gdf["neighborhood"].str.strip().str.title()
    gdf["centroid"] = gdf.centroid
    return gdf

# Fetch OpenWeather API data
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

# Pollutant key conversion
def pollutant_key_map(p):
    return {"pm25": "pm2_5", "pm10": "pm10", "o3": "o3", "no2": "no2", "so2": "so2", "co": "co"}.get(p.lower(), p)

# BallTree for future expansion
@st.cache_resource
def build_ball_tree(coords):
    return BallTree(np.radians(coords), metric='haversine')

# UI HEADER
st.markdown("<h1 style='text-align: center;'>Air Quality Map of Chicago</h1>", unsafe_allow_html=True)
st.markdown("---")

# FILTERS
pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
selected_pollutants = st.multiselect("Select Pollutants", pollutants, default=["pm25"])
color_by = st.selectbox("Color Map By", selected_pollutants)

# LOAD SHAPEFILE
neighborhoods = load_neighborhoods()
coords = np.array([[pt.y, pt.x] for pt in neighborhoods["centroid"]])
tree = build_ball_tree(coords)

# PAGE TABS
tab1, tab2, tab3, tab4 = st.tabs(["Current", "My Location", "Forecast (24h Avg)", "Historic (Date Range)"])

# AQI descriptive labels
def aqi_description(aqi):
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }.get(aqi, "Unknown")

def extract_values(aqi_data, pollutant, mode):
    key = pollutant_key_map(pollutant)
    if mode == "current":
        if aqi_data:
            return aqi_data[0]["components"].get(key)
    else:
        vals = [x["components"].get(key) for x in aqi_data if key in x["components"]]
        vals = [v for v in vals if v is not None]
        return sum(vals)/len(vals) if vals else None
    return None

def make_map(aqi_mode, aqi_data_list, title_suffix):
    neighborhoods["aq_data"] = aqi_data_list
    neighborhoods["color_val"] = neighborhoods["aq_data"].apply(lambda x: extract_values(x, color_by, aqi_mode))
    valid_vals = neighborhoods["color_val"].dropna()
    vmin, vmax = (valid_vals.min(), valid_vals.max()) if not valid_vals.empty else (0, 1)
    colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax, caption=f"{color_by.upper()} ({title_suffix})")
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="cartodbpositron")

    for _, row in neighborhoods.iterrows():
        name = row["neighborhood"]
        geom = row["geometry"]
        aq = row["aq_data"]
        val = row["color_val"]
        fill = colormap(val) if val is not None else "gray"
        html = f"<b>{name}</b><br>"

        if aq:
            if aqi_mode == "current":
                aqi_index = aq[0]['main']['aqi']
                html += f"AQI Index: {aqi_index} ({aqi_description(aqi_index)})<br>"
            for p in selected_pollutants:
                pval = extract_values(aq, p, aqi_mode)
                html += f"{p.upper()}: {round(pval, 2) if pval else 'N/A'}<br>"
        else:
            html += "No data."

        folium.GeoJson(
            geom,
            style_function=lambda _, fill_color=fill: {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.Tooltip(html)
        ).add_to(m)

        folium.map.Marker(
            [row["centroid"].y, row["centroid"].x],
            icon=folium.DivIcon(html=f"<div style='font-size:8pt;color:black'>{name}</div>")
        ).add_to(m)

    colormap.add_to(m)
    st_folium(m, width=1100, height=750)

with tab1:
    st.subheader("Current AQI")
    current_data = [fetch_aqi(row["centroid"].y, row["centroid"].x, "current") for _, row in neighborhoods.iterrows()]
    make_map("current", current_data, "Current")

with tab2:
    st.subheader("My Location")
    address = st.text_input("Enter your address or zip code:")
    if address:
        geolocator = Nominatim(user_agent="aqi_chicago")
        location = geolocator.geocode(address)
        if location:
            lat, lon = location.latitude, location.longitude
            personal_data = fetch_aqi(lat, lon, "current")
            m = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron")
            if personal_data:
                comp = personal_data[0]["components"]
                aqi_index = personal_data[0]['main']['aqi']
                html = f"<b>Your Location</b><br>"
                html += f"AQI Index: {aqi_index} ({aqi_description(aqi_index)})<br>"
                for p in pollutants:
                    val = comp.get(pollutant_key_map(p))
                    html += f"{p.upper()}: {round(val, 2) if val else 'N/A'}<br>"
                folium.Marker([lat, lon], tooltip=html, icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
                st_folium(m, width=1100, height=600)
            else:
                st.warning("No AQI data available for this location.")
        else:
            st.warning("Address not found.")

with tab3:
    st.subheader("24-Hour Forecast Average")
    forecast_data = [fetch_aqi(row["centroid"].y, row["centroid"].x, "forecast")[:24] for _, row in neighborhoods.iterrows()]
    make_map("forecast", forecast_data, "24h Forecast")

with tab4:
    st.subheader("Historic AQI Average")
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=2), max_value=datetime.today())
    end_date = st.date_input("End Date", datetime.today(), min_value=start_date, max_value=datetime.today())
    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end_date, datetime.min.time()).timestamp())
    historic_data = [fetch_aqi(row["centroid"].y, row["centroid"].x, "historic", start_ts, end_ts) for _, row in neighborhoods.iterrows()]
    make_map("historic", historic_data, f"{start_date} to {end_date}")
