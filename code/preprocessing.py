import os
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import ee

DATA_DIR = "/Users/mustafa/Mustafa_Mac/UChicago/Year 2/Winter 2026/DAP2/Final Project/dap_final_project/data"

AQI_PATH = os.path.join(DATA_DIR, "PAQI_lahore_hourly_pm25_2019_2024.csv")

VIIRS_OUT = os.path.join(DATA_DIR, "lahore_viirs_nightlights_monthly_2019_2023.csv")
NDVI_OUT  = os.path.join(DATA_DIR, "lahore_ndvi_monthly_2019_2023.csv")
WIND_OUT  = os.path.join(DATA_DIR, "lahore_wind_monthly_2019_2023.csv")
FIRE_OUT  = os.path.join(DATA_DIR, "lahore_fires_monthly_2019_2023.csv")

PANEL_OUT = os.path.join(DATA_DIR, "lahore_monthly_panel.csv")

#GASOLINE_PATH = os.path.join(DATA_DIR, "energy_institute_table.csv")



# TIME WINDOW

START_DATE = date(2019, 1, 1)
END_EXCL   = date(2024, 1, 1) 



# AOI (Lahore bounding box)

LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def month_starts(start: date, end_excl: date) -> list[date]:
    out = []
    cur = date(start.year, start.month, 1)
    while cur < end_excl:
        out.append(cur)
        cur = (cur + relativedelta(months=1)).replace(day=1)
    return out


def lahore_geometry() -> ee.Geometry:
    xmin, ymin, xmax, ymax = LAHORE_BBOX
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])


def to_df_from_featurecollection(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    feats = info.get("features", [])
    rows = [f.get("properties", {}) for f in feats]
    return pd.DataFrame(rows)


def wind_dir_from_uv(u: ee.Image, v: ee.Image) -> ee.Image:
    dir_deg = u.atan2(v).multiply(180.0 / 3.141592653589793)
    dir_deg = dir_deg.add(360).mod(360)
    return dir_deg.rename("wind_dir_deg")


def ee_init() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# EARTH ENGINE EXTRACTS

def extract_viirs_monthly(aoi: ee.Geometry, months: list[date]) -> pd.DataFrame:

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(START_DATE.isoformat(), END_EXCL.isoformat())
             .select("avg_rad"))

    def one_month(d: date) -> ee.Feature:
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")
        img = viirs.filterDate(mstart, mend).mean()

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=500,
            maxPixels=1e13
        )

        return ee.Feature(None, {
            "date": mstart.format("YYYY-MM-dd"),
            "nightlights_avg_rad_mean": stats.get("avg_rad")
        })

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def extract_ndvi_monthly(aoi: ee.Geometry, months: list[date]) -> pd.DataFrame:
    
    modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
             .filterDate(START_DATE.isoformat(), END_EXCL.isoformat())
             .select("NDVI"))

    def one_month(d: date) -> ee.Feature:
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")

        img = modis.filterDate(mstart, mend).mean().multiply(0.0001).rename("ndvi")

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=250,
            maxPixels=1e13
        )

        return ee.Feature(None, {
            "date": mstart.format("YYYY-MM-dd"),
            "ndvi_mean": stats.get("ndvi")
        })

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def extract_wind_monthly(aoi: ee.Geometry, months: list[date]) -> pd.DataFrame:

    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(START_DATE.isoformat(), END_EXCL.isoformat())
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"]))

    def one_month(d: date) -> ee.Feature:
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")

        uv = era5.filterDate(mstart, mend).mean()
        u = uv.select("u_component_of_wind_10m")
        v = uv.select("v_component_of_wind_10m")

        speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")
        direction = wind_dir_from_uv(u, v)

        speed_stats = speed.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10000,
            maxPixels=1e13
        )

        dir_stats = direction.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10000,
            maxPixels=1e13
        )

        return ee.Feature(None, {
            "date": mstart.format("YYYY-MM-dd"),
            "wind_speed_mean": speed_stats.get("wind_speed"),
            "wind_dir_deg_mean": dir_stats.get("wind_dir_deg")
        })

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def extract_fires_monthly(aoi, months):

    # NASA FIRMS VIIRS fire detections (public + stable)
    fires = ee.ImageCollection("FIRMS").filterDate("2019-01-01", "2025-01-01")

    def one_month(d):
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")

        monthly = fires.filterDate(mstart, mend)

        # count fire pixels
        count_img = monthly.select("T21").sum()

        stats = count_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e13
        )

        return ee.Feature(None, {
            "date": mstart.format("YYYY-MM-dd"),
            "fire_detections_count": stats.get("T21")
        })

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)



# AQI:


def load_aqi_hourly_to_monthly(aqi_path: str) -> pd.DataFrame:
    df = pd.read_csv(aqi_path)

    if "timestamp_utc" not in df.columns or "pm25_ugm3" not in df.columns:
        raise ValueError(f"Unexpected AQI columns: {df.columns.tolist()}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])

    # keep only 2019–2023 inclusive
    df = df[(df["timestamp_utc"] >= "2019-01-01") & (df["timestamp_utc"] < "2025-01-01")].copy()

    df["year"] = df["timestamp_utc"].dt.year
    df["month"] = df["timestamp_utc"].dt.month

    monthly = (
        df.groupby(["year", "month"], as_index=False)
          .agg(pm25_mean=("pm25_ugm3", "mean"))
    )

    monthly["date"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01")
    monthly = monthly[["date", "pm25_mean"]].sort_values("date").reset_index(drop=True)
    return monthly


# MAIN


def main() -> None:
    ensure_dir(DATA_DIR)

    # 1) Earth Engine init
    ee_init()

    aoi = lahore_geometry()
    months = month_starts(START_DATE, END_EXCL)

    # 2) Pull satellite covariates + save
    print("Pulling VIIRS nightlights...")
    df_viirs = extract_viirs_monthly(aoi, months)
    df_viirs.to_csv(VIIRS_OUT, index=False)
    print(f"Saved: {VIIRS_OUT}")

    print("Pulling MODIS NDVI...")
    df_ndvi = extract_ndvi_monthly(aoi, months)
    df_ndvi.to_csv(NDVI_OUT, index=False)
    print(f"Saved: {NDVI_OUT}")

    print("Pulling ERA5-Land wind...")
    df_wind = extract_wind_monthly(aoi, months)
    df_wind.to_csv(WIND_OUT, index=False)
    print(f"Saved: {WIND_OUT}")

    print("Pulling MODIS fires...")
    df_fire = extract_fires_monthly(aoi, months)
    df_fire.to_csv(FIRE_OUT, index=False)
    print(f"Saved: {FIRE_OUT}")

    # 3) Load AQI hourly -> monthly
    print("Loading AQI hourly and aggregating to monthly...")
    aqi_monthly = load_aqi_hourly_to_monthly(AQI_PATH)

    # 4) Merge panel
    print("Merging monthly panel...")
    panel = (aqi_monthly
             .merge(df_viirs, on="date", how="left")
             .merge(df_ndvi, on="date", how="left")
             .merge(df_wind, on="date", how="left")
             .merge(df_fire, on="date", how="left"))
 

    panel.to_csv(PANEL_OUT, index=False)
    print(f"Saved final panel: {PANEL_OUT}")
    print(panel.head(10))


if __name__ == "__main__":
    main()
