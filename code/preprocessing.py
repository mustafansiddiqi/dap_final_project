from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import ee

# Paths

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

AQI_PATH = DATA_DIR / "PAQI_lahore_hourly_pm25_2019_2024.csv"
MOTOR_VEHICLES_PATH = DATA_DIR / "motor-vehicles-registered-by-type-division-and-district-the-punjab-uptil-2021.csv"

VIIRS_OUT = DATA_DIR / "lahore_viirs_nightlights_monthly_2019_2023.csv"
NDVI_OUT = DATA_DIR / "lahore_ndvi_monthly_2019_2023.csv"
PANEL_OUT = DATA_DIR / "lahore_monthly_panel.csv"
MOTOR_VEHICLES_TIDY_OUT = DATA_DIR / "motor_vehicles_subset_tidy.csv"



# Time window + AOI

START_DATE = date(2019, 1, 1)
END_EXCL = date(2024, 1, 1)

LAHORE_BBOX = [74.10, 31.35, 74.50, 31.65]


def month_starts(start: date, end_excl: date):
    out = []
    cur = date(start.year, start.month, 1)
    while cur < end_excl:
        out.append(cur)
        cur = (cur + relativedelta(months=1)).replace(day=1)
    return out


def lahore_geometry():
    xmin, ymin, xmax, ymax = LAHORE_BBOX
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])


def ee_init():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def to_df_from_featurecollection(fc: ee.FeatureCollection) :
    info = fc.getInfo()
    feats = info.get("features", [])
    rows = [f.get("properties", {}) for f in feats]
    return pd.DataFrame(rows)



# AQI: hourly to monthly mean

def load_aqi_hourly_to_monthly(aqi_path: Path) :
    df = pd.read_csv(aqi_path)

    if "timestamp_utc" not in df.columns or "pm25_ugm3" not in df.columns:
        raise ValueError(f"Unexpected AQI columns: {df.columns.tolist()}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "pm25_ugm3"])

    df = df[(df["timestamp_utc"] >= "2019-01-01") & (df["timestamp_utc"] < "2024-01-01")].copy()
    df["date"] = df["timestamp_utc"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("date", as_index=False).agg(pm25_mean=("pm25_ugm3", "mean"))
    return monthly.sort_values("date").reset_index(drop=True)



# Earth Engine extracts

def extract_viirs_monthly(aoi: ee.Geometry, months: list[date]):
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(START_DATE.isoformat(), END_EXCL.isoformat())
        .select("avg_rad")
    )

    def one_month(d: date):
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")
        img = viirs.filterDate(mstart, mend).mean()

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=500,
            maxPixels=1e13,
        )

        return ee.Feature(None, {"date": mstart.format("YYYY-MM-dd"), "nightlights_avg_rad_mean": stats.get("avg_rad")})

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def extract_ndvi_monthly(aoi: ee.Geometry, months: list[date]):
    modis = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(START_DATE.isoformat(), END_EXCL.isoformat())
        .select("NDVI")
    )

    def one_month(d: date):
        mstart = ee.Date(d.isoformat())
        mend = mstart.advance(1, "month")

        img = modis.filterDate(mstart, mend).mean().multiply(0.0001).rename("ndvi")

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=250,
            maxPixels=1e13,
        )

        return ee.Feature(None, {"date": mstart.format("YYYY-MM-dd"), "ndvi_mean": stats.get("ndvi")})

    fc = ee.FeatureCollection([one_month(d) for d in months])
    df = to_df_from_featurecollection(fc)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)



# Motor vehicles: tidy subset (Lahore + Lahore Divn merged)

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


def tidy_motor_vehicles(mv_path: Path):
    mv = pd.read_csv(mv_path)

    region_col = "Division/ District"
    if region_col not in mv.columns:
        raise ValueError(f"Expected column '{region_col}' not found. Got: {mv.columns.tolist()}")

    keep_regions = ["Lahore", "Sheikhupura", "Lahore Divn."]
    sub = mv[mv[region_col].isin(keep_regions)].copy()

    # merge Lahore + Lahore Divn.
    def map_region(r: str):
        if r in {"Lahore", "Lahore Divn."}:
            return "Lahore (incl. Divn.)"
        return r

    sub["region"] = sub[region_col].map(map_region)

    value_cols = [c for c in sub.columns if c not in {region_col, "region"}]
    value_cols = [c for c in value_cols if c.lower().strip() != "total"]  # keep only types here

    tidy = sub.melt(id_vars=["region"], value_vars=value_cols, var_name="vehicle_type_raw", value_name="count_raw")
    tidy["count"] = tidy["count_raw"].map(clean_num)

    tidy["vehicle_type"] = (
        tidy["vehicle_type_raw"]
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("Scoo- ters", "Scooters", regex=False)
        .str.replace("Auto Rick- shaws", "Auto Rickshaws", regex=False)
        .str.replace("Pick- ups/ Deli- very Vans", "Pick-ups / Delivery Vans", regex=False)
        .str.strip()
    )

    tidy = tidy[["region", "vehicle_type", "count"]]
    tidy = tidy.groupby(["region", "vehicle_type"], as_index=False)["count"].sum()
    tidy = tidy.sort_values(["region", "count"], ascending=[True, False]).reset_index(drop=True)
    return tidy


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not AQI_PATH.exists():
        raise FileNotFoundError(f"Missing AQI file: {AQI_PATH}")

    ee_init()
    aoi = lahore_geometry()
    months = month_starts(START_DATE, END_EXCL)

    print("Pulling VIIRS nightlights...")
    df_viirs = extract_viirs_monthly(aoi, months)
    df_viirs.to_csv(VIIRS_OUT, index=False)

    print("Pulling MODIS NDVI...")
    df_ndvi = extract_ndvi_monthly(aoi, months)
    df_ndvi.to_csv(NDVI_OUT, index=False)

    print("Loading AQI hourly and aggregating to monthly...")
    aqi_monthly = load_aqi_hourly_to_monthly(AQI_PATH)

    print("Merging monthly panel...")
    panel = aqi_monthly.merge(df_viirs, on="date", how="left").merge(df_ndvi, on="date", how="left")
    panel.to_csv(PANEL_OUT, index=False)

    if MOTOR_VEHICLES_PATH.exists():
        mv_tidy = tidy_motor_vehicles(MOTOR_VEHICLES_PATH)
        mv_tidy.to_csv(MOTOR_VEHICLES_TIDY_OUT, index=False)
        print(f"Saved motor vehicles tidy subset: {MOTOR_VEHICLES_TIDY_OUT}")


main()