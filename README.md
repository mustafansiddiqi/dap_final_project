# README.md
# Lahore Air Quality Analysis (2019–2024)

## Authors

**Mustafa Siddiqi**
University of Chicago
GitHub: https://github.com/mustafansiddiqi

**Khushan Shahad**
University of Chicago
GitHub: https://github.com/kshahad

## Project Overview

This project analyzes environmental and economic drivers of air pollution in **Lahore, Pakistan** using publicly available environmental, satellite, and transportation datasets. The goal is to explore **how air quality has evolved over time and what factors are associated with these changes**.

The analysis combines **ground-based air quality monitoring data**, **satellite-derived environmental indicators**, and **transportation and fuel consumption data** to examine patterns in PM2.5 pollution.

Rather than conducting causal inference, this project focuses on **descriptive policy analysis** to better understand pollution dynamics and highlight potential drivers of air quality deterioration.

---

## Live Interactive Dashboard

A fully interactive version of this analysis is available on **Streamlit Community Cloud**:

**Dashboard:**
https://lahore-aqi.streamlit.app/

The dashboard allows users to:

* Explore **monthly PM2.5 levels** across monitoring stations
* Compare **satellite indicators of urban activity and vegetation**
* Analyze **vehicle emissions and transportation composition**
* View **satellite imagery layers (nightlights and greenness)**
* Investigate **temporal changes in Lahore’s environmental conditions**

---

## Repository Structure

```
lahore-aqi/
│
├── code/
│   ├── app.py                # Streamlit dashboard
│   ├── preprocessing.py      # Data processing pipeline
│   └── analysis.ipynb        # Exploratory analysis and chart generation
│
├── data/
│   ├── lahore_monthly_panel.csv
│   ├── PAQI_lahore_hourly_pm25_2019_2024.csv
│   ├── motor_vehicles_subset_tidy.csv
│   ├── motor-vehicles-registered-by-type-division-and-district-the-punjab-uptil-2021.csv
│   ├── energy_institute_table.csv
│   └── satellite_images/
│       ├── viirs/
│       └── ndvi/
│
├── figures/
│
├── report/
│   └── lahore_air_quality_analysis.qmd
│
└── README.md
```

---

## Research Question

Lahore frequently ranks among the most polluted cities in the world. Policymakers often attribute the problem to several potential drivers including:

* Vehicular emissions
* Agricultural burning
* Urban construction and dust
* Cross-border pollution
* Meteorological conditions

This project asks:

**What factors are associated with the deterioration of air quality in Lahore, and do available data suggest that current policy interventions are improving conditions?**

To address this question we examine three key dimensions:

1. **Air quality trends** from ground monitoring stations
2. **Urban economic activity** using satellite nightlights
3. **Urban environmental change** using vegetation indices (NDVI)

Transportation indicators such as **vehicle registrations and gasoline consumption** are also incorporated to evaluate possible emission contributions.

---

## Data Sources

The project integrates multiple publicly available datasets.

### 1. Pakistan Air Quality Initiative (PAQI)

**Source:**
https://aqicn.org/

**Data:**
Hourly PM2.5 readings from monitoring stations across Lahore.

**Coverage:**
2019–2024

**Processing Steps:**

* Raw hourly measurements are cleaned and filtered
* Invalid timestamps and missing values are removed
* Observations are aggregated into **monthly averages per station**
* Station-level data are then averaged to produce **city-level monthly PM2.5**

---

### 2. VIIRS Nighttime Lights (NASA / NOAA)

**Source:**
https://earthdata.nasa.gov/

**Dataset:**
VIIRS Day/Night Band (VNP46A2)

**Purpose:**
Nighttime light intensity serves as a proxy for **economic activity, urban expansion, and energy use**.

**Processing Steps:**

1. Satellite imagery is accessed using **Google Earth Engine**
2. A **bounding box covering Lahore** is defined
3. Monthly mean radiance values (`avg_rad`) are calculated
4. Values are exported as a **monthly time series**
5. Satellite images are also exported as **PNG layers for the dashboard**

---

### 3. MODIS NDVI (Vegetation Index)

**Source:**
https://lpdaac.usgs.gov/

**Dataset:**
MODIS MOD13Q1 NDVI

**Purpose:**
NDVI measures vegetation greenness and is used as a proxy for **urban green space and environmental change**.

**Processing Steps:**

* NDVI imagery is retrieved through **Google Earth Engine**
* Monthly NDVI values are averaged across the Lahore bounding box
* Values are scaled using the MODIS factor (×0.0001)
* Monthly averages are exported to the panel dataset
* Satellite images are saved for map overlays

---

### 4. Punjab Motor Vehicle Registration Data

**Source:**
Punjab Excise & Taxation Department

**Dataset:**
Motor vehicles registered by district and vehicle type.

**Purpose:**
Provides insight into the **composition of Lahore’s vehicle fleet**.

**Processing Steps:**

* Data are cleaned and reshaped into a **tidy format**
* Vehicle types are aggregated
* Emission intensity estimates (g/km) are applied to each category
* Aggregate emission indicators are calculated

Vehicle emission factors are approximated using typical emission estimates for:

* Motorcycles
* Cars
* Trucks
* Buses
* Auto-rickshaws
* Delivery vehicles

---

### 5. Energy Institute Fuel Consumption Data

**Source:**
Energy Institute Statistical Review of World Energy

**Dataset:**
Pakistan gasoline consumption (thousand barrels per day).

**Purpose:**
Provides a national-level indicator of **fuel usage trends**.

**Processing Steps:**

1. Annual gasoline consumption is extracted
2. Monthly estimates are calculated using days per month
3. Lahore’s share is approximated using its proportion of Punjab vehicle registrations
4. Monthly gasoline consumption is estimated for Lahore

---

## Data Integration

All processed datasets are merged into a **single monthly panel dataset**:

```
lahore_monthly_panel.csv
```

Key variables include:

* `date`
* `pm25_mean`
* `ndvi_mean`
* `nightlights_avg_rad_mean`
* `lahore_gasoline_barrels_est`

This unified dataset enables **temporal comparisons between pollution, urban development, and transportation trends**.

---

## Analytical Approach

The analysis focuses on **visual exploration and descriptive statistics** rather than causal identification.

Key components include:

### PM2.5 Trend Analysis

Monthly pollution trends are compared with estimated gasoline consumption to assess whether transportation activity tracks pollution levels.

### Vehicle Emissions Decomposition

Vehicle counts are combined with emission intensity estimates to evaluate which vehicle types contribute most to emissions.

### Satellite Environmental Indicators

Nighttime lights and NDVI trends are analyzed to examine how economic activity and urban greenness evolve alongside pollution.

---

## Key Findings

Several patterns emerge from the analysis:

**Strong Seasonality in Air Pollution**

PM2.5 levels consistently spike during winter months (November–February), suggesting that seasonal factors such as agricultural burning or atmospheric inversions play a major role.

**Fuel Consumption Alone Does Not Explain Pollution Trends**

Estimated gasoline usage remains relatively stable while PM2.5 fluctuates significantly.

**Heavy Vehicles Are High Emitters**

Although motorcycles dominate the vehicle fleet, trucks and buses emit substantially higher pollution per kilometer.

**Economic Activity Appears Stable**

Nighttime lights show modest growth but do not track pollution spikes.

**Vegetation Trends Are Seasonal**

NDVI peaks during monsoon months, reflecting increased greenness.

---

## Interactive Dashboard Features

The Streamlit application provides several tools for exploring these relationships.

**Satellite Map Layers**

Users can toggle between:

* NDVI (vegetation greenness)
* VIIRS nighttime lights

**Air Quality Monitoring Map**

Monthly PM2.5 levels are displayed across monitoring stations.

**Vehicle Emissions Analysis**

Users can dynamically explore vehicle fleet composition and emissions contributions.

**Temporal Trends**

Charts visualize the evolution of:

* PM2.5
* Satellite indicators
* Estimated gasoline consumption

---

## Technology Stack

The project was implemented using:

* **Python**
* **Pandas**
* **Altair**
* **Google Earth Engine**
* **Streamlit**
---