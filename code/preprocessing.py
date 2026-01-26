import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely import wkt

script_dir = Path(__file__).parent

# Process fire data
raw_fire = script_dir / '../data/raw-data/fire.csv'
output_fire = script_dir / '../data/derived-data/fire_filtered.gpkg'

df = pd.read_csv(raw_fire)
df['geometry'] = df['geometry'].apply(wkt.loads)
fire_gdf = gpd.GeoDataFrame(df, geometry='geometry')

fire_gdf = fire_gdf[fire_gdf['FIRE_YEAR'] > 2015]

fire_gdf.to_file(output_fire)

# Process Canadian CPI data
raw_cpi = script_dir / '../data/raw-data/canadian_cpi.csv'
output_cpi = script_dir / '../data/derived-data/cpi_filtered.csv'

cpi_df = pd.read_csv(raw_cpi)
cpi_df = cpi_df.rename(columns={cpi_df.columns[0]: 'Product'})

# Filter to columns from 2020 onwards
date_cols = [col for col in cpi_df.columns if '2020' in col or '2021' in col or '2022' in col or '2023' in col or '2024' in col]
cpi_filtered = cpi_df[['Product'] + date_cols].dropna(subset=['Product'])

cpi_filtered.to_csv(output_cpi, index=False)
print(f"CPI data filtered: {len(cpi_filtered)} products, {len(date_cols)} months")