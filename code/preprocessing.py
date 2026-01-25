import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely import wkt

script_dir = Path(__file__).parent
raw_data = script_dir / '../data/raw-data/fire.csv'
output_path = script_dir / '../data/derived-data/fire_filtered.gpkg'

df = pd.read_csv(raw_data)
df['geometry'] = df['geometry'].apply(wkt.loads)
fire_gdf = gpd.GeoDataFrame(df, geometry='geometry')

fire_gdf = fire_gdf[fire_gdf['FIRE_YEAR'] > 2015]

fire_gdf.to_file(output_path)