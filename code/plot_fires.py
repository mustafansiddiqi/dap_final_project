import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely import wkt

script_dir = Path(__file__).parent
data_path = script_dir / '../data/derived-data/fire_filtered.gpkg'

fire_df = gpd.read_file(data_path)

fig, ax = plt.subplots(figsize=(10, 8))
fire_df.plot(ax=ax, color='red', edgecolor='black', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Fire Perimeters (Post-2015)')
plt.savefig(script_dir / '../data/derived-data/fire_plot.png')
plt.show()
