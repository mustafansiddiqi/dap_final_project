import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent
data_path = script_dir / '../data/derived-data/cpi_filtered.csv'
output_path = script_dir / '../data/derived-data/cpi_plot.png'

df = pd.read_csv(data_path)
df = df.set_index('Product')

# Select key categories to plot
categories = ['All-items', 'Food 5', 'Gasoline', 'Shelter 6']
df_plot = df.loc[categories].T

# Convert index to datetime
df_plot.index = pd.to_datetime(df_plot.index, format='%B %Y')
df_plot = df_plot.sort_index()

# Rename columns for cleaner legend
df_plot.columns = ['All Items', 'Food', 'Gasoline', 'Shelter']

plt.figure(figsize=(12, 6))
for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col, linewidth=2)

plt.title('Canadian Consumer Price Index (2020-2024)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('CPI (2002=100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
