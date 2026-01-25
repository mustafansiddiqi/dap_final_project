# Fire Perimeter Analysis

This project processes and visualizes historical fire perimeter data.

## Setup

```bash
conda env create -f environment.yml
conda activate fire_analysis
```

## Project Structure

```
data/
  raw-data/         # Raw fire.csv data
  derived-data/     # Filtered data and output plots
code/
  preprocessing.py  # Filters fire data to post-2015
  plot_fires.py     # Plots fire perimeters
```

## Usage

1. Run preprocessing to filter data:
   ```bash
   python code/preprocessing.py
   ```

2. Generate the fire perimeter plot:
   ```bash
   python code/plot_fires.py
   ```
