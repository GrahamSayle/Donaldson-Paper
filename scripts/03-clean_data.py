#### Preamble ####
# Purpose: Cleans the raw "Homeless deaths by cause" records published by
# Toronto Public Health into the analysis dataset used by the paper.
# Author: Graham Sayle
# Date: 20 September 2026
# Contact: g.sayle@mail.utoronto.ca
# License: MIT
# - `polars` must be installed (pip install polars)
# - `numpy` must be installed (pip install numpy)
# - 02-download_data.R must have been run


#### Workspace setup ####
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path("data/01-raw_data/raw_data.csv")

## Clean data to make plotting easier, namely setting "Suppressed" counts to NA for easy filtering
raw_data = pl.read_csv(DATA_PATH)

raw_data = raw_data.with_columns((pl.col("Count") == "Suppressed").alias("is_suppressed"))

raw_data = raw_data.with_columns(
    pl.when(pl.col("is_suppressed")).then(None).otherwise(pl.col("Count")).alias("Count")
)

## Write data
raw_data.write_csv("data/02-analysis_data/analysis_data.csv")