#### Preamble ####
# Purpose: Simulates a dataset on the cause of death of people experiencing
# homelessness in Toronto.
# Author: Graham Sayle
# Date: 17 September 2026
# Contact: g.sayle@mail.utoronto.ca
# License: MIT
# Pre-requisites:
# - `polars` must be installed (pip install polars)
# - `numpy` must be installed (pip install numpy)


#### Workspace setup ####
from pathlib import Path

import numpy as np
import polars as pl

np.random.seed(853)


#### Simulate data ####

# Years covered by the simulated data
years = [2022, 2023, 2024]

# Number of deaths recorded in each year
deaths_per_year = {2022: 187, 2023: 261, 2024: 236}

# Cause of death categories
causes = [
    "Acute Drug Toxicity",
    "Cardiovascular Disease",
    "Homicide",
    "Other Diseases",
    "Pending",
    "Suicide",
    "Unintentional Injury",
    "Unknown",
]

# Age groups
age_groups = ["<20", "20-39", "40-59", "60+", "Unknown"]

# Genders
genders = ["Female", "Male", "Unknown"]

# Probabilities for cause, age group, and gender distribution
cause_probs = [0.57, 0.10, 0.02, 0.11, 0.06, 0.04, 0.03, 0.07]
age_group_probs = [0.01, 0.30, 0.45, 0.20, 0.04]
gender_probs = [0.19, 0.80, 0.01]

# Generate the data using numpy and polars
n_deaths = sum(deaths_per_year.values())

years_sampled = np.repeat(years, [deaths_per_year[year] for year in years])
causes_sampled = np.random.choice(causes, size=n_deaths, replace=True, p=cause_probs)
age_groups_sampled = np.random.choice(
    age_groups, size=n_deaths, replace=True, p=age_group_probs
)
genders_sampled = np.random.choice(
    genders, size=n_deaths, replace=True, p=gender_probs
)

# Create a polars DataFrame
analysis_data = pl.DataFrame(
    {
        "death_id": [f"Death {i}" for i in range(1, n_deaths + 1)],
        "year": years_sampled,
        "cause_of_death": causes_sampled,
        "age_group": age_groups_sampled,
        "gender": genders_sampled,
    }
)


#### Save data ####
Path("data/00-simulated_data").mkdir(parents=True, exist_ok=True)
analysis_data.write_csv("data/00-simulated_data/simulated_data.csv")
