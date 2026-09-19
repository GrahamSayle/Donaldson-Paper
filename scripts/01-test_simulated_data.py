#### Preamble ####
# Purpose: Tests the simulation of the homelessness deaths dataset.
# Author: Graham Sayle
# Date: 19 September 2026
# Contact: g.sayle@mail.utoronto.ca
# License: MIT
# Pre-requisites:
# - `polars` must be installed (pip install polars)
# - `numpy` must be installed (pip install numpy)
# - 00-simulate_data.py must have been run


#### Workspace setup ####
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path("data/00-simulated_data/simulated_data.csv")


#### Expected values ####
# These mirror the constants in 00-simulate_data.py. If the simulation
# changes, change them here too.

DEATHS_PER_YEAR = {2022: 187, 2023: 261, 2024: 236}

CAUSES = [
    "Acute Drug Toxicity",
    "Cardiovascular Disease",
    "Homicide",
    "Other Diseases",
    "Pending",
    "Suicide",
    "Unintentional Injury",
    "Unknown",
]
AGE_GROUPS = ["<20", "20-39", "40-59", "60+", "Unknown"]
GENDERS = ["Female", "Male", "Unknown"]

CAUSE_PROBS = [0.57, 0.10, 0.02, 0.11, 0.06, 0.04, 0.03, 0.07]
AGE_GROUP_PROBS = [0.01, 0.30, 0.45, 0.20, 0.04]
GENDER_PROBS = [0.19, 0.80, 0.01]

# How many standard errors an observed proportion may sit from its expected
# value before the distribution test fails. At 4, a correct simulation fails
# roughly once in 15,000 runs per category.
N_STANDARD_ERRORS = 4

REQUIRED_COLUMNS = ["year", "cause_of_death", "age_group", "gender"]


#### Load data ####
def load_data(path):
    """Read the simulated data, adding a count column if the data are
    individual-level (one row per death) rather than aggregated counts."""
    data = pl.read_csv(path)

    if "count" not in data.columns:
        data = data.with_columns(pl.lit(1).alias("count"))

    return data


#### Test 1: the data frame and all features are valid ####
def test_validity(data):
    failures = []

    if data.height == 0:
        failures.append("the data frame has no rows")
        return failures

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        failures.append(f"missing expected columns: {missing}")
        return failures

    for col in REQUIRED_COLUMNS + ["count"]:
        n_null = data[col].null_count()
        if n_null > 0:
            failures.append(f"column '{col}' has {n_null} missing values")

    allowed_values = {
        "year": list(DEATHS_PER_YEAR.keys()),
        "cause_of_death": CAUSES,
        "age_group": AGE_GROUPS,
        "gender": GENDERS,
    }

    for col, allowed in allowed_values.items():
        observed = set(data[col].drop_nulls().to_list())
        unexpected = sorted(observed - set(allowed), key=str)
        if unexpected:
            failures.append(f"column '{col}' has unexpected values: {unexpected}")

        absent = sorted(set(allowed) - observed, key=str)
        if absent:
            failures.append(f"column '{col}' never takes the values: {absent}")

    counts = data["count"].drop_nulls().to_numpy()
    if counts.size > 0:
        if not np.all(counts == np.floor(counts)):
            failures.append("column 'count' contains non-integer values")
        if np.min(counts) < 1:
            failures.append("column 'count' contains values below 1")

    if "death_id" in data.columns:
        n_duplicated = data.height - data["death_id"].n_unique()
        if n_duplicated > 0:
            failures.append(f"column 'death_id' has {n_duplicated} duplicates")

    return failures


#### Test 2: the distributions are roughly as simulated ####
def test_distributions(data):
    failures = []
    n = int(data["count"].sum())

    if n == 0:
        failures.append("no deaths to check distributions against")
        return failures

    features = [
        ("cause_of_death", CAUSES, CAUSE_PROBS),
        ("age_group", AGE_GROUPS, AGE_GROUP_PROBS),
        ("gender", GENDERS, GENDER_PROBS),
    ]

    for col, categories, probs in features:
        totals = data.group_by(col).agg(pl.col("count").sum())
        observed = dict(zip(totals[col].to_list(), totals["count"].to_list()))

        for category, expected_prob in zip(categories, probs):
            observed_prob = observed.get(category, 0) / n
            standard_error = np.sqrt(expected_prob * (1 - expected_prob) / n)
            tolerance = N_STANDARD_ERRORS * standard_error

            if abs(observed_prob - expected_prob) > tolerance:
                failures.append(
                    f"'{col}' = '{category}': observed {observed_prob:.3f}, "
                    f"expected {expected_prob:.3f} "
                    f"(tolerance +/- {tolerance:.3f})"
                )

    return failures


#### Test 3: the counts sum to the number simulated ####
def test_totals(data):
    failures = []

    expected_total = sum(DEATHS_PER_YEAR.values())
    observed_total = int(data["count"].sum())

    if observed_total != expected_total:
        failures.append(
            f"total deaths: observed {observed_total}, expected {expected_total}"
        )

    totals = data.group_by("year").agg(pl.col("count").sum())
    observed_by_year = dict(zip(totals["year"].to_list(), totals["count"].to_list()))

    for year, expected in DEATHS_PER_YEAR.items():
        observed = int(observed_by_year.get(year, 0))
        if observed != expected:
            failures.append(
                f"deaths in {year}: observed {observed}, expected {expected}"
            )

    return failures


#### Run tests ####
def main():
    if not DATA_PATH.exists():
        print(f"FAIL  no simulated data found at {DATA_PATH}")
        print("      run 00-simulate_data.py first")
        sys.exit(1)

    data = load_data(DATA_PATH)

    tests = [
        ("Data frame and features are valid", test_validity),
        ("Distributions match the simulation", test_distributions),
        ("Counts sum to the number simulated", test_totals),
    ]

    all_passed = True

    for name, test in tests:
        try:
            failures = test(data)
        except Exception as error:  # a crashing test is a failing test
            failures = [f"test raised {type(error).__name__}: {error}"]

        if failures:
            all_passed = False
            print(f"FAIL  {name}")
            for failure in failures:
                print(f"      - {failure}")
        else:
            print(f"PASS  {name}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()