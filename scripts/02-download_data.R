#### Preamble ####
# Purpose: Downloads and saves the data on deaths of people experiencing
# homelessness in Toronto from Open Data Toronto.
# Author: Graham Sayle
# Date: 19 September 2026
# Contact: g.sayle@mail.utoronto.ca
# License: MIT
# Pre-requisites:
# - `opendatatoronto` must be installed (install.packages("opendatatoronto"))
# - `tidyverse` must be installed (install.packages("tidyverse"))


#### Workspace setup ####
library(opendatatoronto)
library(tidyverse)


#### Download data ####

# Get the package (dataset) metadata
package <- search_packages("Deaths of People Experiencing Homelessness")

# List every resource (file) attached to the package
resources <- list_package_resources("8546b6d9-41ec-48df-895f-09e86784ef26")

# Keep the resource that breaks deaths down by cause
cause_resource <-
  resources |>
  filter(name == "Homeless deaths by cause")

# Download it
raw_data <- get_resource(cause_resource)


#### Save data ####
dir.create("data/01-raw_data", recursive = TRUE, showWarnings = FALSE)
write_csv(raw_data, "data/01-raw_data/.csv")