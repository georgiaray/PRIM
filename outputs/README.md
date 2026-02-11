# Outputs Directory

This directory contains all analysis outputs from the PRIM project.

## Structure

```
outputs/
├── archive/          # Archived outputs from previous attempts
│   ├── attempt_1/   # Outputs from attempt_1
│   └── attempt_2/   # Outputs from attempt_2
└── [future outputs] # New outputs will be placed here
```

## Archive Contents

### `archive/attempt_1/`
- `selected_subsets/` - CSV files containing selected variable subsets by sector
  - Industry_variables.csv
  - Electricity_variables.csv
  - Transport_variables.csv
  - Residential & Commercial_variables.csv
  - AFOLU_variables.csv
  - subset_1.csv through subset_10.csv
  - subset_x.csv
- `selected_data_245.csv` - Selected data with 245 scenarios
- `selected_data_599.csv` - Selected data with 599 scenarios

### `archive/attempt_2/`
- `output/` - Various output CSV files from variable selection
  - selected_variables.csv
  - selected_variables_2.csv through selected_variables_10.csv
  - selected_industry_variables.csv
  - selected_electricity_variables.csv
  - agriculture_df.csv
  - buildings_df.csv
  - industry_df.csv
  - transport_df.csv
  - electricity_df.csv
  - test.csv
- `selected_variables/` - Selected variable combinations
  - 1.csv through 10.csv

## Notes

- All previous outputs have been moved to `archive/` to keep the main outputs directory clean
- New analysis outputs should be placed directly in `outputs/` (not in archive)
- Archive is organized by attempt number to maintain historical context
