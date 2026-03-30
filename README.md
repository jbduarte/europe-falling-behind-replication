# Replication Package

**Paper**: "Europe Falling Behind: Structural Transformation and Labor Productivity Growth Differences Between Europe and the U.S."

**Authors**: Cesare Buiatti, Joao B. Duarte, Luis Felipe Saenz

## Overview

This repository contains the data and code to replicate all figures, tables, and quantitative results in the paper. The paper develops a multi-sector structural transformation model to study the divergence in labor productivity growth between Europe and the United States since the mid-1990s. It quantifies the roles of sectoral productivity growth (within-sector effects) and labor reallocation across sectors (between-sector effects) through a series of counterfactual experiments under three model variants: closed economy, open economy with exogenous trade, and open economy with endogenous trade.

## Data Availability

All data used in the analysis are included in the `data/` folder. Original sources:

| Dataset | Source | Notes |
|---------|--------|-------|
| EU KLEMS 2023 | [EU KLEMS](https://euklems-intanprod-llee.luiss.it/) | Sectoral employment shares and labor productivity |
| OECD GDP per hour | [OECD](https://data.oecd.org/lprdty/gdp-per-hour-worked.htm) | Aggregate labor productivity (USD, constant 2010 PPP) |
| Penn World Table | [PWT 10.01](https://www.rug.nl/ggdc/productivity/pwt/) | GDP levels for initial relative productivity |
| OECD IO Tables | [OECD](https://www.oecd.org/sti/ind/input-outputtables.htm) | Sectoral trade flows (IO panel) |

**Note**: The TFP correlation analysis (Step 16) requires `growth_accounts.csv` (~180 MB), included in `data/raw_data/`. If cloning from GitHub where this file may be tracked via Git LFS, ensure LFS is installed (`git lfs install`) before cloning.

## Computational Requirements

- **Software**: Python 3.8+ with a LaTeX installation (for figure text rendering via matplotlib)
- **Packages**: See `requirements.txt`. Install with `pip install -r requirements.txt`
- **Hardware**: 8 GB RAM recommended; endogenous trade model is computationally intensive
- **Runtime**: Approximately 70 minutes total (tested on Apple M-series iMac)
  - Closed economy (Steps 1--3): ~2 minutes
  - Open economy, exogenous trade (Steps 4--6): ~2 minutes
  - Open economy, endogenous trade (Steps 7--9): ~63 minutes (Step 8 dominates)
  - Standalone analyses and output generation (Steps 10--17): ~1 minute
- **OS**: Tested on macOS; should work on Linux and Windows

## Instructions

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure LaTeX is installed (e.g., MacTeX, TeX Live, or MiKTeX)
4. Run the full replication from the `code/` directory:
   ```bash
   cd code
   python master.py
   ```
   Or run individual scripts in the order listed below.

All output is saved to `output/figures/`, `output/tables/`, and `output/data/`.

## Code Description

### Execution Order

The `master.py` script runs all 17 steps in sequence. Each script depends on variables computed by earlier scripts in the chain.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `model_calibration_USA.py` | Calibrate the baseline (closed economy) model using U.S. data |
| 2 | `model_test_europe.py` | Test the calibrated model on European economies |
| 3 | `counterfactuals.py` | Run counterfactual experiments (closed economy) |
| 4 | `model_calibration_USA_open.py` | Calibrate the open economy model with exogenous trade |
| 5 | `model_test_europe_open.py` | Test the open economy model on European economies |
| 6 | `trade_counterfactuals.py` | Run trade counterfactuals (exogenous trade) |
| 7 | `model_calibration_USA_endogenous_open.py` | Calibrate the open economy model with endogenous trade |
| 8 | `model_test_europe_endogenous_xn.py` | Test the endogenous trade model on European economies |
| 9 | `trade_counterfactuals_endogenous.py` | Run trade counterfactuals (endogenous trade) |
| 10 | `price_specification_comparison.py` | Compare price specifications: 1/A_i vs observed P_i (Table 3) |
| 11 | `utils/facts.py` | Generate stylized facts figures (Figure 1) |
| 12 | `utils/table_1_ss_eu4.py` | Shift-share decomposition (Table 1) |
| 13 | `utils/cfs.py` | Counterfactual results tables (Table 2, Table A.4, Table A.8) |
| 14 | `utils/table_ss_eu15_appendix.py` | EU-15 shift-share tables (Tables A.6, A.7) |
| 15 | `utils/table_ss_core_vs_periphery.py` | Core vs. periphery shift-share tables (Tables A.9, A.10) |
| 16 | `utils/corr_lp_tfp_klems.py` | LP--TFP correlation (Table 4; optional, requires `growth_accounts.csv`) |
| 17 | `generate_paper_outputs.py` | Generate final paper figures, tables, and consolidate with paper-consistent naming |

### Utility Modules (`code/utils/`)

| Module | Purpose |
|--------|---------|
| `construct_dataset_facts.py` | Dataset construction for stylized facts |
| `facts.py` | Stylized facts figures (Figure 1) and shift-share data |
| `table_1_ss_eu4.py` | Shift-share decomposition (Table 1) |
| `cfs.py` | Counterfactual results tables (Tables 2, A.4, A.8) |
| `table_ss_eu15_appendix.py` | EU-15 appendix shift-share tables (Tables A.6, A.7) |
| `table_ss_core_vs_periphery.py` | Core vs. periphery shift-share tables (Tables A.9, A.10) |
| `corr_lp_tfp_klems.py` | LP--TFP correlation analysis (Table 4) |

### Output Mapping

After running the full pipeline, `generate_paper_outputs.py` creates paper-consistent copies in `output/figures/` and `output/tables/`. The mapping between paper locations and output files:

#### Main Figures

| Paper | Output File | Description |
|-------|-------------|-------------|
| Figure 1a | `figure_1a.pdf` | EU Big Four labor productivity relative to U.S. |
| Figure 1b | `figure_1b.pdf` | Employment shares vs. log GDP per hour |
| Figure 2 | `figure_2.pdf` | Closed economy: employment shares scatter + relative productivity |
| Figure 3 | `figure_3.pdf` | Counterfactual decomposition of productivity gap |
| Figure 5 | `figure_5.pdf` | Trade openness by sector (US vs. Europe) |
| Figure 6 | `figure_6.pdf` | Open economy (endogenous trade): employment shares + productivity |

#### Main Tables

| Paper | Output File | Description |
|-------|-------------|-------------|
| Table 1 | `table_1.xlsx` | Shift-share decomposition of LP growth (1995--2018) |
| Table 2 | `table_2a.tex`, `table_2b.tex` | Counterfactual results (CF1, CF2) |
| Table 3 | `table_3.tex`, `table_3.xlsx` | Price specification comparison: 1/A_i vs. P_i |
| Table 4 | `table_4.tex` | LP--TFP correlation (requires `growth_accounts.csv`) |
| Table 5 | `table_5.xlsx` | Export elasticities (beta parameters) |
| Table 6 | `table_6.tex`, `table_6.xlsx` | Open economy counterfactuals (endogenous trade) |

#### Appendix Figures

| Paper | Output File | Description |
|-------|-------------|-------------|
| Figure A.1 | `figure_A1.pdf` | US sectoral labor productivity paths |
| Figure A.3 | `figure_A3a.pdf`--`figure_A3f.pdf` | US calibration fit by sector |
| Figure A.4 | `figure_A4.pdf` | 6-panel model fit: EU4, GBR, EU15 |

#### Appendix Tables

| Paper | Output File | Description |
|-------|-------------|-------------|
| Table A.4 | `table_A4a.tex`, `table_A4b.tex` | Three-sector counterfactuals |
| Table A.6 | `table_A6.tex` | EU-15 shift-share (1970--2019) |
| Table A.7 | `table_A7.tex` | EU-15 shift-share (1995--2019) |
| Table A.8 | `table_A8a.tex`--`table_A8d.tex` | EU4/EU15/GBR counterfactuals |
| Table A.9 | `table_A9.tex` | Core vs. periphery shift-share (1970--2019) |
| Table A.10 | `table_A10.tex` | Core vs. periphery shift-share (1995--2019) |

#### Intermediate Outputs

Additional intermediate data files are saved in `output/figures/` and `output/data/`:
- `Counterfactual_*.xlsx`: Counterfactual experiment results
- `trade_cure_nx_endo.xlsx`: Trade cure net export data
- `lp_KLEMS_data.xlsx`: Labor productivity data from KLEMS
- `beta_last_period_results.xlsx`: Export elasticity parameters

## Raw Data and Data Construction

The `data/raw_data/` folder contains all original source data files. The `code/data_construction/` folder contains the Python (and R) scripts that transform the raw data into the analysis-ready datasets in `data/`.

### Raw Data Sources

| Folder/File | Source | Description |
|-------------|--------|-------------|
| `raw_data/EUKLEMS_2023/data_raw.csv` | EU KLEMS 2023 | Sectoral value added, hours, prices (all countries) |
| `raw_data/EUKLEMS_2009/data_raw.csv` | EU KLEMS 2009 | Historical sectoral data (1970--1995 extension) |
| `raw_data/growth accounts.csv` | EU KLEMS 2023 Growth Accounts | TFP data for LP--TFP correlation (Table 4) |
| `raw_data/IO/` | OECD ICIO Tables | Sectoral input-output and trade flow data |
| `raw_data/OECD_GDP_PPP_NX.csv` | OECD | GDP per hour worked, PPP-adjusted |
| `raw_data/penn_world_table/` | PWT 10.01 | GDP levels for initial relative productivity |
| `raw_data/DR2_QJE_data.xls` | Duarte & Restuccia (2010) | Reference data |

### Data Construction Scripts (`code/data_construction/`)

| Script | Description |
|--------|-------------|
| `get_EUKLEMS.py` | Download and process EU KLEMS raw data |
| `get_WorldKLEMS_USA.py` | Process US KLEMS data |
| `construct_dataset.py` | Build analysis-ready panel from raw KLEMS data |
| `select_data.py` | Select countries and sectors for the six-sector model |
| `get_penn_table.py` | Process Penn World Table GDP data |
| `get_initial_rel_A.py` | Compute initial relative productivity levels |
| `add_EU15_OECD.py` | Add EU-15 aggregate OECD data |
| `construct_dataset_facts.py` | Construct dataset for stylized facts (Section 2) |
| `corr_lp_tfp_klems.py` | LP--TFP correlation (Table 4) |

The data construction pipeline runs: `get_EUKLEMS.py` -> `construct_dataset.py` -> `select_data.py` -> analysis-ready `euklems_2023.csv`. The main analysis scripts (Steps 1--17) use only the processed datasets in `data/`; the raw data and construction scripts are provided for full transparency.

## Directory Structure

```
europe-falling-behind-replication/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── euklems_2023.csv                       # Analysis-ready KLEMS panel
│   ├── penn_gdp.xlsx                          # Penn World Table GDP
│   ├── io_panel.xlsx                          # OECD IO panel (trade)
│   ├── exp_imp_aggregate_panel.xlsx           # Aggregate trade flows
│   ├── raw/                                   # OECD GDP per hour
│   │   ├── OECD_GDP_ph.xlsx
│   │   └── OECD_GDP_ph_EU15.xlsx
│   └── raw_data/                              # Original source data
│       ├── EUKLEMS_2023/data_raw.csv          # EU KLEMS 2023 release
│       ├── EUKLEMS_2009/data_raw.csv          # EU KLEMS 2009 release
│       ├── growth accounts.csv                # TFP growth accounts (~180 MB)
│       ├── IO/                                # OECD input-output tables
│       ├── penn_world_table/                   # PWT 10.01
│       └── ...                                # Other raw sources
├── code/
│   ├── master.py                              # Master replication script (runs all 17 steps)
│   ├── model_calibration_USA.py               # Step 1: US closed economy calibration
│   ├── model_test_europe.py                   # Step 2: European closed economy test
│   ├── counterfactuals.py                     # Step 3: Closed economy counterfactuals
│   ├── model_calibration_USA_open.py          # Step 4: US open economy (exogenous) calibration
│   ├── model_test_europe_open.py              # Step 5: European open economy (exogenous) test
│   ├── trade_counterfactuals.py               # Step 6: Trade counterfactuals (exogenous)
│   ├── model_calibration_USA_endogenous_open.py # Step 7: US endogenous trade calibration
│   ├── model_test_europe_endogenous_xn.py     # Step 8: European endogenous trade test
│   ├── trade_counterfactuals_endogenous.py    # Step 9: Trade counterfactuals (endogenous)
│   ├── price_specification_comparison.py      # Step 10: Price specification comparison
│   ├── generate_paper_outputs.py              # Step 17: Final outputs + consolidation
│   ├── data_construction/                     # Raw data -> analysis-ready transformations
│   │   ├── get_EUKLEMS.py
│   │   ├── construct_dataset.py
│   │   ├── select_data.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       ├── construct_dataset_facts.py         # Dataset construction for facts
│       ├── facts.py                           # Step 11: Stylized facts (Figure 1)
│       ├── table_1_ss_eu4.py                  # Step 12: Shift-share (Table 1)
│       ├── cfs.py                             # Step 13: Counterfactual tables
│       ├── table_ss_eu15_appendix.py          # Step 14: EU-15 shift-share
│       ├── table_ss_core_vs_periphery.py      # Step 15: Core vs. periphery
│       └── corr_lp_tfp_klems.py               # Step 16: LP--TFP correlation
└── output/
    ├── figures/                               # All figures (paper-consistent names)
    ├── tables/                                # All tables (paper-consistent names)
    └── data/                                  # Intermediate data files
```

## License

MIT License. See `LICENSE`.
