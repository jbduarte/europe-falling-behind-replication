# `data_construction/` — Raw → analysis panel

Five scripts that rebuild the contents of `data/` from original sources. **Running them is optional**: the main replication pipeline (`code/master.py`) reads the pre-built CSV/Excel files already committed in `data/`, so a reviewer who just wants to reproduce the tables and figures can skip this folder entirely.

These scripts exist so that the raw download → analysis panel transformation is transparent and auditable.

## Pipeline

```
┌────────────────────┐
│ External sources   │   EU KLEMS, World KLEMS, PWT, OECD
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│ Stage 1 — Download raw data                  │
│   get_EUKLEMS.py       -> data/raw_data/EUKLEMS_<release>/data_raw.csv
│   get_WorldKLEMS_USA.py -> data/raw_data/World_KLEMS_2013/data_raw_usa.csv
│   get_penn_table.py    -> data/raw_data/Penn_table/penn.xlsx
│                            + data/penn_gdp.xlsx
└─────────┬────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│ Stage 2 — Build the country-sector-year panel │
│   select_data.py       reads EUKLEMS raw + World KLEMS USA
│                        -> data/euklems_2023.csv
│                        (the main analysis panel used by master.py)
└─────────┬────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│ Stage 3 — Build the EU15 GDP-per-hour series  │
│   add_EU15_OECD.py     reads euklems_2023.csv + OECD_GDP_ph.xlsx
│                        -> data/raw/OECD_GDP_ph_EU15.xlsx
└──────────────────────────────────────────────┘
```

## Run order

From inside this folder:

```bash
python get_EUKLEMS.py          # Stage 1
python get_WorldKLEMS_USA.py   # Stage 1
python get_penn_table.py       # Stage 1
python select_data.py          # Stage 2
python add_EU15_OECD.py        # Stage 3
```

Stage 1 scripts are independent and can run in any order. Stage 2 must run after Stage 1. Stage 3 must run after Stage 2.

## What each script does

| Script | Source | Output | Role |
|---|---|---|---|
| `get_EUKLEMS.py` | [EU KLEMS](https://euklems-intanprod-llee.luiss.it/) (2023, and historical 2021/2017/2009 releases) | `data/raw_data/EUKLEMS_<release>/data_raw.csv` | Sectoral value added, hours, prices — the main data source for structural transformation |
| `get_WorldKLEMS_USA.py` | [World KLEMS](https://www.worldklems.net/) (2013 U.S. release) | `data/raw_data/World_KLEMS_2013/data_raw_usa.csv` | Used only to extend the U.S. series back to 1970 (EU KLEMS 2023 starts in 1995) |
| `get_penn_table.py` | [Penn World Table 10.0](https://www.rug.nl/ggdc/productivity/pwt/) | `data/penn_gdp.xlsx` | GDP levels used to anchor initial relative productivity |
| `select_data.py` | EU KLEMS 2023 + 2009 (for back-extrapolation) + World KLEMS USA | `data/euklems_2023.csv` | The country-sector-year analysis panel consumed by `master.py`. Filters to EU15+US, collapses NACE industries into the paper's six structural sectors, back-extrapolates to 1970 using 2009-release growth rates, and appends service composites (`ser`, `prs`) |
| `add_EU15_OECD.py` | `data/euklems_2023.csv` + OECD GDP-per-hour | `data/raw/OECD_GDP_ph_EU15.xlsx` | Builds an hours-weighted EU15 GDP-per-hour aggregate and appends it to the OECD panel so downstream code can read EU15 the same way it reads any country |

## Six-sector taxonomy (from `select_data.py`)

| Code | Sector | NACE Rev.2 industries |
|---|---|---|
| `agr` | agriculture | A |
| `man` | manufacturing + mining + construction + utilities | B, C, D-E, F |
| `trd` | trade (wholesale + retail) | G |
| `bss` | business services | J, M-N |
| `fin` | finance | K |
| `nps` | non-progressive services | H, I, L, O, P, Q, R-S, T |

Composites appended in `select_data.py`: `ser` (all services = bss+fin+trd+nps), `prs` (progressive services = bss+fin+trd), `tot` (aggregate).

## Files NOT produced here

Some inputs under `data/` cannot be regenerated from scripts in this folder — they must be downloaded manually from the original source (OECD portals require interactive filters and do not expose stable CSV URLs). These are shipped as-is in the replication package:

| File | Source |
|---|---|
| `data/io_panel.xlsx`, `data/exp_imp_aggregate_panel.xlsx` | [OECD ICIO Tables](https://www.oecd.org/sti/ind/input-outputtables.htm), 2021 release |
| `data/raw/OECD_GDP_ph.xlsx` | [OECD GDP per hour worked](https://data.oecd.org/lprdty/gdp-per-hour-worked.htm) |
| `data/raw_data/growth accounts.csv` (~180 MB) | [EU KLEMS 2023 Growth Accounts](https://euklems-intanprod-llee.luiss.it/) |
| `data/beta_last_period_results.xlsx` | Authors' construction — regenerated automatically by `model_test_europe_endogenous_xn.py` when the main pipeline runs |
