"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:    add_EU15_OECD.py
Purpose: Build an EU15-aggregate GDP-per-hour series from (i) country-level
         OECD PPP GDP-per-hour and (ii) EUKLEMS total hours, and append it as
         a new row to the OECD panel so downstream code can read EU15 the
         same way it reads any member state.
Pipeline:    Data-build chain (raw -> data/raw/OECD_GDP_ph_EU15.xlsx). Reads
             data/euklems_2023.csv (produced by select_data.py) and the
             OECD raw panel under data/raw_data/OECD_GDP_ph.xlsx.

Method: EU15 GDP-per-hour = (sum_i Value_i * H_i) / (sum_i H_i), where the
sum is over the 14 EU15 member states (US excluded) and H is total hours
from EUKLEMS. This is an hours-weighted average of OECD country-level
GDP-per-hour — equivalent to total EU15 real GDP divided by total EU15 hours
when each country's Value is in the same PPP USD units (2010 constant).
"""

import pandas as pd
import numpy as np

data = pd.read_csv('../data/euklems_2023.csv', index_col=[0, 1])

# EUKLEMS uses ISO2 country codes; OECD uses ISO3. Remap the index level to
# ISO3 so the merge below lines up.
iso2_to_iso3 = {
    'AT': 'AUT', 'BE': 'BEL', 'DE': 'DEU', 'DK': 'DNK', 'ES': 'ESP',
    'FI': 'FIN', 'FR': 'FRA', 'GB': 'GBR', 'GR': 'GRC', 'IE': 'IRL',
    'IT': 'ITA', 'LU': 'LUX', 'NL': 'NLD', 'PT': 'PRT', 'SE': 'SWE',
    'US': 'USA',
}
data.rename(index=iso2_to_iso3, inplace=True)

# Keep only the aggregate (tot) row per country-year; sector-level detail is
# not needed for the hours-weighted GDP-per-hour aggregation.
data = data.loc[data.sector == "tot"]
data = data.reset_index()
data.year = data.year.astype(int)

# OECD PPP GDP-per-hour: USD at constant 2010 PPPs.
GDP_ph = pd.read_excel('../data/raw_data/OECD_GDP_ph.xlsx', index_col=[0, 5],
                       engine='openpyxl')
GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']
GDP_ph = GDP_ph.reset_index()
GDP_ph['country'] = GDP_ph['LOCATION']
GDP_ph['year'] = GDP_ph['TIME']

# Merge OECD (Value = GDP/hour) with EUKLEMS (H = total hours). GDP = Value * H
# reconstructs country-level GDP in the common PPP unit.
data_comb = pd.merge(data, GDP_ph, on=['country', 'year'])
data_comb['GDP'] = data_comb['Value'] * data_comb['H']

# EU15 excludes the U.S. reference.
data_comb = data_comb[data_comb.country != "USA"]

# EU15 hours-weighted GDP-per-hour = sum(GDP) / sum(H) across member states.
data_EU15 = data_comb.groupby(["year"]).agg(sum)
data_EU15["GDP_ph"] = data_EU15["GDP"] / data_EU15["H"]

# Reload the OECD panel (raw copy, no filtering) and append EU15 as a new
# LOCATION so downstream readers see it alongside member states.
GDP_ph = pd.read_excel('../data/raw_data/OECD_GDP_ph.xlsx',
                       engine='openpyxl')

eu15 = pd.DataFrame(columns=GDP_ph.columns)
eu15["Value"] = data_EU15["GDP_ph"]
eu15["TIME"] = np.arange(1970, 2020)
eu15["LOCATION"] = "EU15"
eu15["MEASURE"] = "USD"

GDP_ph = pd.concat((GDP_ph, eu15), axis=0)
GDP_ph.to_excel('../data/raw/OECD_GDP_ph_EU15.xlsx', index=False)
