"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:    get_penn_table.py
Purpose: Download Penn World Table 10.0, filter to the U.S. + EU15 sample,
         build an EU15 aggregate (sum-across-countries of real GDP), and
         persist data/penn_gdp.xlsx for downstream dataset construction.
Pipeline:    Data-build chain (raw -> data/penn_gdp.xlsx; raw cache at
             data/raw_data/Penn_table/penn.xlsx). Optional: the main pipeline
             reads the cached data/penn_gdp.xlsx, not this script's output.

Source: https://www.rug.nl/ggdc/docs/pwt100.xlsx — Penn World Table 10.0 from
GGDC. We use rgdpe (expenditure-side real GDP at chained PPPs) as the GDP
measure; ISO3 codes are remapped to ISO2 via country_converter so downstream
joins with EUKLEMS (which uses ISO2) work.
"""

import pandas as pd
import country_converter as coco

data_temp = pd.read_excel('https://www.rug.nl/ggdc/docs/pwt100.xlsx',
                          sheet_name='Data')

# Cache the raw download before any transformation.
data_temp.to_excel('../data/raw_data/Penn_table/penn.xlsx', index=False)

iso2_list = coco.convert(names=data_temp['countrycode'].values, to='ISO2', not_found=None)
data_temp.loc[:, 'countrycode'] = iso2_list

# EU15 member states plus the U.S. reference. GR = Greece (not EL), GB = UK.
countries = ["US", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"]

country_filter = data_temp['countrycode'].isin(countries)
data_temp = data_temp[country_filter]

# EU15 aggregate = sum of member-state rgdpe by year. This is a simple
# summation, not a PPP-consistent reaggregation; used only as a reference
# series in stylized-fact plots, not in the structural calibration.
data_EU15 = data_temp[data_temp.countrycode != "US"].groupby('year', as_index=False).agg('sum')
data_EU15['countrycode'] = "EU15"
data_EU15['country'] = "EU15"
data_EU15['currency_unit'] = " "

data_temp = pd.concat((data_temp, data_EU15), axis=0)

data_temp = data_temp[['countrycode', 'year', 'rgdpe']]
data_temp.columns = ['country', 'year', 'gdp']

data_temp.to_excel('../data/penn_gdp.xlsx', index=False)
