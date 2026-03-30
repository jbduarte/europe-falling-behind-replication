import matplotlib
matplotlib.use("Agg")
"""
=======================================================================================
LP--TFP correlation analysis (Table 4).

Requires growth_accounts.csv (~180 MB) from EU KLEMS Growth Accounts.
Download from https://euklems-intanprod-llee.luiss.it/ and place in data/raw/.

Author: Joao B. Duarte
Last Modified: Feb 2026
=======================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

growth_acc_path = '../data/raw/growth accounts.csv'
if not os.path.exists(growth_acc_path):
    print(f"  SKIPPED: {growth_acc_path} not found (optional, ~180 MB download).")
    sys.exit(0)

# read data
growth_acc_db = pd.read_csv(growth_acc_path)
lp_db = pd.read_excel('../output/data/lp_KLEMS_data.xlsx')

growth_acc_db = growth_acc_db.loc[growth_acc_db['var'] == 'LP2TFP_I', ['year', 'geo_code', 'nace_r2_code', 'var', 'value']]

# classify sectors
sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "nps", "H": "nps",
  "J": "bss", "K": "fin", "L": "nps", "M-N": "bss", "O": "nps",
  "P": "nps", "Q": "nps", "R-S": "nps", "T": "nps", "TOT": "tot"}

growth_acc_db['sector'] = growth_acc_db['nace_r2_code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
countries = ['AT', 'BE', 'DE', 'DK', 'ES', 'FI', 'FR', 'UK', 'EL', 'IE', 'IT', 'LU', 'NL', 'PT', 'SE']
sector_filter = growth_acc_db['sector'].isin(sectors_dict.values())
country_filter = growth_acc_db['geo_code'].isin(countries)

growth_acc_db = growth_acc_db.loc[sector_filter, :]
growth_acc_db = growth_acc_db.loc[country_filter, :]

# next find growth rates
growth_acc_db['value'] = growth_acc_db.loc[:, ["value", "geo_code", "sector"]].groupby(["geo_code", "sector"], as_index=False).transform(lambda x: sm.tsa.filters.hpfilter(x,100)[1])
growth_acc_db['log_diff_TFP'] = growth_acc_db.loc[:, ["value", "geo_code", "sector"]].groupby(["geo_code", "sector"], as_index=False).transform(lambda x: np.log(x).diff())

# subset both db to 1996-2019 period
growth_acc_db = growth_acc_db.loc[(growth_acc_db.year >= 1996) & (growth_acc_db.year <= 2019), :]
lp_db = lp_db.loc[(lp_db.year >= 1996) & (lp_db.year <= 2019), :]

lp_db = pd.melt(lp_db, id_vars=['year', 'country'], value_vars=lp_db.columns[2:],
        var_name='sector', value_name='value')

growth_acc_db.reset_index(inplace=True, drop=True)

countries_old = ['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE']
ids = {countries[i]: countries_old[i] for i in range(len(countries))}
lp_db['geo_code'] = lp_db['country']
lp_db['sector'] = lp_db['sector'].apply(lambda x: x[6:])
growth_acc_db['geo_code'] = growth_acc_db.loc[:, ["geo_code"]].replace(ids, regex=True)


final = pd.merge(growth_acc_db, lp_db, on=['geo_code', 'sector', 'year'])
final['log_diff_LP_paper'] = final['value_y']

# aggregation of sectors are not comparable. Here we only have a valid value for agr, trd, fin and tot
for sector in pd.unique(['agr', 'trd', 'fin', 'tot']):
    select_sec_year = (final.sector == sector)

    print(sector)
    print(final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr())
    print('\n')

    # concatenate by sector in rows in a dataframe
    if sector == 'agr':
        corr = final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr()
        corr['sector'] = sector
    else:
        temp = final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr()
        temp['sector'] = sector
        corr = pd.concat([corr, temp])

def format_corr_table(corr):
    latex_table = ""
    sectors = corr['sector'].unique()
    for sector in sectors:
        sector_data = corr[corr['sector'] == sector].drop(columns=['sector'])
        latex_table += f"\\multicolumn{{2}}{{c}}{{{sector}}} \\\\\n"
        latex_table += sector_data.to_latex(header=True, index=True)
        latex_table += "\\\\\n"
    return latex_table

# Generate the formatted LaTeX table
corr_latex_table = format_corr_table(corr)

# Write the LaTeX table to a file
with open('../output/figures/corr_lp_tfp_klems.tex', 'w') as f:
    f.write(corr_latex_table)

# break corr into 4 tables by sector
corr_agr = corr.loc[corr.sector =='agr', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_agr.index = ['LP in paper', 'TFP in EUKLEMS']
corr_agr.columns = ['LP in paper', 'TFP in EUKLEMS']

# repeat for trd
corr_trd = corr.loc[corr.sector =='trd', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_trd.index = ['LP in paper', 'TFP in EUKLEMS']
corr_trd.columns = ['LP in paper', 'TFP in EUKLEMS']

# repeat for fin
corr_fin = corr.loc[corr.sector =='fin', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_fin.index = ['LP in paper', 'TFP in EUKLEMS']
corr_fin.columns = ['LP in paper', 'TFP in EUKLEMS']

# repeat for tot
corr_tot = corr.loc[corr.sector =='tot', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_tot.index = ['LP in paper', 'TFP in EUKLEMS']
corr_tot.columns = ['LP in paper', 'TFP in EUKLEMS']

# write a latex table with all corr_agr, corr_trd, corr_fin, corr_tot
corr = pd.concat([corr_agr, corr_trd, corr_fin, corr_tot], axis=0)

# write corr in a latex table with sector headers every 2 rows





with open('../output/figures/corr_lp_tfp_klems.tex', 'w') as f:
    f.write(corr.to_latex())

# write a latex table with the printed results


