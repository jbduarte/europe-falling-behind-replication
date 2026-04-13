import matplotlib
matplotlib.use("Agg")
"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        corr_lp_tfp_klems.py
Purpose:     Compute correlations between the paper's sectoral labour-productivity
             growth series and TFP growth from the EU KLEMS Growth Accounts
             (sectors with comparable aggregations: agr, trd, fin, tot).
Pipeline:    Step 18/19 — Generates Table 4. OPTIONAL: skips gracefully if the
             EU KLEMS growth-accounts file is not present.
Inputs:      ../data/raw_data/growth accounts.csv (EU KLEMS Growth Accounts; ~180 MB,
             not shipped with the replication package — download from
             https://euklems-intanprod-llee.luiss.it/ and place in data/raw_data/),
             ../output/data/lp_KLEMS_data.xlsx (sectoral LP series produced upstream).
Outputs:     ../output/figures/corr_lp_tfp_klems.tex (correlation table for Table 4).
Dependencies: lp_KLEMS_data.xlsx must already exist (built earlier in the pipeline).
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

growth_acc_path = '../data/raw_data/growth accounts.csv'
if not os.path.exists(growth_acc_path):
    # The EU KLEMS growth-accounts file is ~180 MB and not redistributed with
    # this package. Exit cleanly so master.py can flag Step 18 as skipped
    # without aborting the whole pipeline.
    print(f"  SKIPPED: {growth_acc_path} not found (optional, ~180 MB download).")
    sys.exit(0)

growth_acc_db = pd.read_csv(growth_acc_path)
lp_db = pd.read_excel('../output/data/lp_KLEMS_data.xlsx')

# LP2TFP_I is the EU KLEMS series that yields TFP growth once a sector-level
# log-diff is taken (industry-level total factor productivity index). Other
# variables in the file (LP_I, GVA_Q_I, ...) are ignored here.
growth_acc_db = growth_acc_db.loc[growth_acc_db['var'] == 'LP2TFP_I', ['year', 'geo_code', 'nace_r2_code', 'var', 'value']]

# Map NACE Rev. 2 industries to the paper's six-sector aggregation. Note the
# bundling decisions: B/C/D-E/F (mining + manufacturing + utilities + construction)
# all collapse into 'man'; J + M-N go into 'bss' (information + professional
# services); H + I + L + O-T go into 'nps' (transport, hospitality, real estate,
# public sector). 'TOT' is preserved as a level check on the aggregation.
sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "nps", "H": "nps",
  "J": "bss", "K": "fin", "L": "nps", "M-N": "bss", "O": "nps",
  "P": "nps", "Q": "nps", "R-S": "nps", "T": "nps", "TOT": "tot"}

growth_acc_db['sector'] = growth_acc_db['nace_r2_code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
# EU-15 sample. EU KLEMS uses Eurostat ISO-2 codes ('UK' for the United
# Kingdom, 'EL' for Greece). These get harmonised to ISO-3 below for the merge.
countries = ['AT', 'BE', 'DE', 'DK', 'ES', 'FI', 'FR', 'UK', 'EL', 'IE', 'IT', 'LU', 'NL', 'PT', 'SE']
sector_filter = growth_acc_db['sector'].isin(sectors_dict.values())
country_filter = growth_acc_db['geo_code'].isin(countries)

growth_acc_db = growth_acc_db.loc[sector_filter, :]
growth_acc_db = growth_acc_db.loc[country_filter, :]

# Apply the same HP filter (lambda=100, annual) used elsewhere in the paper
# and then log-difference within country-sector to obtain TFP growth rates.
# Smoothing first ensures the correlation in Table 4 contrasts low-frequency
# components rather than year-to-year measurement noise.
growth_acc_db['value'] = growth_acc_db.loc[:, ["value", "geo_code", "sector"]].groupby(["geo_code", "sector"], as_index=False).transform(lambda x: sm.tsa.filters.hpfilter(x,100)[1])
growth_acc_db['log_diff_TFP'] = growth_acc_db.loc[:, ["value", "geo_code", "sector"]].groupby(["geo_code", "sector"], as_index=False).transform(lambda x: np.log(x).diff())

# Common 1996-2019 sample: EU KLEMS coverage starts in 1995 across all EU-15
# countries; one observation is lost to the first log-difference, so 1996 is
# the first usable correlation year.
growth_acc_db = growth_acc_db.loc[(growth_acc_db.year >= 1996) & (growth_acc_db.year <= 2019), :]
lp_db = lp_db.loc[(lp_db.year >= 1996) & (lp_db.year <= 2019), :]

lp_db = pd.melt(lp_db, id_vars=['year', 'country'], value_vars=lp_db.columns[2:],
        var_name='sector', value_name='value')

growth_acc_db.reset_index(inplace=True, drop=True)

# Harmonise to ISO-3 so the inner merge below succeeds. lp_db ships sector
# columns prefixed with a 6-character tag (e.g. "LP_I_agr"), so x[6:] strips
# it down to the bare sector code.
countries_old = ['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE']
ids = {countries[i]: countries_old[i] for i in range(len(countries))}
lp_db['geo_code'] = lp_db['country']
lp_db['sector'] = lp_db['sector'].apply(lambda x: x[6:])
growth_acc_db['geo_code'] = growth_acc_db.loc[:, ["geo_code"]].replace(ids, regex=True)


final = pd.merge(growth_acc_db, lp_db, on=['geo_code', 'sector', 'year'])
final['log_diff_LP_paper'] = final['value_y']

# Only agr, trd, fin and the total aggregate map cleanly between the two
# sector taxonomies. The other paper sectors (man, bss, nps) bundle multiple
# NACE industries differently from EU KLEMS, so their correlations would not
# be apples-to-apples and are dropped from Table 4.
for sector in pd.unique(['agr', 'trd', 'fin', 'tot']):
    select_sec_year = (final.sector == sector)

    print(sector)
    print(final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr())
    print('\n')

    # Stack 2x2 sector correlation matrices vertically. The first sector
    # initialises the dataframe; subsequent sectors are appended so the final
    # `corr` is a (4*2) x 3 long-format table (rows: LP and TFP per sector).
    if sector == 'agr':
        corr = final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr()
        corr['sector'] = sector
    else:
        temp = final.loc[select_sec_year, ['log_diff_LP_paper', 'log_diff_TFP']].corr()
        temp['sector'] = sector
        corr = pd.concat([corr, temp])

def format_corr_table(corr):
    """Stack per-sector 2x2 correlation matrices into a single LaTeX block,
    with a \\multicolumn header naming each sector. Used as the legacy
    formatter; the file ultimately overwrites the output with a simpler
    `corr.to_latex()` call below."""
    latex_table = ""
    sectors = corr['sector'].unique()
    for sector in sectors:
        sector_data = corr[corr['sector'] == sector].drop(columns=['sector'])
        latex_table += f"\\multicolumn{{2}}{{c}}{{{sector}}} \\\\\n"
        latex_table += sector_data.to_latex(header=True, index=True)
        latex_table += "\\\\\n"
    return latex_table

corr_latex_table = format_corr_table(corr)

# First write — overwritten further down. Kept so an early failure still
# leaves a partial Table 4 file on disk for inspection.
with open('../output/figures/corr_lp_tfp_klems.tex', 'w') as f:
    f.write(corr_latex_table)

# Re-shape to the publication layout: each sector contributes two labelled
# rows (LP from paper, TFP from EU KLEMS) so that the 8x2 table prints with
# clear row labels, no \multicolumn headers needed.
corr_agr = corr.loc[corr.sector =='agr', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_agr.index = ['LP in paper', 'TFP in EUKLEMS']
corr_agr.columns = ['LP in paper', 'TFP in EUKLEMS']

corr_trd = corr.loc[corr.sector =='trd', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_trd.index = ['LP in paper', 'TFP in EUKLEMS']
corr_trd.columns = ['LP in paper', 'TFP in EUKLEMS']

corr_fin = corr.loc[corr.sector =='fin', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_fin.index = ['LP in paper', 'TFP in EUKLEMS']
corr_fin.columns = ['LP in paper', 'TFP in EUKLEMS']

corr_tot = corr.loc[corr.sector =='tot', ['log_diff_LP_paper', 'log_diff_TFP']]
corr_tot.index = ['LP in paper', 'TFP in EUKLEMS']
corr_tot.columns = ['LP in paper', 'TFP in EUKLEMS']

# Vertical stack: agr, trd, fin, tot in the order required by Table 4.
corr = pd.concat([corr_agr, corr_trd, corr_fin, corr_tot], axis=0)


# Final Table 4 file — replaces the earlier write with the cleaned, stacked
# layout. Downstream copying in generate_paper_outputs.py renames this to
# table_4.tex.
with open('../output/figures/corr_lp_tfp_klems.tex', 'w') as f:
    f.write(corr.to_latex())
