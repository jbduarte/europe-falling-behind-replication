import matplotlib
matplotlib.use("Agg")
"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        facts.py
Purpose:     Build the motivating facts of the paper — Figure 1a (EU Big Four labour
             productivity relative to the U.S., 1970-2018) and Figure 1b (sectoral
             employment shares against log GDP per hour for trade, business and
             financial services, U.S. vs. EU-15) — plus the Table 1 shift-share
             decomposition objects (1995-2018).
Pipeline:    Step 13/19 — Generates Figure 1 (motivating facts).
Inputs:      ../data/euklems_2023.csv (sectoral VA, VA_Q, H, P for US + EU-15),
             ../data/raw/OECD_GDP_ph.xlsx (OECD GDP per hour, USD),
             ../data/penn_gdp.xlsx (Penn World Tables GDP).
Outputs:     ../output/figures/fig_1a.pdf, ../output/figures/fig_1b.pdf.
             Shift-share DataFrames (LP_growth_i, growth_effect_i,
             reallocation_effect_i) are built in memory but not exported here.
Dependencies: utils.construct_dataset_facts (helper for per-country panel build).
"""

import matplotlib.pyplot as plt
from .construct_dataset_facts import *
from statsmodels.tsa.filters.hp_filter import hpfilter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

data = pd.read_csv('../data/euklems_2023.csv', index_col=[0, 1]).reset_index()
data.rename(columns={'level_0': 'country', 'level_1': 'year'}, inplace=True, errors='ignore')
# The 2023 EUKLEMS release ships only nominal VA and chained-volume VA_Q; the
# implicit sectoral price deflator P is recovered as the ratio of the two.
data['P'] = data['VA'] / data['VA_Q']
data_lp = pd.read_excel('../data/raw/OECD_GDP_ph.xlsx')

# Loop over the US plus all EU-15 members. EU15 is also kept as its own
# row because EUKLEMS ships a pre-aggregated region with the same name —
# useful as a cross-check on the country-level aggregation built later.
countries = ["US", "EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"]
countries_fulldata = []
for country in countries:
    _, sample_full = construct_dataset_facts(data, smooth=True, country=country)
    countries_fulldata.append(sample_full)

'''
Construct EU BIG 4 Labor Productivity
'''

big4 = ['DEU', 'ITA', 'GBR', 'FRA']
rel_lp = np.empty((len(np.arange(1970, 2019)), 4))

# Relative LP series = (GDP_per_hour_country / GDP_per_hour_USA), then smoothed
# with the Ravn-Uhlig (2002) lambda=6.25 — the annual-data HP-filter constant
# they derive from the standard quarterly value of 1600. Smoothing strips
# business-cycle noise so that Figure 1a shows the long-run convergence
# pattern (1995-2018 European fall-back).
for i, country in enumerate(big4):
    filter = (data_lp["LOCATION"] == country) & (data_lp["MEASURE"] == "USD") & (data_lp["TIME"] <= 2018)
    filter_usa = (data_lp["LOCATION"] == "USA") & (data_lp["MEASURE"] == "USD") & (data_lp["TIME"] <= 2018)
    _, trend = hpfilter(data_lp.loc[filter, "Value"].values / data_lp.loc[filter_usa, "Value"].values, lamb=6.25)
    rel_lp[:,i] = trend

'''
Figure 1a: EU Big Four relative to US
'''
fig, ax = plt.subplots()
ax.plot(np.arange(1970, 2019), np.ones(len(np.arange(1970, 2019))), 'grey', linestyle='dashed')
for i in range(4):
    ax.plot(np.arange(1970, 2019), rel_lp[:,i],
        lw=2,
        label=big4[i])
ax.set_ylabel('GDP per hour relative to U.S.', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
plt.xticks(np.arange(1975, 2020 + 5, 5))
ax.legend()
ax.grid(alpha=0.2)
fig.savefig('../output/figures/fig_1a.pdf', bbox_inches="tight")
plt.close(fig)

'''
Figure 1b: LP growth and the Rise of the Services sector
'''
gdp = pd.read_excel('../data/penn_gdp.xlsx')
gdp['gdp'] = gdp['gdp'] * 1_000_000

markers = ['v', 'o']
fig, ax = plt.subplots()

# Build log GDP per hour for every country in the panel, on a common 1977-2018
# year window. The inner merge guards against year mismatches between the PWT
# GDP series and the EUKLEMS hours panel (some countries enter EUKLEMS later).
gdp_per_hour = {}
for i, c in enumerate(countries):
    tot = countries_fulldata[i].loc[countries_fulldata[i].sector == 'tot', ['year', 'H']].copy()
    tot = tot.rename(columns={'H': 'H_tot'})
    gdp_c = gdp.loc[gdp.country == c, ['year', 'gdp']].copy()
    merged = gdp_c.merge(tot, on='year', how='inner')
    merged = merged[(merged.year >= 1977) & (merged.year <= 2018)].sort_values('year')
    gdp_per_hour[i] = np.log(merged['gdp'].values / merged['H_tot'].values)

# Closure that returns one country's sectoral employment share series on the
# same 1977-2018 grid as gdp_per_hour, so the Figure 1b scatter aligns x and y
# observation-by-observation.
def get_ls(i, sector):
    tot = countries_fulldata[i].loc[countries_fulldata[i].sector == 'tot', ['year']].copy()
    gdp_c = gdp.loc[gdp.country == countries[i], ['year']].copy()
    common_years = gdp_c.merge(tot, on='year', how='inner')
    sec = countries_fulldata[i].loc[countries_fulldata[i].sector == sector, ['year', 'LS']].copy()
    merged = common_years.merge(sec, on='year', how='inner')
    merged = merged[(merged.year >= 1977) & (merged.year <= 2018)].sort_values('year')
    return merged['LS'].values

for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            gdp_per_hour[i], get_ls(i, 'trd'),
            color='red', alpha=1, marker=markers[0], s=15)
    scatter = ax.scatter(
        gdp_per_hour[i], get_ls(i, 'trd'),
        color='red', alpha=0.1, marker=markers[1])
scatter_us.set_label('Wholesale and retail trade: US')
scatter.set_label('Wholesale and retail trade: EU-15')
for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            gdp_per_hour[i], get_ls(i, 'bss'),
            color='blue', alpha=1, marker=markers[0], s=15)
    scatter = ax.scatter(
        gdp_per_hour[i], get_ls(i, 'bss'),
        color='blue', alpha=0.1)
scatter_us.set_label('Business services: US')
scatter.set_label('Business services: EU-15')
for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            gdp_per_hour[i], get_ls(i, 'fin'),
            color='purple', alpha=1, marker=markers[0], s=15)
    scatter = ax.scatter(
        gdp_per_hour[i], get_ls(i, 'fin'),
        color='purple', alpha=0.1)
scatter_us.set_label('Financial services: US')
scatter.set_label('Financial services: EU-15')
ax.set_ylabel('Share in total employment', fontsize=12)
ax.set_xlabel('Log of GDP per hour', fontsize=12)
ax.grid(alpha=0.2)
ax.legend(fontsize=10)
plt.savefig('../output/figures/fig_1b.pdf', bbox_inches="tight")
plt.close(fig)

'''
Table 1: Shift-share decomposition
'''


def annualized(x1, x2, t):
    return ((x2 / x1) ** (1 / t) - 1) * 100


# get LS in 1995 and 2018
sectors = countries_fulldata[0].sector.unique().tolist()
LS_95 = np.empty((2, len(sectors)))
LS_18 = np.empty((2, len(sectors)))
for i in range(2):
    for k, sector in enumerate(sectors):
        LS_95[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 1995), 'LS']
        LS_18[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 2018), 'LS']
LS_data_95 = pd.DataFrame(LS_95[:, :-1], index=['US', 'EU-15'], columns=sectors[:-1])
LS_data_18 = pd.DataFrame(LS_18[:, :-1], index=['US', 'EU-15'], columns=sectors[:-1])

# get A_i in 1995 and 2018
ai_95 = np.empty((2, len(sectors)))
ai_18 = np.empty((2, len(sectors)))
for i in range(2):
    for k, sector in enumerate(sectors):
        ai_95[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 1995), 'L_PROD_normalized']
        ai_18[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 2018), 'L_PROD_normalized']
ai_data_95 = pd.DataFrame(ai_95, index=['US', 'EU-15'], columns=sectors)
ai_data_18 = pd.DataFrame(ai_18, index=['US', 'EU-15'], columns=sectors)

# compute aggregate labor productivity growth
LP_growth = annualized(np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1), np.sum(ai_data_18.iloc[:, :-1] * LS_data_18, axis=1), 2018 - 1995)
LP_growth = pd.DataFrame(LP_growth.values.reshape(1, 2), columns=['US', 'EU-15'])

# Shift-share Decomposition of LP growth
growth_effect = annualized(np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1), np.sum(ai_data_18.iloc[:, :-1] * LS_data_95, axis=1), 2018 - 1995)
growth_effect = pd.DataFrame(growth_effect.values.reshape(1, 2), columns=['US', 'EU-15'])

reallocation_effect = LP_growth - growth_effect

# Sectoral decomposition of LP growth
LP_growth_list = []
growth_effect_list = []
reallocation_effect_list = []

for sector in sectors[:-1]:
    LP_growth_temp = (ai_data_18.loc[:, sector] * LS_data_18.loc[:, sector] - ai_data_95.loc[:, sector] * LS_data_95.loc[:, sector]) / \
                     (np.sum(ai_data_18.iloc[:, :-1] * LS_data_18, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
    LP_growth_temp = pd.DataFrame(LP_growth_temp.values.reshape(1, 2), columns=['US', 'EU-15']) * LP_growth

    growth_effect_temp = (ai_data_18.loc[:, sector] * LS_data_95.loc[:, sector] - ai_data_95.loc[:, sector] * LS_data_95.loc[:, sector]) / \
                         (np.sum(ai_data_18.iloc[:, :-1] * LS_data_95, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
    growth_effect_temp = pd.DataFrame(growth_effect_temp.values.reshape(1, 2), columns=['US', 'EU-15']) * growth_effect

    reallocation_effect_temp = LP_growth_temp - growth_effect_temp

    LP_growth_list.append(LP_growth_temp)
    growth_effect_list.append(growth_effect_temp)
    reallocation_effect_list.append(reallocation_effect_temp)

LP_growth_i = pd.concat(LP_growth_list, ignore_index=True)
growth_effect_i = pd.concat(growth_effect_list, ignore_index=True)
reallocation_effect_i = pd.concat(reallocation_effect_list, ignore_index=True)

LP_growth_i.index = sectors[:-1]
growth_effect_i.index = sectors[:-1]
reallocation_effect_i.index = sectors[:-1]

LP_growth_i = round(LP_growth_i, 2)
growth_effect_i = round(growth_effect_i, 2)
reallocation_effect_i = round(reallocation_effect_i, 2)
