"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        construct_dataset_facts.py
Purpose:     Helper module that builds, for a single country, the harmonised
             sector-level dataset of labour shares and sectoral labour
             productivity used by the stylised-facts script.
Pipeline:    Utility — called by Step 13 (utils.facts).
Inputs:      `data` argument: long-format EUKLEMS table (country, year, sector,
             VA, VA_Q, H, P), plus `../data/penn_gdp.xlsx` for Penn World Tables GDP.
Outputs:     Returns (sample, sample_full) DataFrames in memory; does not write to disk.
Dependencies: None — pure helper invoked by utils.facts.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def construct_dataset_facts(data, smooth=True, country='US'):
    """
    For a given country, compute value-added (constant-price) sectoral shares and
    rescale them by Penn World Tables GDP to obtain sectoral output and labour
    productivity series.

    Returns
    -------
    sample : DataFrame
        Aggregate ('tot') series with year, normalized labor productivity, and total hours.
    sample_full : DataFrame
        Full sector-level panel (including the constructed 'tot' aggregate).
    """
    country_filter = data["country"] == country
    data_temp = data.loc[country_filter, :]
    data_temp = data_temp.reset_index(drop=True)
    # Penn World Tables GDP runs one year earlier than the EUKLEMS panel; drop
    # the leading row so the subsequent merge on (country, year) is balanced.
    gdp = pd.read_excel('../data/penn_gdp.xlsx')
    gdp = gdp.loc[gdp.country == country, :]
    gdp = gdp.iloc[1:, :]

    if smooth:
        # Smooth VA, H and P with the Hodrick-Prescott filter at lambda=100,
        # the standard choice for annual frequency. Smoothing is applied
        # within sector so that low-frequency growth components are extracted
        # before computing labour productivity ratios further down.
        def hp_filter(x, _lambda=100):
            return sm.tsa.filters.hpfilter(x, _lambda)

        for var in ['VA', 'H', 'P']:
            data_temp[var] = data_temp.groupby('sector')[var].transform(lambda x: hp_filter(x.values)[1])

        gdp['gdp'] = hp_filter(gdp['gdp'].values)[1]

    # Real (constant-price) sectoral value added.
    data_temp['VA_Q'] = data_temp['VA'] / data_temp['P']

    # ── Construct the 'tot' aggregate via value-added shares ──
    # ws_t = VA_Q_{i,t} / sum_i VA_Q_{i,t} so that ws sums to 1 each year.
    # P_ws is the share-weighted sectoral price; summing across sectors gives
    # an implicit aggregate deflator (Tornqvist-style chained construction).
    data_temp['ws'] = data_temp.groupby('year')['VA_Q'].transform(lambda x: x / x.sum())
    data_temp['P_ws'] = data_temp['ws'] * data_temp['P']
    data_agg = data_temp.groupby('year', as_index=False).agg('sum')
    data_agg['P'] = data_agg['P_ws']
    data_agg['sector'] = 'tot'
    data_agg['country'] = country

    # Some EUKLEMS releases ship a pre-computed 'tot' sector. Strip it here
    # so it is replaced by the share-weighted aggregate built above (avoids
    # double counting and ensures the deflator definition is internally consistent).
    data_temp = data_temp[data_temp['sector'] != 'tot']
    data_temp = pd.concat((data_temp, data_agg), axis=0)

    data_temp = data_temp.merge(gdp, on=['country', 'year'])

    # Rescale value-added shares by Penn World Tables GDP to express each
    # sector in PWT current dollars (cross-country comparable in Figure 1b).
    data_temp['output'] = data_temp['ws'] * data_temp['gdp']

    # Sector labour productivity in real VA terms; labour share = sector hours
    # over total hours (computed from the 'tot' aggregate just constructed).
    data_temp['L_PROD'] = data_temp['VA_Q'] / data_temp['H']
    tot_hours = data_temp.loc[data_temp['sector'] == 'tot', ['year', 'H']].rename(columns={'H': 'H_tot'})
    data_temp = data_temp.merge(tot_hours, on='year', how='left')
    data_temp['LS'] = data_temp['H'] / data_temp['H_tot']
    data_temp = data_temp.drop(columns=['H_tot'])

    # Build a cumulative log-growth index normalised to 1 in the base year.
    # Done iteratively rather than via cumprod so that the base year (j=0)
    # always anchors at exactly 1.0, even when the first growth observation
    # is NaN due to the .diff() above.
    def get_A(x):
        A = np.ones(len(x))
        for j in range(1, len(x)):
            A[j] = (1 + x.iloc[j]) * A[j - 1]
        return A

    data_temp['L_PROD_growth'] = data_temp.groupby('sector')['L_PROD'].transform(lambda x: np.log(x).diff())
    data_temp['L_PROD_normalized'] = data_temp.groupby('sector')['L_PROD_growth'].transform(lambda x: get_A(x))

    sample_full = data_temp.copy()
    sample = sample_full[sample_full.sector == 'tot'][['year', 'L_PROD_normalized', 'H']]

    return sample, sample_full
