"""
=======================================================================================
Dataset construction for stylized facts (Figure 1).

Author: Joao B. Duarte
Last Modified: Feb 2026
=======================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def construct_dataset_facts(data, smooth=True, country='US'):
    """
    Take final data, compute value added (constant prices) shares and multiply by GDP of penn tables

    Returns
    DataFrame with labor shares and sectoral labor productivity
    """
    country_filter = data["country"] == country
    data_temp = data.loc[country_filter, :]
    data_temp = data_temp.reset_index(drop=True)
    # Get Penn table GDP
    gdp = pd.read_excel('../data/penn_gdp.xlsx')
    gdp = gdp.loc[gdp.country == country, :]
    gdp = gdp.iloc[1:, :]

    if smooth:
        # Get smoothed series using an HP-Filter
        def hp_filter(x, _lambda=100):
            return sm.tsa.filters.hpfilter(x, _lambda)

        for var in ['VA', 'H', 'P']:
            data_temp[var] = data_temp.groupby('sector')[var].transform(lambda x: hp_filter(x.values)[1])

        gdp['gdp'] = hp_filter(gdp['gdp'].values)[1]

    data_temp['VA_Q'] = data_temp['VA'] / data_temp['P']

    # 1 - Create aggregate data
    data_temp['ws'] = data_temp.groupby('year')['VA_Q'].transform(lambda x: x / x.sum())
    data_temp['P_ws'] = data_temp['ws'] * data_temp['P']
    data_agg = data_temp.groupby('year', as_index=False).agg('sum')
    data_agg['P'] = data_agg['P_ws']
    data_agg['sector'] = 'tot'
    data_agg['country'] = country

    data_temp = pd.concat((data_temp, data_agg), axis=0)

    data_temp = data_temp.merge(gdp, on=['country', 'year'])

    # Compute sectoral output: Value_added_share * GDP
    data_temp['output'] = data_temp['ws'] * data_temp['gdp']

    data_temp['L_PROD'] = data_temp['VA_Q'] / data_temp['H']
    grouped = data_temp.groupby(['sector'])
    data_temp['LS'] = data_temp.groupby(['sector'])['H'].transform(lambda x: x / grouped['H'].get_group('tot').values)

    def get_A(x):
        A = np.ones(len(x))
        for j in range(1, len(x)):
            A[j] = (1 + x.iloc[j]) * A[j - 1]
        return A

        # Get aggregate labor productivity in last year assuming A_i =1 in first period

    data_temp['L_PROD_growth'] = data_temp.groupby('sector')['L_PROD'].transform(lambda x: np.log(x).diff())
    data_temp['L_PROD_normalized'] = data_temp.groupby('sector')['L_PROD_growth'].transform(lambda x: get_A(x))

    sample_full = data_temp.copy()
    sample = sample_full[sample_full.sector == 'tot'][['year', 'L_PROD_normalized', 'H']]

    return sample, sample_full
