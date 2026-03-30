import matplotlib.pyplot as plt
import pandas as pd

from calibrate import *
from model_labor_shares import *
from construct_dataset import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


def simulate_extrapolate(country):
    """
    Given a country name, get data from that country and simulate model
    sectors: list of sectors
    lps: DataFrame T by N of labor productivity
    Returns
    A
    rel_A
    dataframe with data labor shares and model predictions for labor shares
    """
    base_sector = 'man'
    release = "2021"
    data = pd.read_csv('../Data/Final Data/euklems_' + release + '.csv')

    '''1. Load Parameters'''
    f = open('../Outputs/Parameters/params_last_period.json')
    params_US = json.load(f)
    f.close()

    '''2. Read data'''
    sample = construct_dataset(data, smooth=True, country=country)
    sample = sample.sort_values(by=['year', 'sector'])
    sample_US = construct_dataset(data, smooth=True, country='US')
    sample_US = sample_US.sort_values(by=['year', 'sector'])

    '''3. Get Omegas'''
    data_first_year = sample[(sample['year'] == sample['year'].unique()[0])]
    omegas = data_first_year.loc[:, ['sector', 'LS']]
    omegas = omegas.set_index('sector')

    params_EU = params_US.copy()
    params_EU['omegas'] = omegas.values.tolist()

    '''3. Replace labor productivity in sectors with counterfactual'''
    sample_US['gr'] = sample_US.groupby('sector')['L_PROD_normalized'].transform(lambda x: np.log(x).diff())
    growth_rates_US = sample_US[sample_US.year >= 2010].groupby('sector')['gr'].agg('mean')
    A_2018_US = sample_US[sample_US.year == 2018]['L_PROD_normalized']

    data_extrapolate_US = pd.DataFrame({'year': [2018]*12, 'sector': sample_US.sector.unique(), 'L_PROD_normalized': A_2018_US.values})

    for t in range(1, 33):
        data_extrapolate_US = data_extrapolate_US.append(pd.DataFrame({'year': [2018 + t]*12, 'sector': sample_US.sector.unique(), 'L_PROD_normalized': A_2018_US.values * (
            1+growth_rates_US.values)**t}))

    sample['gr'] = sample.groupby('sector')['L_PROD_normalized'].transform(lambda x: np.log(x).diff())
    growth_rates = sample[sample.year >= 2010].groupby('sector')['gr'].agg('mean')
    A_2018 = sample[sample.year == 2018]['L_PROD_normalized']

    data_extrapolate = pd.DataFrame({'year': [2018] * 12, 'sector': sample.sector.unique(), 'L_PROD_normalized': A_2018.values})

    for t in range(1, 33):
        data_extrapolate = data_extrapolate.append(pd.DataFrame({'year': [2018 + t] * 12, 'sector': sample.sector.unique(), 'L_PROD_normalized': A_2018.values * (
                1 + growth_rates.values) ** t}))

    '''4. Simulate model'''
    data_extrapolate['model_pred'] = data_extrapolate.groupby('year').apply(lambda x: model_labor_shares(params_EU, x, base_sector=base_sector)).explode().values
    data_extrapolate_US['model_pred'] = data_extrapolate_US.groupby('year').apply(lambda x: model_labor_shares(params_US, x, base_sector=base_sector)).explode().values

    def get_A_model(data):
        return (data.model_pred * data.L_PROD_normalized).sum()

    A_model = data_extrapolate.groupby('year').apply(lambda x: get_A_model(x))
    A_model_US = data_extrapolate_US.groupby('year').apply(lambda x: get_A_model(x))
    rel_A_model = A_model / A_model_US

    return {'data': data_extrapolate, 'A_model': A_model, 'rel_A_model': rel_A_model}
