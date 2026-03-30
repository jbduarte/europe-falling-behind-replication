import matplotlib.pyplot as plt
from calibrate import *
from model_labor_shares import *
from construct_dataset import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


def simulate_CF(country, sectors, lps):
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
    for k in sectors:
        sector_filter = sample.sector == k
        sample.loc[sector_filter, 'L_PROD_normalized'] = lps[k].values

    '''4. Simulate model'''
    sample['model_pred'] = sample.groupby('year').apply(lambda x: model_labor_shares(params_EU, x, base_sector=base_sector)).explode().values
    sample_US['model_pred'] = sample_US.groupby('year').apply(lambda x: model_labor_shares(params_US, x, base_sector=base_sector)).explode().values

    def get_A_data(data):
        return (data.LS * data.L_PROD_normalized).sum()

    def get_A_model(data):
        return (data.model_pred * data.L_PROD_normalized).sum()

    A_model = sample.groupby('year').apply(lambda x: get_A_model(x))
    A_data = sample.groupby('year').apply(lambda x: get_A_data(x))
    A_model_US = sample_US.groupby('year').apply(lambda x: get_A_model(x))
    A_data_US = sample_US.groupby('year').apply(lambda x: get_A_data(x))

    rel_A_data = A_data / A_data_US
    rel_A_model = A_model / A_model_US

    return {'data': sample, 'A_data': A_data, 'A_model': A_model, 'rel_A_data': rel_A_data, 'rel_A_model': rel_A_model}
