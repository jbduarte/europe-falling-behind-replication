import pandas as pd
import statsmodels.api as sm

def get_initial_rel_A():
    release = "2021"
    data = pd.read_csv('../Data/Final Data/euklems_' + release + '.csv')
    gdp = pd.read_excel('../Data/Final Data/penn_gdp.xlsx')

    gdp = gdp[gdp.year != 1950]

    data_temp = data.merge(gdp, on=['country', 'year'])

    H_1977 = data_temp.loc[data_temp.year == 1977, ['country', 'sector', 'H']].groupby('country').agg(sum)
    GDP_1977 = data_temp.loc[data_temp.year == 1977, ['country', 'gdp']].groupby('country').agg(lambda x: x.iloc[0])

    L_PROD = GDP_1977 / H_1977.values
    rel_A_level = L_PROD / L_PROD.loc["US"]

    return rel_A_level.iloc[:-1]
