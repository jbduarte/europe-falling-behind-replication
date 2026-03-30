"""
=======================================================================================
Description: This program downloads raw data from the Penn World Table release.
Author: Joao B. Duarte (joao.duarte@novasbe.pt)
=======================================================================================
"""

import pandas as pd
import country_converter as coco

data_temp = pd.read_excel('https://www.rug.nl/ggdc/docs/pwt100.xlsx',
                          sheet_name='Data')

data_temp.to_excel('../Data/Raw Data/Penn_table/penn.xlsx', index=False)

iso2_list = coco.convert(names=data_temp['countrycode'].values, to='ISO2', not_found=None)
data_temp.loc[:, 'countrycode'] = iso2_list

# Filter countries we are interested
countries = ["US", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"]

country_filter = data_temp['countrycode'].isin(countries)
data_temp = data_temp[country_filter]

# Create aggregate EU15 GDP
data_EU15 = data_temp[data_temp.countrycode != "US"].groupby('year', as_index=False).agg('sum')
data_EU15['countrycode'] = "EU15"
data_EU15['country'] = "EU15"
data_EU15['currency_unit'] = " "

data_temp = pd.concat((data_temp, data_EU15), axis=0)

data_temp = data_temp[['countrycode', 'year', 'rgdpe']]

data_temp.columns = ['country', 'year', 'gdp']

# Save data
data_temp.to_excel('../Data/Final Data/penn_gdp.xlsx', index=False)
