import pandas as pd
import numpy as np

data = pd.read_csv('../Data/Final Data/euklems_2023.csv', index_col=[0, 1])
data.rename(index={'AT': 'AUT'}, inplace=True)
data.rename(index={'BE': 'BEL'}, inplace=True)
data.rename(index={'DE': 'DEU'}, inplace=True)
data.rename(index={'DK': 'DNK'}, inplace=True)
data.rename(index={'ES': 'ESP'}, inplace=True)
data.rename(index={'FI': 'FIN'}, inplace=True)
data.rename(index={'FR': 'FRA'}, inplace=True)
data.rename(index={'GB': 'GBR'}, inplace=True)
data.rename(index={'GR': 'GRC'}, inplace=True)
data.rename(index={'IE': 'IRL'}, inplace=True)
data.rename(index={'IT': 'ITA'}, inplace=True)
data.rename(index={'LU': 'LUX'}, inplace=True)
data.rename(index={'NL': 'NLD'}, inplace=True)
data.rename(index={'PT': 'PRT'}, inplace=True)
data.rename(index={'SE': 'SWE'}, inplace=True)
data.rename(index={'US': 'USA'}, inplace=True)

data = data.loc[data.sector == "tot"]
data = data.reset_index()
data.year = data.year.astype(int)

'OECD'
GDP_ph = pd.read_excel('../Data/Raw Data/OECD_GDP_ph.xlsx', index_col=[0, 5],
                       engine='openpyxl')  # Measured in USD (constant prices 2010 and PPPs).
GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']
GDP_ph = GDP_ph.reset_index()
GDP_ph['country'] = GDP_ph['LOCATION']
GDP_ph['year'] = GDP_ph['TIME']

data_comb = pd.merge(data, GDP_ph, on=['country', 'year'])
data_comb['GDP'] = data_comb['Value'] * data_comb['H']

data_comb = data_comb[data_comb.country != "USA"]

data_EU15 = data_comb.groupby(["year"]).agg(sum)

data_EU15["GDP_ph"] = data_EU15["GDP"] / data_EU15["H"]

GDP_ph = pd.read_excel('../Data/Raw Data/OECD_GDP_ph.xlsx',
                       engine='openpyxl')  # Measured in USD (constant prices 2010 and PPPs).

eu15 = pd.DataFrame(columns=GDP_ph.columns)
eu15["Value"] = data_EU15["GDP_ph"]
eu15["TIME"] = np.arange(1970, 2020)
eu15["LOCATION"] = "EU15"
eu15["MEASURE"] = "USD"

GDP_ph = pd.concat((GDP_ph, eu15), axis=0)

GDP_ph.to_excel('../Data/Raw Data/OECD_GDP_ph_EU15.xlsx', index=False)
