import pandas as pd
import statsmodels.api as sm
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

'''
-----------
Data
-----------
'''

'KLEMS'
data = pd.read_csv('../Data/Final Data/euklems_2023.csv', index_col=[0, 1])
data.rename(index={'US': 'USA'}, inplace=True)

'Labor Productivity'
data['y_l'] = (data['VA_Q'] / data['H']) * 100

'US data'
data_us = data.loc['USA']

'EU15 data'
data_eu = data.loc['EU15']

'EU4 data'
tot_sector_filter = data.sector == "tot"
EU4_countries = ['DE', "FR", "IT", "GB"]
data_EU4 = pd.DataFrame()
data = data.reset_index()
data_EU4['H'] = data.loc[data.country.isin(EU4_countries), ["sector", "year", "H"]].groupby(["sector", "year"]).agg(sum)
data_EU4['VA_Q'] = data.loc[data.country.isin(EU4_countries), ["sector", "year", "VA_Q"]].groupby(["sector", "year"]).agg(sum)
data_EU4['y_l'] = data_EU4['VA_Q'] / data_EU4['H'] * 100
data_eu4 = data_EU4.copy()
# Report LP annualised growth rates

def annualized(x, start_year=1970):
    _, x = sm.tsa.filters.hpfilter(x, 100)
    return ((x.iloc[-1] / x.loc[start_year]) ** (1 / (len(x.loc[start_year:])-1)) - 1) * 100

print("\nLP growth rates\n")

print("\nUS 1970-1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_us[data_us.sector == sector].loc[:1995, "y_l"]),
              2)
          )

print("\nEU4 1970-1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_eu4.loc[sector].loc[:1995, "y_l"]),
              2)
          )

print("\nEU15 1970-1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_eu[data_eu.sector == sector].loc[:1995, "y_l"]),
              2)
          )


print("\nUS 1995-2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_us[data_us.sector == sector].loc[1995:, "y_l"],
              start_year=1995),
              2)
          )

print("\nEU4 1995-2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_eu4.loc[sector].loc[1995:, "y_l"],
              start_year=1995),
              2)
          )

print("\nEU15 1995-2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(annualized(
              data_eu[data_eu.sector == sector].loc[1995:, "y_l"],
              start_year=1995),
              2)
          )

# Report Employment shares
data_eu4 = data_eu4.reset_index()
data_eu4.index = data_eu4.year

def employment_share(data, sector, year):
    _, h = sm.tsa.filters.hpfilter(data.loc[data.sector==sector, "H"], 100)
    _, h_tot = sm.tsa.filters.hpfilter(data.loc[data.sector == "tot", "H"], 100)
    return h.loc[year]/h_tot.loc[year]

print("\nEMPLOYMENT SHARES\n")

print("\nUS 1970\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_us, sector, 1970), 2)
          )

print("\nEU4 1970\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu4, sector, 1970), 2)
          )

print("\nEU15 1970\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu, sector, 1970), 2)
          )

print("\nUS 1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_us, sector, 1995), 2)
          )

print("\nEU4 1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu4, sector, 1995), 2)
          )

print("\nEU15 1995\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu, sector, 1995), 2)
          )

print("\nUS 2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_us, sector, 2019), 2)
          )

print("\nEU4 2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu4, sector, 2019), 2)
          )

print("\nEU15 2019\n")

for sector in ['tot', 'agr', 'man', 'ser', 'nps', 'prs', 'bss', 'fin', 'trd']:
    print(sector,
          round(employment_share(data_eu, sector, 2019), 2)
          )
