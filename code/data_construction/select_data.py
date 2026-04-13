"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:    select_data.py
Purpose: Filter raw EUKLEMS releases (2009/2017/2021/2023) to the EU15+US
         sample and collapse the NACE Rev.2 industries into the paper's six
         structural sectors (agr, man, trd, bss, fin, nps; plus tot, ser,
         prs composites), writing the cleaned country-sector-year panel to
         data/euklems_<release>.csv.
Pipeline:    Data-build chain (raw -> data/euklems_<release>.csv). The main
             pipeline reads data/euklems_2023.csv directly, so this is the
             key transformation step from raw download to analysis panel.

Build stages (2023 branch, the one the paper uses):
  1. Load data/raw_data/EUKLEMS_2023/data_raw.csv, remap UK->GB and EL->GR.
  2. Apply the six-sector mapping (sectors_dict below) and filter to EU15+US.
  3. Aggregate industries into the six sectors by summing VA_CP, H_EMP, VA_Q.
  4. If sample_period[0] < 1995, back-extrapolate VA/H/VA_Q to 1970 using
     (a) EUKLEMS 2009 release growth rates for 1970-1994 for all countries,
     (b) World KLEMS 2013 U.S. release for the pre-1977 U.S. extension.
  5. Append 'ser' (all services) and 'prs' (progressive services:
     services excluding nps) composite rows for convenience.

Sector taxonomy (paper's six sectors):
  agr = agriculture (NACE A)
  man = manufacturing + mining + construction + utilities (NACE B/C/D-E/F)
  trd = trade (NACE G)
  bss = business services (NACE J + M-N)
  fin = finance (NACE K)
  nps = non-progressive services (NACE H/I/L/O/P/Q/R-S/T)
"""

import pandas as pd
import country_converter as coco
import numpy as np

def select_data(countries=("US", "EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"),
                sample_period=(1997, 2018),
                release="2021"):
    """Build the country-sector-year analysis panel for a given EUKLEMS
    release.

    Parameters
    ----------
    countries : iterable of str
        ISO2 country codes to retain (EU15 members + US + EU15 aggregate).
    sample_period : tuple (start, end)
        Year range. If start < the release's first year, the function
        back-extrapolates via the 2009 release + World KLEMS U.S.
    release : {"2009", "2017", "2021", "2023"}
        Which EUKLEMS release to process.
    """

    if release == "2023":
        print('Building', release, 'data ...')
        file_path = "../data/raw_data/EUKLEMS_" + release + '/data_raw.csv'
        data_temp = pd.read_csv(file_path)

        # Six-sector NACE Rev.2 mapping. The commented-out richer taxonomy
        # (rst/trs/res/gov/edu/hlt/per) is the granular variant we used for
        # internal checks; the paper collapses all of H/I/L/O-T into the
        # single 'nps' (non-progressive services) aggregate.
        # sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "rst", "H": "trs",
        #                 "J": "bss", "K": "fin", "L": "res", "M-N": "bss", "O": "gov",
        #                 "P": "edu", "Q": "hlt", "R-S": "per", "T": "per"}
        sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "nps", "H": "nps",
                        "J": "bss", "K": "fin", "L": "nps", "M-N": "bss", "O": "nps",
                        "P": "nps", "Q": "nps", "R-S": "nps", "T": "nps", "TOT": "tot"}

        # EUKLEMS uses Eurostat country codes (UK, EL) while downstream code
        # expects ISO2 (GB, GR).
        data_temp.loc[data_temp.geo_code == 'UK', 'geo_code'] = 'GB'
        data_temp.loc[data_temp.geo_code == 'EL', 'geo_code'] = 'GR'

        country_filter = data_temp['geo_code'].isin(countries)
        year_filter = (data_temp['year'] >= sample_period[0]) & (data_temp['year'] <= sample_period[1])

        data_temp['sector'] = data_temp['nace_r2_code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
        sector_filter = data_temp['sector'].isin(sectors_dict.values())

        data_temp = data_temp[country_filter & sector_filter & year_filter]  # Select data based on filter

        # # Create weights of each sector in value added for sectors classification so that we can create price indexes
        # data_temp['VA_CP_sector'] = data_temp.groupby(['geo_code', 'year', 'sector'])['VA_CP'].transform('sum')
        # data_temp['ws'] = data_temp['VA_CP'] / data_temp['VA_CP_sector']
        # data_temp['VA_PI_ws'] = data_temp['VA_PI'] * data_temp['ws']

        data_temp = data_temp.groupby(["geo_code", "year", "sector"]).aggregate({'VA_CP': 'sum', 'H_EMP': 'sum', 'VA_Q': 'sum'})
        data_temp = data_temp.reset_index()
        data_temp.columns = ["country", "year", "sector", "VA", "H", "VA_Q"]

        if sample_period[0] >= 1995:
            data_temp.to_csv('../data/euklems_' + release + '.csv', index=False)

        else:
            # use 2009 release to extrapolate backwards to 1977 using growth rates
            data_2009 = pd.read_csv("../data/euklems_2009.csv")
            data_2009 = data_2009[data_2009.sector != "com"]
            data_2009_growth = data_2009.groupby(["country", "sector"], as_index=False).transform(lambda x: np.log(x).diff())
            data_2009_growth = data_2009_growth.loc[:, "VA":]
            data_2009_growth = data_2009_growth.merge(data_2009[["country", "year", "sector"]], on=data_2009.index)

            outer_data = data_temp.merge(data_2009_growth, on=["country", "year", "sector"], how="outer")
            outer_data = outer_data.sort_values(by=['country', 'sector', 'year'])

            sectors = outer_data.sector.unique()
            countries = outer_data.country.unique()
            for i in countries:
                for j in sectors:
                    country_filter = outer_data.country == i
                    sector_filter = outer_data.sector == j
                    init_filter = outer_data.year == 1995
                    periods = len(outer_data[outer_data.year <= 1995].year.unique())

                    H_0 = outer_data.loc[country_filter & sector_filter & init_filter, "H_x"].values[0]
                    VA_0 = outer_data.loc[country_filter & sector_filter & init_filter, "VA_x"].values[0]
                    VA_Q_0 = outer_data.loc[country_filter & sector_filter & init_filter, "VA_Q_x"].values[0]

                    VA_extended = np.zeros(periods)
                    VA_Q_extended = np.zeros(periods)
                    H_extended = np.zeros(periods)

                    VA_extended[-1] = VA_0
                    VA_Q_extended[-1] = VA_Q_0
                    H_extended[-1] = H_0

                    VA_Q_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1995), "VA_Q_y"].values
                    VA_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1995), "VA_y"].values
                    H_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1995), "H_y"].values

                    for t in range(periods-1, 0, -1):
                        VA_extended[t - 1] = VA_extended[t] / (1 + VA_g[t])
                        VA_Q_extended[t - 1] = VA_Q_extended[t] / (1 + VA_Q_g[t])
                        H_extended[t - 1] = H_extended[t] / (1 + H_g[t])
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1995), "VA_x"] = VA_extended[:-1]
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1995), "VA_Q_x"] = VA_Q_extended[:-1]
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1995), "H_x"] = H_extended[:-1]

            final_data = outer_data.loc[:, :"VA_Q_x"]
            final_data.columns = ['country', 'year', 'sector', 'VA', 'H', 'VA_Q']

            ## Get first 7 years of US data with World KLEMS data
            usa_world_klems_data = pd.read_csv("../data/raw_data/World_KLEMS_2013/data_raw_usa.csv")
            usa_world_klems_data = usa_world_klems_data.melt(id_vars=['Variable', 'desc', 'code'])
            usa_world_klems_data['variable'] = usa_world_klems_data['variable'].apply(lambda x: x[1:])
            usa_world_klems_data['year'] = usa_world_klems_data['variable'].astype(int)

            sectors_dict_worldKLEMS = {"AtB": 'agr', "C": "man", "D": "man", "E": "man", "F": "man", "G": "trd",
                                       "H": "nps",
                                       "60t63": "nps",
                                       "64": "bss", "J": "fin", "70": "nps", "71t74": "bss", "L": "nps",
                                       "M": "nps", "N": "nps", "O": "nps", "P": "nps", "TOT": "tot"}

            usa_world_klems_data = usa_world_klems_data[['code', 'year', 'Variable', 'value']]
            usa_world_klems_data.index = pd.MultiIndex.from_arrays(usa_world_klems_data[['code', 'year', 'Variable']].values.T, names=['sector', 'year', 'variable'])
            usa_world_klems_data = usa_world_klems_data['value'].unstack()
            usa_world_klems_data = usa_world_klems_data[["VA", "H_EMP", "VA_P"]]
            usa_world_klems_data["VA_Q"] = usa_world_klems_data["VA"] / usa_world_klems_data["VA_P"]
            usa_world_klems_data = usa_world_klems_data[["VA", "H_EMP", "VA_Q"]]
            usa_world_klems_data = usa_world_klems_data.reset_index()
            year_filter = (usa_world_klems_data['year'] >= sample_period[0]) & (
                        usa_world_klems_data['year'] <= sample_period[1])
            usa_world_klems_data['sector'] = usa_world_klems_data['sector'].apply(
                lambda x: sectors_dict_worldKLEMS[x] if x in sectors_dict_worldKLEMS.keys() else x)
            sector_filter = usa_world_klems_data['sector'].isin(sectors_dict_worldKLEMS.values())
            usa_world_klems_data = usa_world_klems_data.loc[sector_filter & year_filter]  # Select data based on filter


            # # Create weights of each sector in value added for sectors classification so that we can create price indexes
            # usa_world_klems_data['VA_CP_sector'] = usa_world_klems_data.groupby(['year', 'sector'])['VA'].transform('sum')
            # usa_world_klems_data['ws'] = usa_world_klems_data['VA'] / usa_world_klems_data['VA_CP_sector']
            # usa_world_klems_data['VA_P_ws'] = usa_world_klems_data['VA_P'] * usa_world_klems_data['ws']

            usa_world_klems_data = usa_world_klems_data.groupby(["year", "sector"]).aggregate(
                {'VA': 'sum', 'H_EMP': 'sum', 'VA_Q': 'sum'})
            usa_world_klems_data = usa_world_klems_data.reset_index()
            usa_world_klems_data.columns = ["year", "sector", "VA", "H", "VA_Q"]
            usa_world_klems_data = usa_world_klems_data.sort_values(['sector', 'year'])

            data_us_growth = usa_world_klems_data.groupby(["sector"], as_index=False).transform(
                lambda x: np.log(x).diff())

            data_us_growth = data_us_growth.loc[:, "VA":]
            data_us_growth = data_us_growth.merge(usa_world_klems_data[["year", "sector"]], on=usa_world_klems_data.index)
            data_us_growth = data_us_growth.loc[data_us_growth['year'] <= 1977]

            for j in sectors:
                VA_extended = np.zeros(8)
                VA_Q_extended = np.zeros(8)
                H_extended = np.zeros(8)

                VA_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] == 1977) & \
                               (final_data['sector'] == j), "VA"]
                VA_Q_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                                                 (final_data['year'] == 1977) & \
                                                 (final_data['sector'] == j), "VA_Q"]
                H_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] == 1977) & \
                               (final_data['sector'] == j), "H"]

                sector_filter = data_us_growth['sector'] == j
                VA_g = data_us_growth.loc[sector_filter, "VA"].values
                VA_Q_g = data_us_growth.loc[sector_filter, "VA_Q"].values
                H_g = data_us_growth.loc[sector_filter, "H"].values

                for t in range(8 - 1, 0, -1):
                    VA_extended[t - 1] = VA_extended[t] / (1 + VA_g[t])
                    VA_Q_extended[t - 1] = VA_Q_extended[t] / (1 + VA_Q_g[t])
                    H_extended[t - 1] = H_extended[t] / (1 + H_g[t])

                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "VA"] = VA_extended[:-1]
                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "VA_Q"] = VA_Q_extended[:-1]
                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "H"] = H_extended[:-1]

            # # compute Total
            # data_total = final_data.loc[:, ['country', 'year', 'VA', 'H', 'VA_Q']].groupby(
            #     ["country", "year"]).aggregate(sum)
            # data_total = data_total.reset_index()
            # data_total.columns = ['country', 'year', 'VA', 'H', 'VA_Q']
            # data_total['sector'] = "tot"
            # data_total = data_total[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            # final_data = final_data[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            # final_data = pd.concat((final_data, data_total), axis=0)
            # final_data = final_data.sort_values(['country', 'sector', 'year'])

            # compute services total
            sector_filter = (final_data.sector != 'agr') & (final_data.sector != 'man') & (final_data.sector != 'tot')
            final_data['VA_services'] = final_data.loc[sector_filter].groupby(['country', 'year'])['VA_Q'].transform('sum')

            data_services = final_data.loc[sector_filter, ['country', 'year', 'VA', 'H', 'VA_Q']].groupby(["country", "year"]).aggregate(sum)
            data_services = data_services.reset_index()
            data_services.columns = ['country', 'year', 'VA', 'H', 'VA_Q']
            data_services['sector'] = "ser"
            data_services = data_services[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            final_data = final_data[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            final_data = pd.concat((final_data, data_services), axis=0)
            final_data = final_data.sort_values(['country', 'sector', 'year'])

            # compute progressive services total
            sector_filter = (final_data.sector != 'agr') & (final_data.sector != 'man') & (final_data.sector != 'nps') & (final_data.sector != 'ser') & (final_data.sector != 'tot')
            final_data['VA_services'] = final_data.loc[sector_filter].groupby(['country', 'year'])['VA_Q'].transform(
                'sum')

            data_services = final_data.loc[sector_filter, ['country', 'year', 'VA', 'H', 'VA_Q']].groupby(
                ["country", "year"]).aggregate(sum)
            data_services = data_services.reset_index()
            data_services.columns = ['country', 'year', 'VA', 'H', 'VA_Q']
            data_services['sector'] = "prs"
            data_services = data_services[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            final_data = final_data[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
            final_data = pd.concat((final_data, data_services), axis=0)
            final_data = final_data.sort_values(['country', 'sector', 'year'])

            final_data.to_csv('../data/euklems_' + release + '.csv', index=False)
        print('\n ...Done!')

    elif release == "2021":
        print('Building', release, 'data ...')
        file_path = "../data/raw_data/EUKLEMS_" + release + '/data_raw.csv'
        data_temp = pd.read_csv(file_path)

        # Dictionaries
        # sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "rst", "H": "trs",
        #                 "J": "bss", "K": "fin", "L": "res", "M-N": "bss", "O": "gov",
        #                 "P": "edu", "Q": "hlt", "R-S": "per", "T": "per"}
        sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "nps", "H": "nps",
                        "J": "bss", "K": "fin", "L": "nps", "M-N": "bss", "O": "nps",
                        "P": "nps", "Q": "nps", "R-S": "nps", "T": "nps", "TOT": "tot"}

        data_temp.loc[data_temp.geo_code == 'UK', 'geo_code'] = 'GB'
        data_temp.loc[data_temp.geo_code == 'EL', 'geo_code'] = 'GR'

        country_filter = data_temp['geo_code'].isin(countries)
        year_filter = (data_temp['year'] >= sample_period[0]) & (data_temp['year'] <= sample_period[1])

        data_temp['sector'] = data_temp['nace_r2_code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
        sector_filter = data_temp['sector'].isin(sectors_dict.values())

        data_temp = data_temp[country_filter & sector_filter & year_filter]  # Select data based on filter

        # Create weights of each sector in value added for sectors classification so that we can create price indexes
        data_temp['VA_CP_sector'] = data_temp.groupby(['geo_code', 'year', 'sector'])['VA_CP'].transform('sum')
        data_temp['ws'] = data_temp['VA_CP'] / data_temp['VA_CP_sector']
        data_temp['VA_PI_ws'] = data_temp['VA_PI'] * data_temp['ws']

        data_temp = data_temp.groupby(["geo_code", "year", "sector"]).aggregate({'VA_CP': 'sum', 'H_EMP': 'sum', 'VA_PI_ws': 'sum'})
        data_temp = data_temp.reset_index()
        data_temp.columns = ["country", "year", "sector", "VA", "H", "P"]

        if sample_period[0] >= 1997:
            data_temp.to_csv('../data/euklems_' + release + '.csv', index=False)

        if sample_period[0] < 1997:
            # use 2009 release to extrolate backwards to 1977 using growth rates
            data_2009 = pd.read_csv("../data/euklems_2009.csv")
            data_2009 = data_2009[data_2009.sector != "com"]
            data_2009_growth = data_2009.groupby(["country", "sector"], as_index=False).transform(lambda x: np.log(x).diff())
            data_2009_growth = data_2009_growth.loc[:, "VA":]
            data_2009_growth = data_2009_growth.merge(data_2009[["country", "year", "sector"]], on=data_2009.index)

            outer_data = data_temp.merge(data_2009_growth, on=["country", "year", "sector"], how="outer")
            outer_data = outer_data.sort_values(by=['country', 'sector', 'year'])

            sectors = outer_data.sector.unique()
            countries = outer_data.country.unique()
            for i in countries:
                for j in sectors:
                    country_filter = outer_data.country == i
                    sector_filter = outer_data.sector == j
                    init_filter = outer_data.year == 1997
                    periods = len(outer_data[outer_data.year <= 1997].year.unique())

                    P_0 = outer_data.loc[country_filter & sector_filter & init_filter, "P_x"].values[0]
                    H_0 = outer_data.loc[country_filter & sector_filter & init_filter, "H_x"].values[0]
                    VA_0 = outer_data.loc[country_filter & sector_filter & init_filter, "VA_x"].values[0]

                    VA_extended = np.zeros(periods)
                    P_extended = np.zeros(periods)
                    H_extended = np.zeros(periods)

                    VA_extended[-1] = VA_0
                    P_extended[-1] = P_0
                    H_extended[-1] = H_0

                    VA_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1997), "VA_y"].values
                    P_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1997), "P_y"].values
                    H_g = outer_data.loc[country_filter & sector_filter & (outer_data.year <= 1997), "H_y"].values

                    for t in range(periods-1, 0, -1):
                        VA_extended[t - 1] = VA_extended[t] / (1 + VA_g[t])
                        P_extended[t - 1] = P_extended[t] / (1 + P_g[t])
                        H_extended[t - 1] = H_extended[t] / (1 + H_g[t])
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1997), "VA_x"] = VA_extended[:-1]
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1997), "P_x"] = P_extended[:-1]
                    outer_data.loc[country_filter & sector_filter & (outer_data.year < 1997), "H_x"] = H_extended[:-1]

            final_data = outer_data.loc[:, :"P_x"]
            final_data.columns = ['country', 'year', 'sector', 'VA', 'H', 'P']

            ## Get first 7 years of US data with World KLEMS data
            usa_world_klems_data = pd.read_csv("../data/raw_data/World_KLEMS_2013/data_raw_usa.csv")
            usa_world_klems_data = usa_world_klems_data.melt(id_vars=['Variable', 'desc', 'code'])
            usa_world_klems_data['variable'] = usa_world_klems_data['variable'].apply(lambda x: x[1:])
            usa_world_klems_data['year'] = usa_world_klems_data['variable'].astype(int)

            sectors_dict_worldKLEMS = {"AtB": 'agr', "C": "man", "D": "man", "E": "man", "F": "man", "G": "trd",
                                       "H": "nps",
                                       "60t63": "nps",
                                       "64": "bss", "J": "fin", "70": "nps", "71t74": "bss", "L": "nps",
                                       "M": "nps", "N": "nps", "O": "nps", "P": "nps", "TOT": "tot"}

            usa_world_klems_data = usa_world_klems_data[['code', 'year', 'Variable', 'value']]
            usa_world_klems_data.index = pd.MultiIndex.from_arrays(usa_world_klems_data[['code', 'year', 'Variable']].values.T, names=['sector', 'year', 'variable'])
            usa_world_klems_data = usa_world_klems_data['value'].unstack()
            usa_world_klems_data = usa_world_klems_data[["VA", "H_EMP", "VA_P"]]
            usa_world_klems_data = usa_world_klems_data.reset_index()
            year_filter = (usa_world_klems_data['year'] >= sample_period[0]) & (
                        usa_world_klems_data['year'] <= sample_period[1])
            usa_world_klems_data['sector'] = usa_world_klems_data['sector'].apply(
                lambda x: sectors_dict_worldKLEMS[x] if x in sectors_dict_worldKLEMS.keys() else x)
            sector_filter = usa_world_klems_data['sector'].isin(sectors_dict_worldKLEMS.values())
            usa_world_klems_data = usa_world_klems_data.loc[sector_filter & year_filter]  # Select data based on filter

            # Create weights of each sector in value added for sectors classification so that we can create price indexes
            # usa_world_klems_data['VA_CP_sector'] = usa_world_klems_data.groupby(['year', 'sector'])['VA'].transform('sum')
            # usa_world_klems_data['ws'] = usa_world_klems_data['VA'] / usa_world_klems_data['VA_CP_sector']
            # usa_world_klems_data['VA_P_ws'] = usa_world_klems_data['VA_P'] * usa_world_klems_data['ws']

            usa_world_klems_data = usa_world_klems_data.groupby(["year", "sector"]).aggregate(
                {'VA': 'sum', 'H_EMP': 'sum', 'VA_QI': 'sum'})
            usa_world_klems_data = usa_world_klems_data.reset_index()
            usa_world_klems_data.columns = ["year", "sector", "VA", "H", "VA_Q"]
            usa_world_klems_data = usa_world_klems_data.sort_values(['sector', 'year'])

            data_us_growth = usa_world_klems_data.groupby(["sector"], as_index=False).transform(
                lambda x: np.log(x).diff())

            data_us_growth = data_us_growth.loc[:, "VA":]
            data_us_growth = data_us_growth.merge(usa_world_klems_data[["year", "sector"]], on=usa_world_klems_data.index)
            data_us_growth = data_us_growth.loc[data_us_growth['year'] <= 1977]

            for j in sectors:
                VA_extended = np.zeros(8)
                P_extended = np.zeros(8)
                H_extended = np.zeros(8)

                VA_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] == 1977) & \
                               (final_data['sector'] == j), "VA"]
                P_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] == 1977) & \
                               (final_data['sector'] == j), "P"]
                H_extended[-1] = final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] == 1977) & \
                               (final_data['sector'] == j), "H"]

                sector_filter = data_us_growth['sector'] == j
                VA_g = data_us_growth.loc[sector_filter, "VA"].values
                P_g = data_us_growth.loc[sector_filter, "P"].values
                H_g = data_us_growth.loc[sector_filter, "H"].values

                for t in range(8 - 1, 0, -1):
                    VA_extended[t - 1] = VA_extended[t] / (1 + VA_g[t])
                    P_extended[t - 1] = P_extended[t] / (1 + P_g[t])
                    H_extended[t - 1] = H_extended[t] / (1 + H_g[t])

                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "VA"] = VA_extended[:-1]
                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "P"] = P_extended[:-1]
                final_data.loc[(final_data['country'] == 'US') & \
                               (final_data['year'] <= 1976) & \
                               (final_data['sector'] == j), "H"] = H_extended[:-1]

        # compute Total
        # final_data['VA_services'] = final_data.groupby(['country', 'year'])['VA'].transform('sum')
        # final_data['ws'] = final_data['VA'] / final_data['VA_services']
        # final_data['P_ws'] = final_data['P'] * final_data['ws']
        #
        # data_total = final_data.loc[:, ['country', 'year', 'VA', 'H', 'P_ws']].groupby(
        #     ["country", "year"]).aggregate(sum)
        # data_total = data_total.reset_index()
        # data_total.columns = ['country', 'year', 'VA', 'H', 'P']
        # data_total['sector'] = "tot"
        # data_total = data_total[['country', 'year', 'sector', 'VA', 'H', 'P']]
        # final_data = final_data[['country', 'year', 'sector', 'VA', 'H', 'P']]
        # final_data = pd.concat((final_data, data_total), axis=0)
        # final_data = final_data.sort_values(['country', 'sector', 'year'])

        # compute services total
        sector_filter = (final_data.sector != 'agr') & (final_data.sector != 'man') & (final_data.sector != 'tot')
        final_data['VA_services'] = final_data.loc[sector_filter].groupby(['country', 'year'])['VA'].transform('sum')
        final_data['ws'] = final_data['VA'] / final_data['VA_services']
        final_data['P_ws'] = final_data['P'] * final_data['ws']

        data_services = final_data.loc[sector_filter, ['country', 'year', 'VA', 'H', 'P_ws']].groupby(["country", "year"]).aggregate(sum)
        data_services = data_services.reset_index()
        data_services.columns = ['country', 'year', 'VA', 'H', 'P']
        data_services['sector'] = "ser"
        data_services = data_services[['country', 'year', 'sector', 'VA', 'H', 'P']]
        final_data = final_data[['country', 'year', 'sector', 'VA', 'H', 'P']]
        final_data = pd.concat((final_data, data_services), axis=0)
        final_data = final_data.sort_values(['country', 'sector', 'year'])

        final_data.to_csv('../data/euklems_' + release + '.csv', index=False)
        print('\n ...Done!')


    elif release == "2017":
        file_path = "../data/raw_data/EUKLEMS_" + release + '/data_raw.csv'
        data_temp = pd.read_csv(file_path)

        # First need to put the data into tidy panel format

        data_temp = pd.melt(data_temp, id_vars=['country', 'var', 'code'], var_name='year')
        data_temp['year'] = data_temp['year'].apply(lambda x: x[1:])
        data_temp = data_temp.pivot(index=['country', 'year', 'code'], columns='var')['value']
        data_temp = data_temp.reset_index()
        data_temp['year'] = data_temp['year'].astype(float)

        # Dictionaries
        sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "H": "trs", "I": "rst",
                        "58-60": "pms", "61": "com", "62-63": "bss", "K": "fin", "L": "res", "M-N": "bss", "O": "gov",
                        "P": "edu", "Q": "hlt", "R-S": "per", "T": "per"}

        country_filter = data_temp['country'].isin(countries)
        year_filter = (data_temp['year'] >= sample_period[0]) & (data_temp['year'] <= sample_period[1])

        data_temp['sector'] = data_temp['code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
        sector_filter = data_temp['sector'].isin(sectors_dict.values())

        data_temp = data_temp[country_filter & sector_filter & year_filter]  # Select data based on filter

        # Create weights of each sector in value added for sectors classification so that we can create price indexes
        data_temp['VA_sector'] = data_temp.groupby(['country', 'year', 'sector'])['VA'].transform('sum')
        data_temp['ws'] = data_temp['VA'] / data_temp['VA_sector']
        data_temp['VA_P_ws'] = data_temp['VA_P'] * data_temp['ws']

        data_temp = data_temp.groupby(["country", "year", "sector"]).aggregate({'VA': 'sum', 'H_EMP': 'sum', 'VA_P_ws': 'sum'})
        data_temp = data_temp.reset_index()
        data_temp.columns = ["country", "year", "sector", "VA", "H", "P"]
        data_temp.to_csv('../data/euklems_' + release + '.csv', index=False)

    elif release == "2009":
        print('Building', release, 'data ...')
        file_path = "../data/raw_data/EUKLEMS_" + release + '/data_raw.csv'
        data_temp = pd.read_csv(file_path)

        # First need to put the data into tidy panel format

        data_temp = pd.melt(data_temp, id_vars=['country', 'var', 'code'], var_name='year')
        data_temp['year'] = data_temp['year'].apply(lambda x: x[1:])
        data_temp = data_temp.pivot(index=['country', 'year', 'code'], columns='var')['value']
        data_temp = data_temp.reset_index()
        data_temp['year'] = data_temp['year'].astype(float)

        # Dictionaries
        # sectors_dict = {"AtB": 'agr', "C": "man", "D": "man", "E": "man", "F": "man", "G": "trd", "H": "rst", "60t63": "trs",
        #                 "64": "bss", "J": "fin", "70": "res", "71t74": "bss", "L": "gov",
        #                 "M": "edu", "N": "hlt", "O": "per", "P": "per"}
        sectors_dict = {"AtB": 'agr', "C": "man", "D": "man", "E": "man", "F": "man", "G": "trd", "H": "nps", "60t63": "nps",
                        "64": "bss", "J": "fin", "70": "nps", "71t74": "bss", "L": "nps",
                        "M": "nps", "N": "nps", "O": "nps", "P": "nps", "TOT": "tot"}

        data_temp.loc[data_temp.country == 'GER', 'country'] = 'DE'
        data_temp.loc[data_temp.country == 'GRC', 'country'] = 'GR'
        data_temp.loc[data_temp.country == 'UK', 'country'] = 'GBR'
        iso2_list = coco.convert(names=data_temp['country'].values, to='ISO2', not_found=None)
        data_temp.loc[:, 'country'] = iso2_list
        data_temp.loc[:, 'country'] = data_temp.loc[:, 'country'].replace('USA-NAICS', "US")

        country_filter = data_temp['country'].isin(countries)
        year_filter = (data_temp['year'] >= sample_period[0]) & (data_temp['year'] <= sample_period[1])

        data_temp['sector'] = data_temp['code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
        sector_filter = data_temp['sector'].isin(sectors_dict.values())

        data_temp = data_temp[country_filter & sector_filter & year_filter]  # Select data based on filter

        # # Create weights of each sector in value added for sectors classification so that we can create price indexes
        # data_temp.loc[:, 'VA_sector'] = data_temp.groupby(['country', 'year', 'sector'])['VA'].transform('sum')
        # data_temp['ws'] = data_temp['VA'] / data_temp['VA_sector']
        # data_temp['VA_P_ws'] = data_temp['VA_P'] * data_temp['ws']

        data_temp["VA_Q"] = data_temp["VA"] / data_temp['VA_P']

        data_temp = data_temp.groupby(["country", "year", "sector"]).aggregate({'VA': 'sum', 'H_EMP': 'sum', 'VA_Q': 'sum'})
        data_temp = data_temp.reset_index()
        data_temp.columns = ["country", "year", "sector", "VA", "H", "VA_Q"]
        data_temp.to_csv('../data/euklems_' + release + '.csv', index=False)
        print('\n ...Done!')

    else:
        print('No data found for release ' + release)


# Order matters: the 2023 branch (1970-2019) reads data/euklems_2009.csv to
# back-extrapolate via growth rates for 1970-1994, so the 2009 panel must be
# built first.
select_data(countries=("US", "EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"),
            sample_period=(1970, 2009),
            release="2009")

select_data(countries=("US", "EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"),
            sample_period=(1970, 2019),
            release="2023")
