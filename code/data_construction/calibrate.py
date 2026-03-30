"""
Main script to calibrate the model
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.optimize as opt
import model_labor_shares as model
import json
from scipy.linalg import block_diag
import sys
sys.path.append('/Applications/Stata/utilities')


def calibrate(data_temp, method='last_period', base_sector='man'):
    grouped = data_temp.groupby(['sector'])
    data_temp['LS_' + base_sector] = data_temp.groupby(['sector'])['LS'].transform(lambda x: x / grouped['LS'].get_group(base_sector).values)


    ''' 
    ------------------------
    #	Parameterization 
    ------------------------
    '''

    '''Solving for free parameters (omegas) to match initial values'''
    data_first_year = data_temp[(data_temp['year'] == data_temp['year'].unique()[0]) & (data_temp.sector != 'tot')]
    omegas = data_first_year.loc[:, ['sector', 'LS']]
    omegas = omegas.set_index('sector')
    omega_m = omegas.loc[base_sector].values[0]

    # construct series that match inputs for model: relative labor share, relative labor productivity, productivity and labor share of base sector
    filter_sectors = (data_temp['sector'] != 'tot') & (data_temp['sector'] != base_sector)
    sectors = data_temp[filter_sectors]['sector'].unique()
    sample = data_temp.copy()
    grouped = sample.groupby('sector')
    sample['Li_Lm'] = sample['LS_' + base_sector]
    sample['Am_Ai'] = grouped['L_PROD_normalized'].transform(lambda x: grouped.get_group(base_sector)['L_PROD_normalized'].values / x)
    sample['Am'] = grouped['L_PROD_normalized'].transform(lambda x: grouped.get_group(base_sector)['L_PROD_normalized'].values)
    sample['Lm'] = grouped['LS'].transform(lambda x: grouped.get_group(base_sector)['LS'].values / omega_m)
    sample['A_ws'] = sample['L_PROD_normalized'] * sample['LS']
    sample['A'] = sample[sample.sector != 'tot'].groupby('year')['A_ws'].transform(lambda x: x.sum())

    sample = sample.loc[(sample.sector != base_sector) & (sample.sector != 'tot')]

    sample = sample[['year', 'sector', 'Li_Lm', 'Am_Ai', 'A']]
    sample = sample.pivot_table(index=['year', 'A'], columns='sector',
                                values=['Li_Lm', 'Am_Ai'])
    sample.columns = ['_'.join(map(str, x)) for x in sample.columns]
    sample = sample.reset_index()
    sample_save = sample.copy()

    if method == 'last_period':

        sigma = 0.3  # Comin et al. (2021) CEX data suggest 0.3. OECD data 0.25 to 0.35

        '''1. Calibration using last observation of relative labor shares and path of one sector to get sigma'''

        def compute_epsilon(sigma, sector, data):
            numerator = np.log(data['Li_Lm_' + sector].values[-1]) - \
                        np.log(omegas.loc[sector].values[0]/omega_m) - \
                        (1 - sigma) * np.log(data['Am_Ai_' + sector].values[-1]) + \
                        (1 - sigma) * np.log(data['A'].values[-1])
            denominator = (1 - sigma) * np.log(data['A'].values[-1])

            return numerator / denominator

        epsilons = []
        for j in sectors:
            epsilons.append(compute_epsilon(sigma, j, sample))

        print('Sigma:', sigma)
        for i, sector in enumerate(sectors):
            print('Epsilon_' + sector + ' :', epsilons[i])
            print('Omega_' + sector + ' :', omegas.loc[sector].values[0])

        params = {'sigma': sigma, 'omegas': list(omegas.LS.values), 'epsilons': list(epsilons)}

        # create json object from dictionary
        json_obj = json.dumps(params)
        # open file for writing, "w"
        f = open("../Outputs/Parameters/params_" + method + ".json", "w")
        # write json object to file
        f.write(json_obj)
        # close file
        f.close()

        return params

    # check omegas match with epsilons.


    if method == "estimation_GMM":

        sample.iloc[:, 1:] = np.log(sample.iloc[:, 1:]).diff()
        sample = sample.iloc[1:, :]

        sector_samples = []
        for i in sectors:
            temp = sample.loc[:, ['Am_Ai_' + i, 'Li_Lm_' + i]]
            temp[i] = 1
            sector_samples.append(temp)

        D = block_diag(*sector_samples)
        columns = []
        for i in range(len(sectors)):
            columns += list(sector_samples[i].columns)

        sample2 = pd.DataFrame(D, columns=columns)
        sample2['year'] = np.tile(sample.year.values, len(sectors))
        sample2['Am'] = np.tile(sample.Am.values, len(sectors))
        sample2['Lm'] = np.tile(sample.Lm.values, len(sectors))
        sample2.to_stata('../Outputs/Data/data.dta')


        # Setup Stata from within Python
        from pystata import config
        config.init('mp')

        # Load Python dataframe into Stata
        from pystata import stata
        stata.pdataframe_to_data(sample2, True)

        # Create Stata code string
        eq_code = '''local eq{0}  (Li_Lm_{1} - ( (1-{{sigma=.3}})*({{epsilon_{1}=3}}-1)*Am + (1-{{sigma=.3}})*Am_Ai_{1} + ({{epsilon_{1}=3}}-1)*Lm))
        '''

        stata_code2 = eq_code.format(1, 'agr')
        for i, sector in enumerate(sectors[1:]):
            i = i + 2
            stata_code2 += eq_code.format(i, sector)

        temp_s = "(`eq1')"
        for i, sector in enumerate(sectors[1:]):
            i = i + 2
            temp_s = temp_s + " (`eq" + str(i) + "')"

        temp_template = " instruments({0}: Am_Ai_{1}, noconstant)"
        temp_s2 = temp_template.format(1, "agr")
        for i, sector in enumerate(sectors[1:]):
            i = i + 2
            temp_s2 = temp_s2 + temp_template.format(i, sector)

        stata_code2 = stata_code2 + "\n gmm " + temp_s + ", instruments(Am  Lm, noconstant)" + temp_s2 + " winitial(i)  quickderivatives nocommonesample"

        # Run Stata commands in Python
        stata.run(stata_code2, echo=True)

        # Load Stata saved results to Python
        r = stata.get_return()['r(table)']
        sigma = r[0][0]
        epsilons = r[0][1:]

        params = {'sigma': sigma, 'omegas': list(omegas.LS.values), 'epsilons': list(epsilons)}

        # create json object from dictionary
        json_obj = json.dumps(params)
        # open file for writing, "w"
        f = open("../Outputs/Parameters/params_" + method + ".json", "w")
        # write json object to file
        f.write(json_obj)
        # close file
        f.close()

        return params

    if method == "estimation_NLS":
        # TODO: delete lm data moments (these are linearly dependent)

        '2. Calibration using panel data estimation'

        sample = data_temp.copy()
        sample = sample[sample['sector'] != 'tot']
        sample = sample.sort_values(['sector', 'year'])

        def sector_sorter(column):
            """Sort function"""
            sectors = list(sample.sector.unique())
            man = sectors.pop(sectors.index('man'))
            sectors.append(man)

            correspondence = {sector: order for order, sector in enumerate(sectors)}
            return column.map(correspondence)


        def resids(param):

            grouped = sample.groupby('year')
            model_pred = []
            res = []
            i = 0
            for name, _ in grouped:
                subsample = grouped.get_group(name)[['sector', 'L_PROD_normalized', 'LS']]
                subsample = subsample.sort_values(by=['sector'], key=sector_sorter)
                res = np.concatenate((res, model.model_labor_shares(param, subsample['L_PROD_normalized'].values) - subsample['LS'].values))
                i = i + 1

            return res

        # Run Fixed Effects Regression (sector fixed effect)
        params = np.array([0.05]*12 + [1.2]*12 + [0.6])
        model.model_labor_shares(params, np.ones(13)*1)
        resids(params)
        lbounds = tuple([0] * 25)
        ubounds = tuple([5] * 24 + [1])
        # , bounds = (lbounds, ubounds)
        sol = opt.least_squares(resids, x0=params, bounds=(lbounds, ubounds), verbose=2)
        sigma = sol.x[-1]
        epsilons = {}
        sectors_sorted = list(sectors)
        sectors_sorted.sort()
        for i, name in enumerate(sectors_sorted):
            epsilons[name] = sol.x[i]

        print('Sigma:', sigma)
        for i, sector in enumerate(sectors):
            print('Epsilon_' + sector + ' :', epsilons[sector])
            print('Omega_' + sector + ' :', omegas[sector])

        params = {'sigma': sigma, 'omegas': omegas, 'epsilons': epsilons}

        # create json object from dictionary
        json_obj = json.dumps(params)
        # open file for writing, "w"
        f = open("../Outputs/Parameters/params_" + method + ".json", "w")
        # write json object to file
        f.write(json_obj)
        # close file
        f.close()
        return params




