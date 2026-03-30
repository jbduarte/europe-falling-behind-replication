#  Script to define the model function
import numpy as np
import scipy.optimize as opt


def model_labor_shares(params, sectoral_lp_data, base_sector='man'):
    """
    This function computes the labor shares predicted by the model given a vector of parameters and
    a vector of sectoral labor productivity
    """
    n = len(sectoral_lp_data.sector.unique())   # number of sectors
    man_index = list(sectoral_lp_data.sector.unique()).index(base_sector)
    omegas, epsilons, sigma = params['omegas'].copy(), np.array(params['epsilons']), params['sigma']
    omega_m = omegas.pop(man_index)
    omegas = np.array(omegas)
    # Compute Li/Lm for all sectors i in I
    Am = sectoral_lp_data[sectoral_lp_data.sector == base_sector]['L_PROD_normalized'].values[0]
    As = Am / sectoral_lp_data[sectoral_lp_data.sector != base_sector]['L_PROD_normalized'].values
    Ai = sectoral_lp_data[sectoral_lp_data.sector != base_sector]['L_PROD_normalized'].values

    def resids(LS, sigma, omegas, epsilons, Ai, As, Am):
        res = np.ones(n-1)
        A = np.sum(LS * Ai) + (1 - np.sum(LS)) * Am
        for i in range(n-1):
            res[i] = np.log(LS[i]/(1 - np.sum(LS))) - (
                        np.log(omegas[i] / omega_m) + (1 - sigma) * np.log(As[i]) + (1 - sigma) * (epsilons[i] - 1) * np.log(A))

        return res

    sol = opt.root(lambda LS: resids(LS, sigma, omegas, epsilons, Ai, As, Am), x0=np.ones(n-1)/(n+5), method='lm').x

    lis = np.insert(sol, man_index, 1-np.sum(sol))
    return lis


