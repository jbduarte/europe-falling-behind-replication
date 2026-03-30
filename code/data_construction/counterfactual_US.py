import matplotlib.pyplot as plt
import numpy as np

from calibrate import *
from model_labor_shares import *
from construct_dataset import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

'''
Settings
'''
method = 'last_period'
base_sector = 'man'
release = "2021"
data = pd.read_csv('../Data/Final Data/euklems_' + release + '.csv')
country = "US"

sample = construct_dataset(data, smooth=True, country=country)
sample = sample.sort_values(by=['year', 'sector'])


params = calibrate(sample, method=method, base_sector=base_sector)


'''Simulate baseline'''
def get_A(data):
    return (data.model_pred * data.L_PROD_normalized).sum()

sample['model_pred'] = sample.groupby('year').apply(lambda x: model_labor_shares(params, x, base_sector=base_sector)).explode().values
A_baseline = sample.groupby('year').apply(lambda x: get_A(x))


# Build counterfactual sectoral productivity
sample.loc[sample.sector == 'gov', 'L_PROD_normalized'] = np.linspace(1, 5, len(sample.year.unique()))
sample['model_pred_CF'] = sample.groupby('year').apply(lambda x: model_labor_shares(params, x, base_sector=base_sector)).explode().values


# TODO: think about counterfactural with fixed labor shares.
def get_A_CF(data):
    return (data.model_pred_CF * data.L_PROD_normalized).sum()

A_CF = sample.groupby('year').apply(lambda x: get_A_CF(x))

A_CF_fixed_shares = sample.groupby('year').apply(lambda x: np.sum(sample.loc[sample.year == 1977, 'model_pred'].values * x.L_PROD_normalized.values))

'''PLOTS'''
def plot_ls_fit(sectors, data):
    colors_plot = ['b-', 'c--', 'y-']
    for j, sector in enumerate(sectors):
        plt.plot(data.year.unique(), data[data.sector == sector]['LS'], colors_plot[j], lw=2, alpha=0.95, label=r'Data: \texttt{' + sector + '}')
        plt.plot(data.year.unique(), data[data.sector == sector]['model_pred'], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Employment Share', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks([0.02, 0.04, 0.06], fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_USA_calib_' + sectors[0] + release + '_' + method + '.pdf', bbox_inches='tight')
    plt.close()

'agr'
plot_ls_fit(['agr'], sample)

'man'
plot_ls_fit(['man'], sample)

'trd, bss and fin'
plot_ls_fit(['trd', 'bss', 'fin'], sample)

if release == '2009':
    plot_ls_fit(['rst', 'trs', 'com'], sample)

else:
    'rst, trs'
    plot_ls_fit(['rst', 'trs'], sample)

'gov, hlt, edu'
plot_ls_fit(['gov', 'hlt', 'edu'], sample)

'res, per'
plot_ls_fit(['res', 'per'], sample)

