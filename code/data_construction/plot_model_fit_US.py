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
SETTINGS
'''
method = 'last_period'
base_sector = 'man'
release = "2021"
data = pd.read_csv('../Data/Final Data/euklems_' + release + '.csv')
country = "US"

# TODO: finish model fit with other sectoral breakdown

'''
CALIBRATE MODEL AND RUN MODEL SIMULATIONS
'''
sample = construct_dataset(data, smooth=True)
sample = sample.sort_values(by=['year', 'sector'])

# CALIBRATE
params = calibrate(sample, method=method, base_sector=base_sector)

# SIMULATE MODEL
sample['model_pred'] = sample.groupby('year').apply(lambda x: model_labor_shares(params, x, base_sector=base_sector)).explode().values


def get_A_data(data):
    return (data.LS * data.L_PROD_normalized).sum()


def get_A_model(data):
    return (data.model_pred * data.L_PROD_normalized).sum()


A_model = sample.groupby('year').apply(lambda x: get_A_model(x))
A_data = sample.groupby('year').apply(lambda x: get_A_data(x))


'''PLOT MODEL FIT'''
def plot_ls_fit(sectors, data):
    colors_plot = ['b-', 'c--', 'y-']
    for j, sector in enumerate(sectors):
        plt.plot(data.year.unique(), data[data.sector == sector]['LS'], colors_plot[j], lw=2, alpha=0.95, label=r'Data: \texttt{' + sector + '}')
        plt.plot(data.year.unique(), data[data.sector == sector]['model_pred'], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Employment Share', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=14)
    #plt.yticks([0.02, 0.04, 0.06], fontsize=14)
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

'A'
plt.plot(data.year.unique(), A_data, lw=2, alpha=0.95, label=r'Data: \texttt{A}')
plt.plot(data.year.unique(), A_model, '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(fontsize=14)
# plt.yticks([0.02, 0.04, 0.06], fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_USA_calib_A.pdf', bbox_inches='tight')
plt.close()
