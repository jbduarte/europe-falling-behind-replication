from simulate_country import *
from simulate_CF import *
from simulate_extrapolate import *
from simulate_extrapolate_sector import *
from get_initial_rel_A import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

method = 'last_period'
release = '2021'

'''
1. Simulate EU-15 benchmark 
'''

results_EU = simulate_country('EU15')
results_US = simulate_country('US')

# Unpack results
data_EU = results_EU['data']
data_US = results_US['data']
A_model_EU, rel_A_model_EU = results_EU['A_model'], results_EU['rel_A_model']
A_model_US, rel_A_model_US = results_US['A_model'], results_US['rel_A_model']

temp_data = pd.DataFrame(A_model_US.values, index=A_model_US.index, columns=['Y/L'])
temp_data.to_csv('../Outputs/Data/A_US_benchmark.csv')

temp_data = pd.DataFrame(A_model_EU.values, index=A_model_EU.index, columns=['Y/L'])
temp_data.to_csv('../Outputs/Data/A_EU_benchmark.csv')

rel_A_init = get_initial_rel_A()

'''
2. Counterfactuals
'''

'2.1 Growing like the US: Full Sample'

sectors = data_US.sector.unique().tolist()

for sector in sectors:

    # Get US labor productivity growth for each sector
    lps = pd.DataFrame(data_US.loc[data_US.sector == sector, 'L_PROD_normalized'].values, columns=[sector])

    results_EU_CF = simulate_CF('EU15', [sector], lps)

    # Unpack results
    data_EU_CF = results_EU['data']
    A_model_EU_CF, rel_A_model_EU_CF = results_EU_CF['A_model'], results_EU_CF['rel_A_model']

    # Save Y/L CF into Outputs/Data/
    temp_data = pd.DataFrame(A_model_EU_CF.values, index=A_model_EU_CF.index, columns=['Y/L'])
    temp_data.to_csv('../Outputs/Data/A_EU_cf_' + sector + '_fullsample.csv')
    #(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU)).mean())*100
    #(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU_CF)).mean())*100

    # Plot Relative A CF

    'rel_A_levels'
    plt.plot(data_EU_CF.year.unique(), np.ones(len(data_EU_CF.year.unique())), 'grey', linestyle='dashed')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU_CF * rel_A_init.loc['EU15'].values[0],
             lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Counterfactual: \texttt{Y/L}')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Relative aggregate labor productivity', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
    plt.xticks(fontsize=14)
    # plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_fullsample_' + sector + '.pdf', bbox_inches='tight')
    plt.close()

'2.2 Growing like the US: starting in 1995'

last_period_LS = np.zeros((12, 12))
last_period_LS_model = data_EU.loc[data_EU.year == 2018, 'model_pred'].values

for sector in sectors:

    # Get US labor productivity growth for each sector
    temp = pd.DataFrame(data_EU.loc[(data_US.sector == sector) & (data_US.year <= 1995), 'L_PROD_normalized'].values, columns=[sector])
    gr = np.log(data_US.loc[(data_US.sector == sector) & (data_US.year >= 1995), 'L_PROD_normalized']).diff().values
    gr = gr[1:]

    temp = temp[sector].values.tolist()
    for i in range(len(gr)):
        temp.append(temp[-1] * (1 + gr[i]))

    lps = pd.DataFrame(temp, columns=[sector])

    results_EU_CF = simulate_CF('EU15', [sector], lps)

    # Unpack results
    data_EU_CF = results_EU_CF['data']
    A_model_EU_CF, rel_A_model_EU_CF = results_EU_CF['A_model'], results_EU_CF['rel_A_model']
    last_period_LS[:, sectors.index(sector)] = data_EU_CF.loc[data_EU_CF.year == 2018, 'model_pred'].values

    # Save Y/L CF into Outputs/Data/
    temp_data = pd.DataFrame(A_model_EU_CF.values, index=A_model_EU_CF.index, columns=['Y/L'])
    temp_data.to_csv('../Outputs/Data/A_EU_cf_' + sector + '_1995sample.csv')
    #(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU)).mean())*100
    #(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU_CF)).mean())*100

    # Plot Relative A CF
    rel_A_init = get_initial_rel_A()


    'rel_A_levels'
    plt.plot(data_EU_CF.year.unique(), np.ones(len(data_EU_CF.year.unique())), 'grey', linestyle='dashed')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU_CF * rel_A_init.loc['EU15'].values[0],
             lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Counterfactual: \texttt{Y/L}')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Relative aggregate labor productivity', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
    plt.xticks(fontsize=14)
    # plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_1995sample_' + sector + '.pdf', bbox_inches='tight')
    plt.close()

'TRD + BSS together'
# Get US labor productivity growth for each sector
temp = pd.DataFrame(data_EU.loc[(data_US.sector == 'trd') & (data_US.year <= 1995), 'L_PROD_normalized'].values, columns=['trd'])
gr = np.log(data_US.loc[(data_US.sector == 'trd') & (data_US.year >= 1995), 'L_PROD_normalized']).diff().values
gr = gr[1:]

temp = temp['trd'].values.tolist()
for i in range(len(gr)):
    temp.append(temp[-1] * (1 + gr[i]))

lps = pd.DataFrame(temp, columns=['trd'])

temp = pd.DataFrame(data_EU.loc[(data_US.sector == 'bss') & (data_US.year <= 1995), 'L_PROD_normalized'].values, columns=['bss'])
gr = np.log(data_US.loc[(data_US.sector == 'bss') & (data_US.year >= 1995), 'L_PROD_normalized']).diff().values
gr = gr[1:]

temp = temp['bss'].values.tolist()
for i in range(len(gr)):
    temp.append(temp[-1] * (1 + gr[i]))

lps['bss'] = temp

results_EU_CF = simulate_CF('EU15', ['trd', 'bss'], lps)

# Unpack results
data_EU_CF = results_EU['data']
A_model_EU_CF, rel_A_model_EU_CF = results_EU_CF['A_model'], results_EU_CF['rel_A_model']

# Save Y/L CF into Outputs/Data/
temp_data = pd.DataFrame(A_model_EU_CF.values, index=A_model_EU_CF.index, columns=['Y/L'])
temp_data.to_csv('../Outputs/Data/A_EU_cf_trd_bss_1995sample.csv')
#(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU)).mean())*100
#(np.diff(np.log(A_model_US)).mean() - np.diff(np.log(A_model_EU_CF)).mean())*100

# Plot Relative A CF
rel_A_init = get_initial_rel_A()


'rel_A_levels'
plt.plot(data_EU_CF.year.unique(), np.ones(len(data_EU_CF.year.unique())), 'grey', linestyle='dashed')
plt.plot(data_EU_CF.year.unique(), rel_A_model_EU_CF * rel_A_init.loc['EU15'].values[0],
         lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Counterfactual: \texttt{Y/L}')
plt.plot(data_EU_CF.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Relative aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
plt.xticks(fontsize=14)
# plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_1995sample_trd_bss.pdf', bbox_inches='tight')
plt.close()


'Labor Shares model vs. CF'

sectors = data_EU.sector.unique().tolist()
last_period_LS_model = last_period_LS_model.astype(float)

import matplotlib

cmap = matplotlib.cm.get_cmap('tab20')

for j, sector in enumerate(sectors):
    fig, ax = plt.subplots()
    for i in range(12):
        ax.plot(np.array([last_period_LS_model[j]]), last_period_LS[j, i],
                'D', markersize=12,  color=cmap(i),
                alpha=0.5, label=sectors[i])
    # ax.annotate(sectors, (np.array([last_period_LS_model[j]]*12), last_period_LS[j, :]),
    #             fontsize=14,
    #             alpha=0.6)
    plt.xlabel('Model labor share in 2018', fontsize=16)
    plt.ylabel('Counterfactual labor share in 2018', fontsize=16)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3", label='45 degree line')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig('../Outputs/Figures/fig_LS_model_cf_' + sector + '.pdf', bbox_inches="tight")
    plt.close()

'2.3 Avoiding slowdown after 1995: keep growing as in before 1995'

for sector in sectors:
    temp = pd.DataFrame(data_EU.loc[(data_US.sector == sector) & (data_US.year <= 1995), 'L_PROD_normalized'].values, columns=[sector])
    gr = np.ones(23) * np.log(temp).diff().iloc[1:].values.mean()

    temp = temp[sector].values.tolist()
    for i in range(len(gr)):
        temp.append(temp[-1] * (1 + gr[i]))

    lps = pd.DataFrame(temp, columns=[sector])

    results_EU_CF = simulate_CF('EU15', [sector], lps)

    # Unpack results
    data_EU_CF = results_EU['data']
    A_model_EU_CF, rel_A_model_EU_CF = results_EU_CF['A_model'], results_EU_CF['rel_A_model']

    # Save Y/L CF into Outputs/Data/
    temp_data = pd.DataFrame(A_model_EU_CF.values, index=A_model_EU_CF.index, columns=['Y/L'])
    temp_data.to_csv('../Outputs/Data/A_EU_cf_' + sector + '_avoid.csv')

    # Plot Relative A CF
    plt.plot(data_EU_CF.year.unique(), np.ones(len(data_EU_CF.year.unique())), 'grey', linestyle='dashed')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU_CF * rel_A_init.loc['EU15'].values[0],
             lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Counterfactual: \texttt{Y/L}')
    plt.plot(data_EU_CF.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Relative aggregate labor productivity', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
    plt.xticks(fontsize=14)
    # plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_avoid_' + sector + '.pdf', bbox_inches='tight')
    plt.close()


'2.3 Extrapolating: what if each sector doubles its growth rate in EU, what happens to rel A?'

results_EU_CF_extrapolate = simulate_extrapolate('EU15')
# Unpack results
data_EU_CF_extrapolate = results_EU_CF_extrapolate['data']
A_model_EU_CF_extrapolate, rel_A_model_EU_CF_extrapolate = results_EU_CF_extrapolate['A_model'], results_EU_CF_extrapolate['rel_A_model']

extrapolate_A = rel_A_model_EU.append(rel_A_model_EU_CF_extrapolate.iloc[1:])

plt.plot(extrapolate_A.index, np.ones(len(extrapolate_A)), 'grey', linestyle='dashed')
plt.plot(rel_A_model_EU_CF_extrapolate.iloc[1:].index, rel_A_model_EU_CF_extrapolate.iloc[1:] * rel_A_init.loc['EU15'].values[0],
         lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Extrapolate: \texttt{Y/L}')
plt.plot(data_EU.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Relative aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
plt.xticks(fontsize=14)
# plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_extrapolate.pdf', bbox_inches='tight')
plt.close()


data_EU_CF_extrapolate = data_EU_CF_extrapolate[data_EU_CF_extrapolate.year > 2018]
colors_plot = ['b', 'c', 'y']


def my_plot(sectors):
    for j, sector in enumerate(sectors):
        plt.plot(data_EU.year.unique(), data_EU[data_EU.sector == sector]['model_pred'], '^-', color=colors_plot[j],  markersize=6,
                 markeredgecolor='darkred', alpha=0.95, label=r'Model: ' + sector)
        plt.plot(data_EU_CF_extrapolate.year.unique(), data_EU_CF_extrapolate[data_EU_CF_extrapolate.sector == sector]['model_pred'],
                 label=r'Extrapolate', color='darkviolet')
    plt.ylabel('Employment Share', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=14)
    # plt.yticks([0.02, 0.04, 0.06], fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_Europe_extrapolate_' + sectors[0] + '.pdf', bbox_inches='tight')
    plt.close()


'agr'
my_plot(['agr'])

'man'
my_plot(['man'])

'trd, bss and fin'
my_plot(['trd', 'bss', 'fin'])

'rst, trs'
my_plot(['rst', 'trs'])

'gov, hlt, edu'
my_plot(['gov', 'hlt', 'edu'])

'res, per'
my_plot(['res', 'per'])


for sector in sectors:

    results_EU_CF_extrapolate2 = simulate_extrapolate_sector('EU15', sector)
    # Unpack results
    rel_A_model_EU_CF_extrapolate2 = results_EU_CF_extrapolate2['rel_A_model']

    extrapolate_A2 = rel_A_model_EU.append(rel_A_model_EU_CF_extrapolate2.iloc[1:])

    plt.plot(extrapolate_A.index, np.ones(len(extrapolate_A)), 'grey', linestyle='dashed')
    plt.plot(rel_A_model_EU_CF_extrapolate.iloc[1:].index, rel_A_model_EU_CF_extrapolate.iloc[1:] * rel_A_init.loc['EU15'].values[0],
             lw=2, color='darkviolet', ls='dashed', alpha=0.95, label=r'Extrapolate: \texttt{Y/L}')
    plt.plot(rel_A_model_EU_CF_extrapolate.iloc[1:].index, rel_A_model_EU_CF_extrapolate2.iloc[1:] * rel_A_init.loc['EU15'].values[0],
             lw=2, color='C1', ls='dashed', alpha=0.95, label=r'Double $\Delta \log{Y/L}$ in ' + sector)

    plt.plot(data_EU.year.unique(), rel_A_model_EU * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
             markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
    plt.ylabel('Relative aggregate labor productivity', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=10, bbox_to_anchor=(0.4, 0.9))
    plt.xticks(fontsize=14)
    # plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_counterfactual_rel_A_levels_extrapolate_' + sector + '.pdf', bbox_inches='tight')
    plt.close()

