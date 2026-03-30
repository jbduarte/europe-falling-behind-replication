from simulate_country import *
from construct_dataset import *
from get_initial_rel_A import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

method = 'last_period'
release = '2021'

'''
0. Get initial A in levels to US
'''
rel_A_init = get_initial_rel_A()

'''
1. SIMULATE EU-15
'''

results = simulate_country('EU15')

# Unpack results
sample = results['data']
A_data, A_model, rel_A_data, rel_A_model = results['A_data'], results['A_model'], results['rel_A_data'], results['rel_A_model']

'''1.1 EU15 PLOT MODEL FIT'''


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
    # plt.yticks([0.02, 0.04, 0.06], fontsize=14)
    plt.tight_layout()
    plt.savefig('../Outputs/Figures/fig_Europe_calib_' + sectors[0] + release + '_' + method + '.pdf', bbox_inches='tight')
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
plt.plot(sample.year.unique(), A_data, lw=2, alpha=0.95, label=r'Data: \texttt{A}')
plt.plot(sample.year.unique(), A_model, '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(fontsize=14)
# plt.yticks([0.02, 0.04, 0.06], fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_Europe_calib_A.pdf', bbox_inches='tight')
plt.close()

'rel_A'
plt.plot(sample.year.unique(), rel_A_data, lw=2, alpha=0.95, label=r'Data: \texttt{A}')
plt.plot(sample.year.unique(), rel_A_model, '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(fontsize=14)
plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_Europe_calib_rel_A.pdf', bbox_inches='tight')
plt.close()

'rel_A_levels'
plt.plot(sample.year.unique(), np.ones(len(sample.year.unique())), 'grey', linestyle='dashed')
plt.plot(sample.year.unique(), rel_A_data * rel_A_init.loc['EU15'].values[0], lw=2, alpha=0.95, label=r'Data: \texttt{Y/L}')
plt.plot(sample.year.unique(), rel_A_model * rel_A_init.loc['EU15'].values[0], '^-', markersize=6, color='darkred', markerfacecolor='lightcoral',
         markeredgecolor='darkred', alpha=0.95, label=r'Model (3): All sectors')
plt.ylabel('Relative aggregate labor productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(fontsize=14)
# plt.yticks(np.arange(0.8, 1.5, 0.1), fontsize=14)
plt.tight_layout()
plt.savefig('../Outputs/Figures/fig_Europe_calib_rel_A_levels.pdf', bbox_inches='tight')
plt.close()

'''
2. SIMULATE All European countries that are part of the EU15
'''

countries = ["EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"]

# create arrays to save last period LS for each sector data and model and arrays for aggregate labor producitivty time series

last_period_LS = np.zeros((len(sample.sector.unique()), 2 * len(countries)))  # I by J=2, j1 = data and  j2= model
A = np.zeros((len(sample.year.unique()), 2 * len(countries)))  # T by K=N*2, N number of countries and 2 because of data and model for each country

for i, country in enumerate(countries):
    i = i * 2
    results = simulate_country(country)

    # Unpack results
    sample = results['data']
    A_data, A_model, rel_A_data, rel_A_model = results['A_data'], results['A_model'], results['rel_A_data'], results['rel_A_model']

    # Save results into matrices
    A[:, i] = rel_A_data
    A[:, i + 1] = rel_A_model

    last_period_LS[:, i] = sample[sample.year == sample.year.unique()[-1]].LS.values
    last_period_LS[:, i + 1] = sample[sample.year == sample.year.unique()[-1]].model_pred.astype(float).values

'''
2.1 Plot the Test of European countries model fit last period labor share
'''

sectors = sample.sector.unique().tolist()

for j, sector in enumerate(sectors):
    fig, ax = plt.subplots()
    for i, country in enumerate(countries):
        i = i * 2
        if i == 0:
            ax.plot(last_period_LS[j, i], last_period_LS[j, i + 1],
                    'D',
                    markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=12,
                    alpha=0.95,
                    label='Model (1)')
            ax.annotate(country, (last_period_LS[j, i], last_period_LS[j, i + 1]),
                        color='red',
                        fontweight='bold',
                        fontsize=16)
        else:
            ax.plot(last_period_LS[j, i], last_period_LS[j, i + 1],
                    'D',
                    markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=12,
                    alpha=0.5)
            ax.annotate(country, (last_period_LS[j, i], last_period_LS[j, i + 1]),
                        fontsize=14,
                        alpha=0.6)
    plt.xlabel('Data labor share in 2018', fontsize=23)
    plt.ylabel('Model labor share in 2018', fontsize=23)
    plt.xticks(np.arange(0, 0.34, 0.04), fontsize=18)
    plt.yticks(np.arange(0, 0.34, 0.04), fontsize=18)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3", label='45 degree line')
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)
    plt.savefig('../Outputs/Figures/fig_test_' + sector + '.pdf', bbox_inches="tight")
    plt.close()

'''
2.2 Plot the Test of European countries model fit relative A
'''

fig, axes = plt.subplots(5, 3)
for i, country in enumerate(countries[1:]):
    i = i * 2
    axes[int(i / 2) // 3, int(i / 2) % 3].plot(sample.year.unique(), A[:, i + 2], 'b-', alpha=0.75, label=r'$\frac{Y_t}{L_t}$: Data')
    axes[int(i / 2) // 3, int(i / 2) % 3].plot(sample.year.unique(), A[:, i + 1 + 2], '^-', markersize=2, markevery=2, color='darkred',
                                               markerfacecolor='lightcoral',
                                               markeredgecolor='darkred', linewidth=0.8, alpha=0.95, label=r'Model (3): All sectors')
    axes[int(i / 2) // 3, int(i / 2) % 3].grid(axis='y', alpha=0.3)
    axes[int(i / 2) // 3, int(i / 2) % 3].set_title(r' ' + country, fontsize=12, y=-0.06)
    axes[int(i / 2) // 3, int(i / 2) % 3].set_xticks([1980, 1990, 2000, 2010, 2020])
    if country == 'IE':
        axes[int(i / 2) // 3, int(i / 2) % 3].set_yticks(np.arange(0.8, 3.4, 0.6))
    elif country == 'PT':
        axes[int(i / 2) // 3, int(i / 2) % 3].set_yticks(np.arange(0.8, 2.8, 0.4))
    else:
        axes[int(i / 2) // 3, int(i / 2) % 3].set_yticks(np.arange(0.6, 1.6, 0.2))
    axes[int(i / 2) // 3, int(i / 2) % 3].tick_params(labelsize=9)
axes[-1, -2].legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=3)
fig.tight_layout()
fig.savefig('../Outputs/Figures/fig_test_rel_A.pdf', bbox_inches="tight")

fig, axes = plt.subplots(5, 3)
for i, country in enumerate(countries[1:]):
    i = i * 2
    axes[int(i / 2) // 3, int(i / 2) % 3].plot(sample.year.unique(), np.ones(len(sample.year.unique())), 'grey', linestyle='dashed')
    axes[int(i / 2) // 3, int(i / 2) % 3].plot(sample.year.unique(), A[:, i + 2] * rel_A_init.loc[country].values[0],
                                               'b-', alpha=0.75, label=r'$\frac{Y_t}{L_t}$: Data')
    axes[int(i / 2) // 3, int(i / 2) % 3].plot(sample.year.unique(), A[:, i + 1 + 2] * rel_A_init.loc[country].values[0],
                                               '^-', markersize=2, markevery=2, color='darkred',
                                               markerfacecolor='lightcoral',
                                               markeredgecolor='darkred', linewidth=0.8, alpha=0.95, label=r'Model (3): All sectors')
    axes[int(i / 2) // 3, int(i / 2) % 3].grid(axis='y', alpha=0.3)
    axes[int(i / 2) // 3, int(i / 2) % 3].set_title(r' ' + country, fontsize=12, y=-0.06)
    axes[int(i / 2) // 3, int(i / 2) % 3].set_xticks([1980, 1990, 2000, 2010, 2020])

    axes[int(i / 2) // 3, int(i / 2) % 3].set_yticks(np.arange(0.1, 1.4, 0.3))
    axes[int(i / 2) // 3, int(i / 2) % 3].tick_params(labelsize=7)
axes[-1, -2].legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=3)
fig.tight_layout()
fig.savefig('../Outputs/Figures/fig_test_rel_A_levels.pdf', bbox_inches="tight")
