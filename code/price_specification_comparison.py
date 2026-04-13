"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        price_specification_comparison.py
Purpose:     Robustness check on the price specification underlying sectoral
             demand. Compares model-implied employment shares under:
               (A) Model prices        p_i = 1/A_i      (theoretical)
               (B) Observed prices     P_i = VA_i/VA_Q_i (empirical deflator)
             and reports three metrics: level differences, year-to-year first
             differences, and cumulative differences from the base year.
             Answers the referee's request for a direct test of the model
             pricing assumption.
Pipeline:    Step 12/19 — Table 3 (price-specification robustness).
Inputs:      ../data/euklems_2023.csv (EUKLEMS 2023 VA, VA_Q, H by country-sector)
             ../data/raw/OECD_GDP_ph.xlsx (OECD GDP per hour for C_tilde recovery)
             Applied to US, DE, FR, GB, IT.
Outputs:     ../output/tables/table_3.tex and ../output/tables/table_3.xlsx.
Dependencies: Standalone — re-calibrates preference parameters internally from
              US data (does not import model_calibration_USA.py), so it can be
              run independently of Steps 1-9.
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: CALIBRATE PARAMETERS FROM US DATA
# ============================================================
# The preference parameters (sigma, eps_i, omega_i) are recalibrated internally
# here — we do NOT import from model_calibration_USA.py — so this file is
# self-contained and can be run even when the main calibration pipeline has not
# executed. Trade-off: the numbers must match those in model_calibration_USA.py
# to a few decimal places; if they ever diverge, the divergence indicates that
# either the HP filter smoothing constant (100) or the EUKLEMS input file has
# changed between steps.

data_all = pd.read_csv('../data/euklems_2023.csv', index_col=[0, 1])
data_all.rename(index={'US': 'USA'}, inplace=True)
# Labor productivity in sectoral VA per hour (x100 for unit convenience).
data_all['y_l'] = (data_all['VA_Q'] / data_all['H']) * 100

# OECD GDP per hour — aggregate anchor for C_tilde (utility index) level recovery.
# Filtered to USD-denominated series so cross-country level comparisons are valid.
GDP_ph = pd.read_excel('../data/raw/OECD_GDP_ph.xlsx', index_col=[0, 5], engine='openpyxl')
GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']

# --- US calibration ---
data_us = data_all.loc['USA']
GDP_ph_us = GDP_ph.loc['USA']

sectors_all = ['agr', 'man', 'trd', 'bss', 'fin', 'nps', 'ser', 'tot']
six_sectors = ['agr', 'man', 'trd', 'bss', 'fin', 'nps']

data_sec_us = {}
for s in sectors_all:
    data_sec_us[s] = data_us[data_us['sector'] == s]

# HP-filter all US series with lambda=100 (standard value for annual data, e.g.
# Ravn-Uhlig 2002). The cyclical component is discarded; only the trend component
# (second return value) enters the structural-transformation decomposition.
_, GDP_hp = sm.tsa.filters.hpfilter(GDP_ph_us['Value'], 100)
g_GDP = np.array(GDP_hp / GDP_hp.shift(1) - 1).flatten()

h_us = {}
y_l_us = {}
p_us = {}
for s in sectors_all:   # sector loop over {agr, man, trd, bss, fin, nps, ser, tot}
    _, h_us[s] = sm.tsa.filters.hpfilter(data_sec_us[s]['H'], 100)
    _, y_l_us[s] = sm.tsa.filters.hpfilter(data_sec_us[s]['y_l'], 100)
    # Observed sectoral price deflator P_i = VA (nominal) / VA_Q (real quantity).
    # This is Specification B — the "observed prices" arm of the robustness test.
    _, p_us[s] = sm.tsa.filters.hpfilter(data_sec_us[s]['VA'] / data_sec_us[s]['VA_Q'], 100)

# Employment shares (6 sectors)
h_sum_6_us = sum(h_us[s] for s in six_sectors)
share_us = {}
for s in six_sectors + ['ser']:
    _, share_us[s] = sm.tsa.filters.hpfilter(h_us[s] / h_sum_6_us, 100)

# Initial employment shares (omega)
om = {}
for s in six_sectors + ['ser']:
    om[s] = np.array(share_us[s])[0]

# Last-period relative values
rel_l_last = {}
rel_p_last = {}
for s in ['agr', 'trd', 'bss', 'fin', 'nps', 'ser']:
    rel_l_last[s] = np.array(h_us[s] / h_us['man'])[-1]
    rel_p_last[s] = np.array(p_us[s] / p_us['man'])[-1]

_, E_pm = sm.tsa.filters.hpfilter((data_sec_us['tot']['VA'] * 100) / p_us['man'], 100)
E_pm_last = np.array(E_pm)[-1]

_, man_va_share = sm.tsa.filters.hpfilter(
    data_sec_us['man']['VA'] / data_sec_us['tot']['VA'], 100
)
man_va_share_last = np.array(man_va_share)[-1]

# Closed-form calibration helpers:
#   sigma_ft(eps_s): invert the services vs. manufacturing relative-labor equation
#       for the CES elasticity sigma given a chosen services preference elasticity.
#   eps_i_ft(sigma, ...): invert the sector-i vs. manufacturing relative-labor
#       equation for eps_i given sigma and the observed terminal-period ratios.
# Both use last-period observations (rel_l_last, rel_p_last, E_pm_last) as the
# identifying moments.
def sigma_ft(eps_s):
    nom = (np.log(rel_l_last['ser']) - np.log(om['ser']/om['man'])
           - (eps_s - 1)*np.log(man_va_share_last/om['man']))
    den = np.log(rel_p_last['ser']) + (eps_s - 1)*np.log(E_pm_last)
    return 1 - nom/den

def eps_i_ft(sigma, rel_l, rel_om, rel_p):
    nom = np.log(rel_l) - np.log(rel_om) - (1-sigma)*np.log(rel_p)
    den = np.log(man_va_share_last/om['man']) + (1-sigma)*np.log(E_pm_last)
    return 1 + nom/den

# Calibrate sigma using the paper's reference eps_ser = 1.2. The paper's
# reported value is sigma ~ 0.79 (see MEMORY and sec_introduction.tex L63).
sigma = sigma_ft(1.2)
eps = {'man': 1.0}
for s in ['agr', 'trd', 'bss', 'fin', 'nps', 'ser']:
    eps[s] = eps_i_ft(sigma, rel_l_last[s], om[s]/om['man'], rel_p_last[s])

# Closed-form C_tilde (utility index) recovery from the services-vs-manufacturing
# relative labor equation — no fsolve needed, unlike the 6-sector HP specification
# in counterfactuals.py. Level is base-year-normalised (C[0] = 1); only the
# growth path g_C enters the downstream comparison.
def C_index(om_i, li_lm, pi_pm, sigma_val, epsilon_i):
    C_level = ((om['man'] / om_i) * li_lm * (pi_pm ** (sigma_val - 1))) ** (1 / (epsilon_i - 1))
    g_C = np.array(C_level / C_level.shift(1) - 1)
    C = [1]
    for i in range(len(g_C) - 1):
        C.append((1 + g_C[i + 1]) * C[i])
    return C

l_ser_l_man_us = h_us['ser'] / h_us['man']
p_ser_p_man_us = p_us['ser'] / p_us['man']
C_tilde_us = C_index(om['ser'], l_ser_l_man_us, p_ser_p_man_us, sigma, eps['ser'])

print("=" * 70)
print("CALIBRATED PARAMETERS (from US data)")
print("=" * 70)
print(f"sigma = {sigma:.6f}")
for s in six_sectors:
    print(f"  eps_{s} = {eps[s]:.6f},  omega_{s} = {om[s]:.6f}")
print()


# ============================================================
# STEP 2: COMPUTE MODEL SHARES FOR EACH COUNTRY
# ============================================================

# Country set for the price-specification comparison. US is the calibration
# anchor; DE/FR/GB/IT are the EU4 countries used throughout the paper. Note the
# EUKLEMS country codes are 2-letter (DE, not DEU) here because the 2023 release
# uses ISO alpha-2 in this file; we display 'US' not 'USA' in the output tables.
countries = {'USA': 'US', 'DE': 'DE', 'FR': 'FR', 'GB': 'GB', 'IT': 'IT'}

results = []

for country_code, country_label in countries.items():   # country loop
    print(f"\n--- Processing {country_label} ---")

    if country_code == 'USA':
        data_c = data_all.loc['USA']
    else:
        data_c = data_all.loc[country_code]

    data_sec = {}
    for s in sectors_all:
        df_s = data_c[data_c['sector'] == s]
        if len(df_s) > 0:
            data_sec[s] = df_s

    missing = [s for s in six_sectors + ['tot', 'ser'] if s not in data_sec]
    if missing:
        print(f"  Missing sectors: {missing}, skipping")
        continue

    years = sorted(data_sec['agr'].index.unique())
    n_years = len(years)
    print(f"  Years: {years[0]:.0f} - {years[-1]:.0f} ({n_years} years)")

    h_c = {}
    y_l_c = {}
    p_raw_c = {}
    A_raw_c = {}

    for s in six_sectors + ['ser', 'tot']:
        _, h_c[s] = sm.tsa.filters.hpfilter(data_sec[s]['H'], 100)
        yl_raw = (data_sec[s]['VA_Q'] / data_sec[s]['H']) * 100
        _, y_l_c[s] = sm.tsa.filters.hpfilter(yl_raw, 100)
        p_raw = data_sec[s]['VA'] / data_sec[s]['VA_Q']
        _, p_raw_c[s] = sm.tsa.filters.hpfilter(p_raw, 100)

    h_sum_6_c = sum(h_c[s] for s in six_sectors)
    share_data_c = {}
    for s in six_sectors:
        _, share_data_c[s] = sm.tsa.filters.hpfilter(h_c[s] / h_sum_6_c, 100)

    g_y_l_c = {}
    for s in six_sectors:
        g_y_l_c[s] = np.array(y_l_c[s] / y_l_c[s].shift(1) - 1)

    g_p_c = {}
    for s in six_sectors:
        g_p_c[s] = np.array(p_raw_c[s] / p_raw_c[s].shift(1) - 1)

    ts_length = n_years - 1

    A_norm = {}
    for s in six_sectors:
        A_norm[s] = [1.0]
        for i in range(ts_length):
            A_norm[s].append((1 + g_y_l_c[s][i+1]) * A_norm[s][i])

    P_norm = {}
    for s in six_sectors:
        P_norm[s] = [1.0]
        for i in range(ts_length):
            P_norm[s].append((1 + g_p_c[s][i+1]) * P_norm[s][i])

    l_ser_l_man_c = h_c['ser'] / h_c['man']
    p_ser_p_man_c = p_raw_c['ser'] / p_raw_c['man']
    C_tilde_c = C_index(om['ser'], l_ser_l_man_c, p_ser_p_man_c, sigma, eps['ser'])

    assert len(C_tilde_c) == n_years, f"C_tilde length {len(C_tilde_c)} != {n_years}"

    for t in range(n_years):   # time loop: compute model-implied shares year by year
        C_t = C_tilde_c[t]

        # Specification A — THEORETICAL: substitute p_i = 1/A_i into the CES share
        # formula. This is the specification used throughout the paper.
        # Weight: om_i * C^eps_i * A_i^(sigma-1)  (A^(sigma-1) replaces p^(1-sigma)).
        num_A = {}
        denom_A = 0
        for s in six_sectors:   # sector loop over 6 sectors
            val = om[s] * (C_t ** eps[s]) * (A_norm[s][t] ** (sigma - 1))
            num_A[s] = val
            denom_A += val

        # Specification B — EMPIRICAL: use the observed sectoral deflator P_i
        # directly. Weight: om_i * C^eps_i * P_i^(1-sigma). The test asks how
        # much implied employment shares move when we swap the model price for
        # its empirical analogue — small differences support the model's pricing
        # assumption (see Table 3).
        num_P = {}
        denom_P = 0
        for s in six_sectors:   # sector loop
            val = om[s] * (C_t ** eps[s]) * (P_norm[s][t] ** (1 - sigma))
            num_P[s] = val
            denom_P += val

        year_t = years[t]
        for s in six_sectors:
            share_1A = num_A[s] / denom_A
            share_P = num_P[s] / denom_P
            share_d = np.array(share_data_c[s])[t]

            results.append({
                'country': country_label,
                'sector': s,
                'year': year_t,
                'share_1overA': share_1A,
                'share_P': share_P,
                'share_data': share_d
            })

# Build DataFrame
df = pd.DataFrame(results)
df = df.sort_values(['country', 'sector', 'year']).reset_index(drop=True)

print(f"\nTotal observations: {len(df)}")
print(f"Countries: {df['country'].unique()}")
print(f"Sectors: {df['sector'].unique()}")


# ============================================================
# ANALYSIS — three metrics on the (A) vs (B) specification gap
# ============================================================
#
# Metric 1 (Panel A): direct level difference share_1/A - share_P. Mean |gap|,
#   max |gap|, std of signed gap — answers "how close are the two specs?".
# Metric 2 (Panel B): year-to-year first-differences — answers "do the specs
#   track the same short-run movements?" via correlation and MAE against data.
# Metric 3 (Panel C): cumulative differences from base year — answers "do the
#   specs track the same long-run trajectory?" same correlation/MAE logic.

df['diff_specs'] = df['share_1overA'] - df['share_P']
df['abs_diff_specs'] = np.abs(df['diff_specs'])

# First differences: year-on-year Δshare (country×sector); one value per year
# except the first (which is NaN after diff() and is skipped).
fd_rows = []
for c in ['US', 'DE', 'FR', 'GB', 'IT']:
    for s in six_sectors:
        mask = (df['country'] == c) & (df['sector'] == s)
        sub = df[mask].sort_values('year')
        d_1A = sub['share_1overA'].diff()
        d_P = sub['share_P'].diff()
        d_data = sub['share_data'].diff()
        for idx in sub.index[1:]:
            fd_rows.append({
                'country': c,
                'sector': s,
                'year': sub.loc[idx, 'year'],
                'dshare_1A': d_1A.loc[idx],
                'dshare_P': d_P.loc[idx],
                'dshare_data': d_data.loc[idx],
            })
df_fd = pd.DataFrame(fd_rows)

# Cumulative changes
cc_rows = []
for c in ['US', 'DE', 'FR', 'GB', 'IT']:
    for s in six_sectors:
        mask = (df['country'] == c) & (df['sector'] == s)
        sub = df[mask].sort_values('year')
        base_1A = sub['share_1overA'].iloc[0]
        base_P = sub['share_P'].iloc[0]
        base_data = sub['share_data'].iloc[0]
        for idx, row in sub.iterrows():
            cc_rows.append({
                'country': c,
                'sector': s,
                'year': row['year'],
                'cumchange_1A': row['share_1overA'] - base_1A,
                'cumchange_P': row['share_P'] - base_P,
                'cumchange_data': row['share_data'] - base_data,
            })
df_cc = pd.DataFrame(cc_rows)
df_cc_nonzero = df_cc[df_cc['cumchange_data'] != 0].copy()


# ============================================================
# BUILD TABLE 3: Summary comparison across specifications
# ============================================================

# Panel A: Direct differences (by country)
panel_a_rows = []
for c in ['US', 'DE', 'FR', 'GB', 'IT']:
    mask = df['country'] == c
    sub = df[mask]
    panel_a_rows.append({
        'Country': c,
        'Mean $|\\Delta|$': sub['abs_diff_specs'].mean(),
        'Max $|\\Delta|$': sub['abs_diff_specs'].max(),
        'Std($\\Delta$)': sub['diff_specs'].std(),
    })
panel_a_rows.append({
    'Country': 'All',
    'Mean $|\\Delta|$': df['abs_diff_specs'].mean(),
    'Max $|\\Delta|$': df['abs_diff_specs'].max(),
    'Std($\\Delta$)': df['diff_specs'].std(),
})
panel_a = pd.DataFrame(panel_a_rows).set_index('Country')

# Panel B: First differences correlations and MAE (by country)
panel_b_rows = []
for c in ['US', 'DE', 'FR', 'GB', 'IT']:
    sub = df_fd[df_fd['country'] == c]
    corr_specs = sub['dshare_1A'].corr(sub['dshare_P'])
    corr_1A_data = sub['dshare_1A'].corr(sub['dshare_data'])
    corr_P_data = sub['dshare_P'].corr(sub['dshare_data'])
    mae_1A = np.abs(sub['dshare_1A'] - sub['dshare_data']).mean()
    mae_P = np.abs(sub['dshare_P'] - sub['dshare_data']).mean()
    panel_b_rows.append({
        'Country': c,
        'Corr($\\Delta_{1/A}$, $\\Delta_P$)': corr_specs,
        'Corr($\\Delta_{1/A}$, data)': corr_1A_data,
        'Corr($\\Delta_P$, data)': corr_P_data,
        'MAE($1/A$)': mae_1A,
        'MAE($P$)': mae_P,
    })
sub_all = df_fd
panel_b_rows.append({
    'Country': 'All',
    'Corr($\\Delta_{1/A}$, $\\Delta_P$)': sub_all['dshare_1A'].corr(sub_all['dshare_P']),
    'Corr($\\Delta_{1/A}$, data)': sub_all['dshare_1A'].corr(sub_all['dshare_data']),
    'Corr($\\Delta_P$, data)': sub_all['dshare_P'].corr(sub_all['dshare_data']),
    'MAE($1/A$)': np.abs(sub_all['dshare_1A'] - sub_all['dshare_data']).mean(),
    'MAE($P$)': np.abs(sub_all['dshare_P'] - sub_all['dshare_data']).mean(),
})
panel_b = pd.DataFrame(panel_b_rows).set_index('Country')

# Panel C: Cumulative changes correlations and MAE (by country)
panel_c_rows = []
for c in ['US', 'DE', 'FR', 'GB', 'IT']:
    sub = df_cc_nonzero[df_cc_nonzero['country'] == c]
    corr_1A_data = sub['cumchange_1A'].corr(sub['cumchange_data'])
    corr_P_data = sub['cumchange_P'].corr(sub['cumchange_data'])
    corr_specs = sub['cumchange_1A'].corr(sub['cumchange_P'])
    mae_1A = np.abs(sub['cumchange_1A'] - sub['cumchange_data']).mean()
    mae_P = np.abs(sub['cumchange_P'] - sub['cumchange_data']).mean()
    panel_c_rows.append({
        'Country': c,
        'Corr($\\Delta^c_{1/A}$, data)': corr_1A_data,
        'Corr($\\Delta^c_P$, data)': corr_P_data,
        'Corr($\\Delta^c_{1/A}$, $\\Delta^c_P$)': corr_specs,
        'MAE($1/A$)': mae_1A,
        'MAE($P$)': mae_P,
    })
sub_cc = df_cc_nonzero
panel_c_rows.append({
    'Country': 'All',
    'Corr($\\Delta^c_{1/A}$, data)': sub_cc['cumchange_1A'].corr(sub_cc['cumchange_data']),
    'Corr($\\Delta^c_P$, data)': sub_cc['cumchange_P'].corr(sub_cc['cumchange_data']),
    'Corr($\\Delta^c_{1/A}$, $\\Delta^c_P$)': sub_cc['cumchange_1A'].corr(sub_cc['cumchange_P']),
    'MAE($1/A$)': np.abs(sub_cc['cumchange_1A'] - sub_cc['cumchange_data']).mean(),
    'MAE($P$)': np.abs(sub_cc['cumchange_P'] - sub_cc['cumchange_data']).mean(),
})
panel_c = pd.DataFrame(panel_c_rows).set_index('Country')

# Save outputs
os.makedirs('../output/tables', exist_ok=True)

# LaTeX output
with open('../output/tables/table_3.tex', 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Price Specification Comparison: $1/A_i$ vs.\\ Observed $P_i$}\n")
    f.write("\\label{tab:price_spec}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * 3 + "}\n")
    f.write("\\toprule\n")

    # Panel A
    f.write("\\multicolumn{4}{l}{\\textbf{Panel A: Direct differences in employment shares}} \\\\\n")
    f.write("\\midrule\n")
    f.write(" & Mean $|\\Delta|$ & Max $|\\Delta|$ & Std($\\Delta$) \\\\\n")
    f.write("\\midrule\n")
    for idx, row in panel_a.iterrows():
        f.write(f"{idx} & {row.iloc[0]:.4f} & {row.iloc[1]:.4f} & {row.iloc[2]:.4f} \\\\\n")

    f.write("\\midrule\n")
    f.write("\\multicolumn{4}{l}{\\textbf{Panel B: First differences}} \\\\\n")
    f.write("\\midrule\n")
    f.write(" & Corr($\\Delta_{1/A}$, $\\Delta_P$) & MAE($1/A$) & MAE($P$) \\\\\n")
    f.write("\\midrule\n")
    for idx, row in panel_b.iterrows():
        f.write(f"{idx} & {row.iloc[0]:.4f} & {row.iloc[3]:.6f} & {row.iloc[4]:.6f} \\\\\n")

    f.write("\\midrule\n")
    f.write("\\multicolumn{4}{l}{\\textbf{Panel C: Cumulative changes from base year}} \\\\\n")
    f.write("\\midrule\n")
    f.write(" & Corr($\\Delta^c_{1/A}$, $\\Delta^c_P$) & MAE($1/A$) & MAE($P$) \\\\\n")
    f.write("\\midrule\n")
    for idx, row in panel_c.iterrows():
        f.write(f"{idx} & {row.iloc[2]:.4f} & {row.iloc[3]:.6f} & {row.iloc[4]:.6f} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Excel output
with pd.ExcelWriter('../output/tables/table_3.xlsx') as writer:
    panel_a.to_excel(writer, sheet_name='Panel A - Direct Diff')
    panel_b.to_excel(writer, sheet_name='Panel B - First Diff')
    panel_c.to_excel(writer, sheet_name='Panel C - Cumulative')

print("\nTable 3 saved to ../output/tables/table_3.tex and ../output/tables/table_3.xlsx")


# ============================================================
# PRINT SUMMARY
# ============================================================

corr_all = df_fd['dshare_1A'].corr(df_fd['dshare_P'])
corr_1A_all = df_fd['dshare_1A'].corr(df_fd['dshare_data'])
corr_P_all = df_fd['dshare_P'].corr(df_fd['dshare_data'])
mae_1A_all = np.abs(df_fd['dshare_1A'] - df_fd['dshare_data']).mean()
mae_P_all = np.abs(df_fd['dshare_P'] - df_fd['dshare_data']).mean()
corr_1A_cc = df_cc_nonzero['cumchange_1A'].corr(df_cc_nonzero['cumchange_data'])
corr_P_cc = df_cc_nonzero['cumchange_P'].corr(df_cc_nonzero['cumchange_data'])
corr_spec = df_cc_nonzero['cumchange_1A'].corr(df_cc_nonzero['cumchange_P'])
mae_1A_cc = np.abs(df_cc_nonzero['cumchange_1A'] - df_cc_nonzero['cumchange_data']).mean()
mae_P_cc = np.abs(df_cc_nonzero['cumchange_P'] - df_cc_nonzero['cumchange_data']).mean()

print(f"""
Overall direct difference between specifications:
  Mean |share(1/A) - share(P)|  = {df['abs_diff_specs'].mean():.6f}  ({df['abs_diff_specs'].mean()*100:.3f} pp)
  Max  |share(1/A) - share(P)|  = {df['abs_diff_specs'].max():.6f}  ({df['abs_diff_specs'].max()*100:.3f} pp)
  Std  (share(1/A) - share(P))  = {df['diff_specs'].std():.6f}

First differences (pooled):
  Corr(d_share(1/A), d_share(P))       = {corr_all:.6f}
  Corr(d_share(1/A), d_share_data)     = {corr_1A_all:.6f}
  Corr(d_share(P),   d_share_data)     = {corr_P_all:.6f}
  MAE(d_share(1/A) vs data)            = {mae_1A_all:.6f}  ({mae_1A_all*100:.3f} pp)
  MAE(d_share(P)   vs data)            = {mae_P_all:.6f}  ({mae_P_all*100:.3f} pp)

Cumulative changes from base year (pooled):
  Corr(cumchange(1/A), cumchange_data)  = {corr_1A_cc:.6f}
  Corr(cumchange(P),   cumchange_data)  = {corr_P_cc:.6f}
  Corr(cumchange(1/A), cumchange(P))    = {corr_spec:.6f}
  MAE(cumchange(1/A) vs data)           = {mae_1A_cc:.6f}  ({mae_1A_cc*100:.3f} pp)
  MAE(cumchange(P)   vs data)           = {mae_P_cc:.6f}  ({mae_P_cc*100:.3f} pp)
""")
