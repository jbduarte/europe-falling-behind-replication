"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        master.py
Purpose:     Orchestrate the full 19-step replication pipeline in the correct
             order, aborting on failure (except for two optional steps).
Pipeline:    Entry point — invokes every other script in the package.
Inputs:      All raw and cleaned data under ../data/ (assumes data-construction
             stage already populated ../data/euklems_2023.csv, ../data/io_panel.xlsx,
             ../data/exp_imp_aggregate_panel.xlsx, etc.).
Outputs:     All figures and tables to ../output/figures/ and ../output/tables/.
Dependencies: Python 3.x with pandas, numpy, scipy, statsmodels, scikit-learn,
             matplotlib (with working LaTeX for usetex=True), openpyxl, dill.

Pipeline steps (19):
     1. model_calibration_USA.py               — Calibrate closed-economy model on US
     2. model_test_europe.py                   — Apply closed-economy model to EU
     3. counterfactuals.py                     — Closed-economy counterfactuals
     4. model_calibration_USA_open.py          — Calibrate open-economy (exog. trade) on US
     5. model_test_europe_open.py              — Apply open-economy model to EU
     6. trade_counterfactuals.py               — Open-economy (exog. trade) counterfactuals
     7. model_calibration_USA_endogenous_open.py — Calibrate endogenous-trade model on US
     8. model_test_europe_endogenous_xn.py     — Apply endogenous-trade model to EU
     9. trade_counterfactuals_endogenous.py    — Endogenous-trade counterfactuals
    10. generate_fig_reallocation.py           — Figure 4 (labor reallocation)
    11. generate_fig_opennes.py                — Figure 5 (trade openness)
    12. price_specification_comparison.py      — Table 3 (price robustness)
    13. utils.facts                            — Figure 1 (motivating facts, optional)
    14. utils.table_1_ss_eu4                   — Table 1 (shift-share EU4)
    15. utils.cfs                              — Tables 2, A4, A8 (counterfactual tables)
    16. utils.table_ss_eu15_appendix           — Tables A6, A7
    17. utils.table_ss_core_vs_periphery       — Tables A9, A10
    18. utils.corr_lp_tfp_klems                — Table 4 (LP-TFP correlation, optional)
    19. generate_paper_outputs.py              — Consolidate all outputs with paper-consistent naming

Usage:
    cd code/
    python master.py
"""

import subprocess
import sys
import os
import time

# All scripts use paths relative to code/ (e.g. "../data/...") and assume that
# is the current working directory. Anchor here regardless of how the user
# invokes master.py (`python code/master.py`, IDE run button, etc.).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Each output folder is referenced by at least one downstream script. Create
# them up front so individual scripts can write without their own existence
# check (and so a fresh clone produces the expected layout).
for d in ["../output/figures", "../output/tables", "../output/data"]:
    os.makedirs(d, exist_ok=True)

scripts = [
    # ── Closed economy ──
    ("model_calibration_USA.py",              "Step  1/19: Calibrating baseline model (USA)"),
    ("model_test_europe.py",                  "Step  2/19: Testing model on European economies"),
    ("counterfactuals.py",                    "Step  3/19: Running counterfactual experiments"),

    # ── Open economy (exogenous trade) ──
    ("model_calibration_USA_open.py",         "Step  4/19: Calibrating open economy model (USA)"),
    ("model_test_europe_open.py",             "Step  5/19: Testing open economy model (Europe)"),
    ("trade_counterfactuals.py",              "Step  6/19: Running trade counterfactuals"),

    # ── Open economy (endogenous trade) ──
    ("model_calibration_USA_endogenous_open.py", "Step  7/19: Calibrating endogenous trade model (USA)"),
    ("model_test_europe_endogenous_xn.py",       "Step  8/19: Testing endogenous trade model (Europe)"),
    ("trade_counterfactuals_endogenous.py",      "Step  9/19: Running endogenous trade counterfactuals"),

    # ── Reallocation figure (requires Steps 1-3) ──
    ("generate_fig_reallocation.py",          "Step 10/19: Labor reallocation figure (Figure 4)"),
    ("generate_fig_opennes.py",               "Step 11/19: Trade openness figure (Figure 5)"),

    # ── Standalone analyses ──
    ("price_specification_comparison.py",     "Step 12/19: Price specification comparison (Table 3)"),
    ("-m utils.facts",                        "Step 13/19: Generating facts figures (Figure 1)"),
    ("-m utils.table_1_ss_eu4",              "Step 14/19: Shift-share decomposition (Table 1)"),
    ("-m utils.cfs",                         "Step 15/19: Counterfactual tables (Table 2, Table A.4, Table A.8)"),
    ("-m utils.table_ss_eu15_appendix",      "Step 16/19: EU-15 shift-share tables (Tables A.6, A.7)"),
    ("-m utils.table_ss_core_vs_periphery",  "Step 17/19: Core vs. periphery tables (Tables A.9, A.10)"),
    ("-m utils.corr_lp_tfp_klems",           "Step 18/19: LP--TFP correlation (Table 4, optional)"),

    # ── Paper outputs ──
    ("generate_paper_outputs.py",             "Step 19/19: Generating final paper outputs"),
]

print("=" * 70)
print("REPLICATION: Europe Falling Behind (Buiatti, Duarte, Sáenz)")
print("=" * 70)

total_start = time.time()
for i, (script, desc) in enumerate(scripts):
    print(f"\n{'─' * 70}")
    print(f"  {desc}")
    print(f"{'─' * 70}")
    t0 = time.time()
    if script.startswith("-m "):
        # utils/* steps are dispatched as modules so their relative imports
        # (`from .table_1_ss_eu4 import ...`) resolve against the utils package.
        module = script[3:]
        result = subprocess.run([sys.executable, "-m", module])
    else:
        result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode != 0:
        # facts.py depends on the OECD GDP-per-hour file (Figure 1) and
        # corr_lp_tfp_klems.py on the ~180 MB EU KLEMS Growth Accounts CSV
        # (Table 4). Neither is shipped with the replication package, so a
        # missing-file failure here is expected and must not abort the run.
        if script in ("-m utils.facts", "-m utils.corr_lp_tfp_klems"):
            print(f"\n  WARNING: {script} failed (exit code {result.returncode}). Continuing.")
        else:
            print(f"\n  ERROR: {script} failed (exit code {result.returncode})")
            print("  Aborting replication.")
            sys.exit(1)
    print(f"  Completed in {elapsed:.1f}s")

total_elapsed = time.time() - total_start
minutes = int(total_elapsed // 60)
seconds = int(total_elapsed % 60)
print(f"\n{'=' * 70}")
print(f"  All steps completed successfully in {minutes}m {seconds}s")
print("  Outputs saved to: ../output/")
print(f"{'=' * 70}")
