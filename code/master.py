"""
Master replication script for:
"Europe Falling Behind: Structural Transformation and Labor Productivity
Growth Differences Between Europe and the U.S."

Author: Joao B. Duarte
Last Modified: Feb 2026

Usage:
    cd code/
    python master.py

Runs all scripts in sequence. Each script builds on results from prior ones.
"""

import subprocess
import sys
import os
import time

# Ensure we're running from the code/ directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create output directories if they don't exist
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
print("REPLICATION: Europe Falling Behind (Buiatti, Duarte, Saenz)")
print("=" * 70)

total_start = time.time()
for i, (script, desc) in enumerate(scripts):
    print(f"\n{'─' * 70}")
    print(f"  {desc}")
    print(f"{'─' * 70}")
    t0 = time.time()
    if script.startswith("-m "):
        # Run as module (e.g., "-m utils.facts")
        module = script[3:]
        result = subprocess.run([sys.executable, "-m", module])
    else:
        result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode != 0:
        # Steps 11 and 16 are optional (facts figures and LP-TFP correlation)
        if script in ("-m utils.facts", "-m utils.corr_lp_tfp_klems"):
            print(f"\n  WARNING: {script} failed (exit code {result.returncode}). Continuing.")
        else:
            print(f"\n  ERROR: {script} failed (exit code {result.returncode})")
            print(f"  Aborting replication.")
            sys.exit(1)
    print(f"  Completed in {elapsed:.1f}s")

total_elapsed = time.time() - total_start
minutes = int(total_elapsed // 60)
seconds = int(total_elapsed % 60)
print(f"\n{'=' * 70}")
print(f"  All steps completed successfully in {minutes}m {seconds}s")
print(f"  Outputs saved to: ../output/")
print(f"{'=' * 70}")
