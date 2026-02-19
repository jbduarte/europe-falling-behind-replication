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
    ("model_calibration_USA.py",              "Step  1/17: Calibrating baseline model (USA)"),
    ("model_test_europe.py",                  "Step  2/17: Testing model on European economies"),
    ("counterfactuals.py",                    "Step  3/17: Running counterfactual experiments"),

    # ── Open economy (exogenous trade) ──
    ("model_calibration_USA_open.py",         "Step  4/17: Calibrating open economy model (USA)"),
    ("model_test_europe_open.py",             "Step  5/17: Testing open economy model (Europe)"),
    ("trade_counterfactuals.py",              "Step  6/17: Running trade counterfactuals"),

    # ── Open economy (endogenous trade) ──
    ("model_calibration_USA_endogenous_open.py", "Step  7/17: Calibrating endogenous trade model (USA)"),
    ("model_test_europe_endogenous_xn.py",       "Step  8/17: Testing endogenous trade model (Europe)"),
    ("trade_counterfactuals_endogenous.py",      "Step  9/17: Running endogenous trade counterfactuals"),

    # ── Standalone analyses ──
    ("price_specification_comparison.py",     "Step 10/17: Price specification comparison (Table 3)"),
    ("-m utils.facts",                        "Step 11/17: Generating facts figures (Figure 1)"),
    ("-m utils.table_1_ss_eu4",              "Step 12/17: Shift-share decomposition (Table 1)"),
    ("-m utils.cfs",                         "Step 13/17: Counterfactual tables (Table 2, Table A.4, Table A.8)"),
    ("-m utils.table_ss_eu15_appendix",      "Step 14/17: EU-15 shift-share tables (Tables A.6, A.7)"),
    ("-m utils.table_ss_core_vs_periphery",  "Step 15/17: Core vs. periphery tables (Tables A.9, A.10)"),
    ("-m utils.corr_lp_tfp_klems",           "Step 16/17: LP--TFP correlation (Table 4, optional)"),

    # ── Paper outputs ──
    ("generate_paper_outputs.py",             "Step 17/17: Generating final paper outputs"),
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
