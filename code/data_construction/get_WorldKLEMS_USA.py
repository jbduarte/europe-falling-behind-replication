"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:    get_WorldKLEMS_USA.py
Purpose: Download the World KLEMS U.S. release (2013 vintage) and persist the
         'DATA' sheet as CSV under data/raw_data/World_KLEMS_2013/.
Pipeline:    Data-build chain; optional. The main pipeline reads pre-built
             csv files under data/, not this output directly.

Source: https://www.worldklems.net/data/basic/usa_wk_apr_2013.xlsx. Worldklems
mirrors the U.S. BEA/BLS growth-accounts data; this is the canonical U.S. file
used for the pre-1970 extension.
"""

import pandas as pd
import os

def get_WorldKLEMS_USA(release="2013"):
    """Download the given World KLEMS U.S. release and write the 'DATA' sheet
    as CSV under data/raw_data/World_KLEMS_<release>/data_raw_usa.csv."""

    if release == "2013":
        link = "https://www.worldklems.net/data/basic/usa_wk_apr_2013.xlsx"
        data_temp = pd.read_excel(link, sheet_name='DATA')

        newpath = "../data/raw_data/World_KLEMS_2013"
        os.makedirs(newpath, exist_ok=True)

        data_temp.to_csv(newpath + '/data_raw_usa.csv', index=False)

    else:
        print("There is no " + release + " World KLEMS release available")

get_WorldKLEMS_USA(release="2013")