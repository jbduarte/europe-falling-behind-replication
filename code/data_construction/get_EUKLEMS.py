"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:    get_EUKLEMS.py
Purpose: Download raw EUKLEMS national-accounts panels (2009/2017/2021/2023
         releases) and persist them to data/raw_data/EUKLEMS_<release>/data_raw.csv.
Pipeline:    Data-build chain; run manually before master.py if regenerating
             data from sources. The main pipeline reads the cached csv written
             by select_data.py, so this download step is optional.

Source URLs (fragile — we mirror through our Dropbox for the 2021/2023 releases
because EUKLEMS 2023 is hosted at https://euklems-intanprod-llee.luiss.it/ behind
a landing-page redirect and the 2021 release was moved after initial download;
2017 uses the original euklems.net TCB path; 2009 uses archive.org snapshots
because euklems.net/data/09i was decommissioned).

Failure modes: any of these URLs may go dark. pandas.read_csv raises
URLError/HTTPError on 404/timeout — caller will see a stack trace. No retry
loop: a persistent failure means the source has moved and a human needs to
find the new location.
"""

import pandas as pd
import os

def get_EUKLEMS(release="2021"):
    """Download a given release of EUKLEMS and write the raw CSV under
    data/raw_data/EUKLEMS_<release>/data_raw.csv."""

    if release == "2023":
        # LUISS-hosted mirror redirected through our Dropbox for stability.
        link = "https://www.dropbox.com/s/nkz7mdp0onken1j/national%20accounts.csv?dl=1"
        data_temp = pd.read_csv(link)

        newpath = "../data/raw_data/EUKLEMS_2023"
        os.makedirs(newpath, exist_ok=True)

        # Drop the leading export index column that the 2023 release ships with.
        data_temp = data_temp.iloc[:, 1:]
        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    elif release == "2021":
        link = "https://www.dropbox.com/s/2coh5p5yp4t3k77/national%20accounts.csv?dl=1"
        data_temp = pd.read_csv(link)

        newpath = "../data/raw_data/EUKLEMS_2021"
        os.makedirs(newpath, exist_ok=True)

        # 2021 release ships with two leading helper columns (row id + export id).
        data_temp = data_temp.iloc[:, 2:]
        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    elif release == "2017":
        link = "http://www.euklems.net/TCB/2018/ALL_output_17ii.txt"
        data_temp = pd.read_csv(link)

        newpath = "../data/raw_data/EUKLEMS_2017"
        os.makedirs(newpath, exist_ok=True)

        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    elif release == "2009":
        # 2009 release lives only on archive.org; the all-countries file is a
        # single-column CSV whose only row *is* the comma-delimited content,
        # so we split it manually.
        link = "https://web.archive.org/web/20210916113152fw_/http://euklems.net/data/09i/all_countries_09I.txt"
        data_temp = pd.read_csv(link)
        colnames = list(data_temp.columns)[0].split(',')
        data_temp = data_temp.squeeze().str.split(',', expand=True)
        data_temp.columns = colnames
        for j in data_temp.columns[3:]:
            data_temp[j] = pd.to_numeric(data_temp[j], errors='coerce')

        # EU aggregates are shipped separately in the 2009 release.
        link_EU = "https://web.archive.org/web/20211105172749fw_/http://euklems.net/data/09i/aggregates_09I.txt"
        data_temp_EU = pd.read_csv(link_EU)
        data_temp = pd.concat((data_temp, data_temp_EU), axis=0)

        newpath = "../data/raw_data/EUKLEMS_2009"
        os.makedirs(newpath, exist_ok=True)

        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    else:
        print("There is no " + release + " EUKLEMS available")

get_EUKLEMS(release="2023")