"""
=======================================================================================
Description: This program downloads raw data from the EU KLEMS releases.
Author: Joao B. Duarte (joao.duarte@novasbe.pt)
=======================================================================================
"""

import pandas as pd
import os

def get_WorldKLEMS_USA(release="2013"):
    """
    Downloads a given release of the World KLEMS data and saves the raw data in the Data/Raw Data folder

    Inputs
    ------

    release: string
    """

    if release == "2013":
        link = "https://www.worldklems.net/data/basic/usa_wk_apr_2013.xlsx"
        data_temp = pd.read_excel(link, sheet_name='DATA')

        # Create directory for saving data if it does not exist already
        parent_directory = os.getcwd()
        newpath = "../Data/Raw data/World_KLEMS_2013"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        data_temp.to_csv(newpath + '/data_raw_usa.csv', index=False)

    else:
        print("There is no " + release + "EUKLEMS available")

get_WorldKLEMS_USA(release="2013")