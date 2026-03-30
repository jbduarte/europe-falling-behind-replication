"""
=======================================================================================
Description: This program downloads raw data from the EU KLEMS releases.
Author: Joao B. Duarte (joao.duarte@novasbe.pt)
=======================================================================================
"""

import pandas as pd
import os

def get_EUKLEMS(release="2021"):
    """
    Downloads a given release of the EUKLEMS data and saves the raw data in the Data/Raw Data folder

    Inputs
    ------

    release: string
    """

    if release == "2023":
        link = "https://www.dropbox.com/s/nkz7mdp0onken1j/national%20accounts.csv?dl=1"
        data_temp = pd.read_csv(link)

        # Create directory for saving data if it does not exist already
        parent_directory = os.getcwd()
        newpath = "../Data/Raw data/EUKLEMS_2023"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        data_temp = data_temp.iloc[:, 1:]
        data_temp.to_csv(newpath + '/data_raw.csv', index=False)


    if release == "2021":
        link = "https://www.dropbox.com/s/2coh5p5yp4t3k77/national%20accounts.csv?dl=1"
        data_temp = pd.read_csv(link)

        # Create directory for saving data if it does not exist already
        parent_directory = os.getcwd()
        newpath = "../Data/Raw data/EUKLEMS_2021"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        data_temp = data_temp.iloc[:, 2:]
        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    if release == "2017":
        link = "http://www.euklems.net/TCB/2018/ALL_output_17ii.txt"
        data_temp = pd.read_csv(link)

        # Create directory for saving data if it does not exist already
        parent_directory = os.getcwd()
        newpath = "../Data/Raw data/EUKLEMS_2017"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    if release == "2009":
        link = "https://web.archive.org/web/20210916113152fw_/http://euklems.net/data/09i/all_countries_09I.txt"
        data_temp = pd.read_csv(link)
        colnames = list(data_temp.columns)[0].split(',')
        data_temp = data_temp.squeeze().str.split(',', expand=True)
        data_temp.columns = colnames
        for j in data_temp.columns[3:]:
            data_temp[j] = pd.to_numeric(data_temp[j], errors='coerce')

        # Append the EU aggregates
        link_EU = "https://web.archive.org/web/20211105172749fw_/http://euklems.net/data/09i/aggregates_09I.txt"
        data_temp_EU = pd.read_csv(link_EU)

        data_temp = pd.concat((data_temp, data_temp_EU), axis=0)

        # Create directory for saving data if it does not exist already
        parent_directory = os.getcwd()
        newpath = "../Data/Raw data/EUKLEMS_2009"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        data_temp.to_csv(newpath + '/data_raw.csv', index=False)

    else:
        print("There is no " + release + "EUKLEMS available")

get_EUKLEMS(release="2023")