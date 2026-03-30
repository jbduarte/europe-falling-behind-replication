##### Script to find correaltion between labor productivity and TFP

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(readxl)

# read data
growth_acc_db <- readRDS('../Data/Raw Data/growth accounts.rds')
lp_db <- read_excel('../Outputs/Data/lp_KLEMS_data.xlsx')

growth_acc_db <- growth_acc_db[growth_acc_db$var == 'LP2TFP_I', c('year', 'geo_code', 'nace_r2_name', 'var', 'value')]

sectors_dict = {"A": 'agr', "B": "man", "C": "man", "D-E": "man", "F": "man", "G": "trd", "I": "nps", "H": "nps",
  "J": "bss", "K": "fin", "L": "nps", "M-N": "bss", "O": "nps",
  "P": "nps", "Q": "nps", "R-S": "nps", "T": "nps", "TOT": "tot"}

data_temp.loc[data_temp.geo_code == 'UK', 'geo_code'] = 'GB'
data_temp.loc[data_temp.geo_code == 'EL', 'geo_code'] = 'GR'

country_filter = data_temp['geo_code'].isin(countries)
year_filter = (data_temp['year'] >= sample_period[0]) & (data_temp['year'] <= sample_period[1])

data_temp['sector'] = data_temp['nace_r2_code'].apply(lambda x: sectors_dict[x] if x in sectors_dict.keys() else x)
sector_filter = data_temp['sector'].isin(sectors_dict.values())

