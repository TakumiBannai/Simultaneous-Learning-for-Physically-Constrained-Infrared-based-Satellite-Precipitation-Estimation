#!/usr/bin/env python
# coding: utf-8
"""Download ERA5 dataset.
- Pre-requirement
1. Get your API key and Pase on .cdsapirc
2. Agree to the Terms of Use (my-page)
- Document (httpz1s://cds.climate.copernicus.eu/api-how-to)

- Usage
- arg_1 = data type
- arg_2 = year
- example: python LoadDataset.py total_column_cloud_liquid_water 2020
"""


import sys
import cdsapi

data_type = sys.argv[1]
year = sys.argv[2]
save_path = "./raw/"

print ("Start data downloading...: ", data_type, year)
print("Save at..", f"{data_type}_{year}.nc")

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': data_type,
        'year': year,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    f"{save_path}{data_type}_{year}.nc")
