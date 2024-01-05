# %%
# GridSat: https://www.ncdc.noaa.gov/gridsat/conusgoes-index.php?name=variables
import os
import glob
import subprocess
import datetime
import pandas as pd

def sh(command = "pwd", output=True):
    cp = subprocess.run(command.split(), encoding='utf-8', stdout=subprocess.PIPE)
    if output is False:
        pass
    else:
        print(cp.stdout)

def create_url(target_time):
    # target_time = datetime.datetime.strptime(target_time,"%Y-%m-%d %H:%M")
    conus = "https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/"
    url = f"{conus}{target_time.year}/{target_time.month:02}/GridSat-CONUS.goes13.{target_time.year}.{target_time.month:02}.{target_time.day:02}.{target_time.time().strftime('%H%M')}.v01.nc"
    return url

def iter_dataloading(date_list):
    for i in range(len(date_list)):
        url = create_url(date_list[i])
        print(f"Start... {url[-22:]}")
        sh(f"wget -P ./raw {url}")
        print(f"Done... {url[-22:]}")

def get_datetime(file_name):
    year = file_name[-22:-18]
    month = file_name[-17:-15]
    day = file_name[-14:-12]
    return year, month, day

def move_files(target_file):
    year, month, day = get_datetime(target_file)
    target_dir = f"./raw/{year}/{month+day}"
    os.makedirs(target_dir, exist_ok=True)
    sh(f"mv {target_file} {target_dir}")
    print("Move: ", target_file, "to", target_dir)

# %%
# Run
date_list = pd.date_range(start="2012-06-01 00:00",
                          end="2012-06-01 00:45",
                          freq="15min").to_list()
iter_dataloading(date_list)

# Organise Dir. structure
files = glob.glob("./raw/*.nc")
[move_files(files[i]) for i in range(len(files))]

# %%



