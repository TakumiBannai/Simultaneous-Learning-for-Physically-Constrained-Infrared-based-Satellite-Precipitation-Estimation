# %%
import os
import glob
import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.dates as mdates

import sys
sys.path.append("../model/")
from eval import get_image
from eval import *


# %%
PERSIANN_CNN = "EnsembleMean/PERSIAN"
PERSIANN_CW = "EnsembleMean/PERSIAN_CW"
PERSIANN_MTL = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy"
PERSIANN_WES = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy(WES)"


def make_datelist(start="2013-08-01 00:00", end="2013-08-31 23:30", freq="30min"):
    datelists = pd.date_range(start=start, end=end, freq=freq)
    datelists = [datelists[i].strftime('%Y%m%d%H') for i in range(len(datelists))]
    return datelists


def validate_datelist(timelist):
    val_date, invalid_date = [], []
    for date in timelist:
        try:
            path = f"../data/stage4_net/crop/{date}.npy"
            arr = np.load(path)
            path = f"../data/prediction/{PERSIANN_MTL}/{date}.npy"
            arr = np.load(path)
            path = f"../data/persiann_ccs/crop/{date}.npy"
            arr = np.load(path)
            path = f"../data/imerg/crop_IRprecipitation/{date+'00'}.npy"
            arr = np.load(path)
            path = f"../data/imerg/crop_precipitationUncal/{date+'00'}.npy"
            arr = np.load(path)
            path = f"../data/gsmap/crop/gsmap_nrt.{date[:8]}.{date[8:]}00.npy"
            arr = np.load(path)
            path = f"../data/era5/crop_prcp/{date}00.npy"
            arr = np.load(path)
            val_date.append(date)
        except BaseException:
            invalid_date.append(date)
            # print(traceback.print_exc())
    return val_date, invalid_date


def load_as_1daray(timelist, dtype):
    arr_1d = []
    for date in timelist:
        if dtype == "stage4":
            path = f"../data/stage4_net/crop/{date}.npy"
        elif dtype == "pred_cnn":
            path = f"../data/prediction/{PERSIANN_CNN}/{date}.npy"
        elif dtype == "pred_cw":
            path = f"../data/prediction/{PERSIANN_CW}/{date}.npy"
        elif dtype == "pred_mtl":
            path = f"../data/prediction/{PERSIANN_MTL}/{date}.npy"
        elif dtype == "pred_wes":
            path = f"../data/prediction/{PERSIANN_WES}/{date}.npy"
        elif dtype == "persian_ccs":
            path = f"../data/persiann_ccs/crop/{date}.npy"
        elif dtype == "imerg_ir":
            path = f"../data/imerg/crop_IRprecipitation/{date+'00'}.npy"
        elif dtype == "imerg_uncal":
            path = f"../data/imerg/crop_precipitationUncal/{date+'00'}.npy"
        elif dtype == "gsmap":
            path = f"../data/gsmap/crop/gsmap_nrt.{date[:8]}.{date[8:]}00.npy"
        elif dtype == "era5":
            path = f"../data/era5/crop_prcp/{date}00.npy"
        arr = np.load(path).reshape(-1)
        arr_1d.append(arr)
    arr_1d = np.array(arr_1d).reshape(-1)
    return arr_1d

# %%
timelist = make_datelist(start="2013-08-01 00:00", end="2013-08-31 00:00", freq="1h")
val_datelist, invalid_datelist = validate_datelist(timelist)
print(len(timelist), len(val_datelist), len(invalid_datelist))

label = load_as_1daray(val_datelist, "stage4")
# pred_cnn = load_as_1daray(val_datelist, "pred_cnn")
# pred_cw = load_as_1daray(val_datelist, "pred_cw")
# pred_wes = load_as_1daray(val_datelist, "pred_wes")
pred_mtl = load_as_1daray(val_datelist, "pred_mtl")
# persian_ccs = load_as_1daray(val_datelist, "persian_ccs")
# imerg_ir = load_as_1daray(val_datelist, "imerg_ir")
# imerg_uncal = load_as_1daray(val_datelist, "imerg_uncal")
# gsmap = load_as_1daray(val_datelist, "gsmap")
# era5 = load_as_1daray(val_datelist, "era5")*1000

# %%
# データ数・割合・水量・水量割合
def compute_stats(pred_mtl, label, return_type="label"):
    """ Compute data size, total precipitation by bin

    Args:
        pred_mtl (float): estimated precipitaton (1d array)
        label (float): label precipitaton (1d array)

    Returns:
        dataframe: dataframe (size, prcp)
    """
    pred_norain, label_norain = binning(pred_mtl, label, bin="no_rain")
    pred_weak, label_weak = binning(pred_mtl, label, bin="weak")
    pred_mod, label_mod = binning(pred_mtl, label, bin="moderate")
    pred_strong, label_strong = binning(pred_mtl, label, bin="strong")
    if return_type == "label":
        arr_norain = label_norain
        arr_weak = label_weak
        arr_mod = label_mod
        arr_strong = label_strong
    elif return_type == "pred":
        arr_norain = label_norain
        arr_weak = pred_weak
        arr_mod = pred_mod
        arr_strong = pred_strong        

    sr_norain = pd.Series(arr_norain)
    sr_weak = pd.Series(arr_weak)
    sr_mod = pd.Series(arr_mod)
    sr_strong = pd.Series(arr_strong)
    df = pd.DataFrame([
        [sr_norain.size, sr_weak.size, sr_mod.size, sr_strong.size],
        [sr_norain.sum(), sr_weak.sum(), sr_mod.sum(), sr_strong.sum()]
        ],
        index=["sample", "total_prcp"],
        columns=["no_rain", "weak", "moderate", "heavy"]
    ).T
    return df
# %%
df_label = compute_stats(pred_mtl, label, "label")
df_pred = compute_stats(pred_mtl, label, "pred")
# %%


# %%
pd.concat([df_label, df_pred], axis=1).to_csv("output/statistics.csv")

# %%
