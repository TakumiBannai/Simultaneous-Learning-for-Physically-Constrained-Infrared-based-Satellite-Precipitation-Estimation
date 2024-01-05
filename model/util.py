#!/usr/bin/env python
# coding: utf-8

import random
import glob
import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.enums import Resampling
import geopandas as gpd
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.dates as mdates
from tqdm import tqdm
import traceback
import torch
import torch.nn as nn

# 乱数シード固定
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def retrieve_result(device, model, dataloader, input_type="IR_WV", MTL=False, MultiInput=False):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for y, x in dataloader:
            y, x = y.to(device).float(), x.to(device).float()
            wv = x[:, 0:2, :, :]
            ir = x[:, 2:4, :, :]
            cw = x[:, 4:6, :, :]
            ci = x[:, 6:8, :, :]
            if input_type == "IR_WV":
                if MTL is False:
                    y_hat = model(ir, wv)
                elif MTL is True:
                    if MultiInput is False:
                        y_hat, mask_hat = model(ir, wv)
                    elif MultiInput is True:
                        y_hat, mask_hat = model(ir, wv)

            elif input_type == "CW":
                if MTL is False:
                    y_hat = model(ir, wv, cw)
                elif MTL is True:
                    if MultiInput is False:
                        y_hat, mask_hat, y_cw = model(ir, wv)
                    elif MultiInput is True:
                        y_hat, mask_hat, y_cw = model(ir, wv, cw)

            elif input_type == "CI":
                if MTL is False:
                    y_hat = model(ir, wv, ci)
                elif MTL is True:
                    if MultiInput is False:
                        y_hat, mask_hat, y_ci = model(ir, wv)
                    elif MultiInput is True:
                        y_hat, mask_hat, y_ci = model(ir, wv, ci)

            elif input_type == "CWCI":
                if MTL is False:
                    y_hat = model(ir, wv, cw, ci)
                elif MTL is True:
                    if MultiInput is False:
                        y_hat, mask_hat, y_cw, y_ci = model(ir, wv)
                    elif MultiInput is True:
                        y_hat, mask_hat, y_cw, y_ci = model(ir, wv, cw, ci)
            preds.append(y_hat)
            labels.append(y)
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    return preds, labels


# Train, val patch visualization
def quickcheck(model, device, dataloader, index, input_type="IR_WV", MTL=False):
    model.eval()
    with torch.no_grad():
        y, x = dataloader.dataset[index]
        x = x.to(device).float()
        x = x.unsqueeze(0)
        wv = x[:, 0:2, :, :]
        ir = x[:, 2:4, :, :]
        cw = x[:, 4:6, :, :]
        ci = x[:, 6:8, :, :]
        if input_type == "IR_WV":
            if MTL is False:
                y_hat = model(ir, wv)
            elif MTL is True:
                y_hat, mask_hat = model(ir, wv)
        elif input_type == "CW":
            if MTL is False:
                y_hat = model(ir, wv, cw)
            elif MTL is True:
                y_hat, mask_hat, cw = model(ir, wv)
        elif input_type == "CI":
            if MTL is False:
                y_hat = model(ir, wv, ci)
            elif MTL is True:
                y_hat, mask_hat, ci = model(ir, wv)
        elif input_type == "CWCI":
            if MTL is False:
                y_hat = model(ir, wv, cw, ci)
            elif MTL is True:
                y_hat, mask_hat, cw, ci = model(ir, wv)
    return ir.cpu(), wv.cpu(), y_hat.cpu(), y

def quickshow(model, device, dataloader, index, input_type="IR_WV", MTL=False):
    ir, wv, pred, label = quickcheck(model, device, dataloader, index, input_type, MTL)
    # Plot
    plt.figure(figsize=(35, 7))
    plt.subplot(141)
    plt.imshow(pred[0][0]);plt.colorbar(); plt.title("Pred")
    plt.subplot(142)
    plt.imshow(label[0]);plt.colorbar(); plt.title("Label")
    plt.subplot(143)
    plt.imshow(ir[0, 0]);plt.colorbar(); plt.title("IR")
    plt.subplot(144)
    plt.imshow(wv[0, 0]);plt.colorbar(); plt.title("WV")


def evaluation_matrix_comparison():
    df = pd.DataFrame()
    dir = glob.glob("./output/*/Evaluation/index_eval.csv")
    # index_eval.csv loop
    for i in range(len(dir)):
        index = dir[i].split("/")[2]
        table = pd.read_csv(dir[i])
        table.index = [index]
        df = pd.concat([df, table], axis=0)
    # dir = glob.glob("./output/*/Evaluation/index_eval_raininly.csv")
    # index_eva_raininlyl.csv loop
    # for i in range(len(dir)):
    #     index = dir[i].split("/")[2]
    #     table = pd.read_csv(dir[i])
    #     table.index = [index]
    #     df = pd.concat([df, table], axis=0)
    return df


# Image reconstruction
def reconst_patch(patch, original_image_shape = (448, 448), window_shape = (112, 112)):
    num_patch_h = int(original_image_shape[0] / window_shape[0])
    num_patch_w = int(original_image_shape[1] / window_shape[1])
    patch = patch.reshape(num_patch_h, num_patch_w, window_shape[0], window_shape[1])
    reconst = []
    for i in range(patch.shape[0]):
        reconst.append(np.hstack(patch[i, :, :, :]))
    reconst = np.vstack(reconst)
    return reconst

def get_image(dtype_index=0, target_dir = "../data/dataset_64_full/test/2013080100"):
    """
    dtype_index = {0: precp, 1-2: wv, 3-4: ir, 5-6: cw, 7-8:ci}
    """
    patch = []
    for i in range(1, 17):# Iter 16(4*4) patch for a image
        arr = np.load(f"{target_dir}/{i}.npy")[dtype_index]# Get Precp.
        patch.append(arr)
    patch = np.array(patch)
    image = reconst_patch(patch, original_image_shape = (448, 448), window_shape = (112, 112))
    return image

def vis_image(pred_image, label_image, wv_image, ir_image, cw_image, ci_image, prcp_max=None):
    # Visualization
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(17, 10))
    # Pred
    sc = ax[0, 0].imshow(pred_image, cmap="jet", vmin=0, vmax=prcp_max)
    ax[0, 0].set_title("Precip (Prediction)")
    fig.colorbar(sc, ax=ax[0, 0])
    # Label
    sc = ax[0, 1].imshow(label_image, cmap="jet", vmin=0, vmax=prcp_max)
    ax[0, 1].set_title("Precip (Label)")
    fig.colorbar(sc, ax=ax[0, 1])
    # WV
    sc = ax[0, 2].imshow(wv_image, cmap="inferno")
    ax[0, 2].set_title("WV")
    fig.colorbar(sc, ax=ax[0, 2])
    # IR
    sc = ax[1, 0].imshow(ir_image, cmap="inferno")
    ax[1, 0].set_title("IR")
    fig.colorbar(sc, ax=ax[1, 0])
    # CW
    sc = ax[1, 1].imshow(cw_image, cmap="Blues")
    ax[1, 1].set_title("Cloud Water")
    fig.colorbar(sc, ax=ax[1, 1])
    # CI
    sc = ax[1, 2].imshow(ci_image, cmap="Blues")
    ax[1, 2].set_title("Cloud Ice")
    fig.colorbar(sc, ax=ax[1, 2])

def get_images(target_date = "2013080311", save_dir = "./output/PERSIAN"):
    # Get image
    pred_image = get_image(dtype_index=0, target_dir=f"{save_dir}/Evaluation/dataset_112_full/test/{target_date}")
    label_image = get_image(dtype_index=0, target_dir=f"../data/dataset_112_full/test/{target_date}")
    wv_image = get_image(dtype_index=2, target_dir=f"../data/dataset_112_full/test/{target_date}")
    ir_image = get_image(dtype_index=4, target_dir=f"../data/dataset_112_full/test/{target_date}")
    cw_image = get_image(dtype_index=5, target_dir=f"../data/dataset_112_full/test/{target_date}")
    ci_image = get_image(dtype_index=7, target_dir=f"../data/dataset_112_full/test/{target_date}")

    # Visualize
    vis_image(pred_image, label_image, wv_image, ir_image, cw_image, ci_image, prcp_max=5)
    plt.show()

    # Diff
    diff = pred_image - label_image
    plt.imshow(diff, cmap="bwr", vmin=-20, vmax=20)
    plt.colorbar()
    plt.title("Diff: Pred - Label")
    plt.show()


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def make_datelist(start="2013-08-01 00:00", end="2013-08-31 23:30", freq="30min"):
    datelists = pd.date_range(start=start, end=end, freq=freq)
    datelists = [datelists[i].strftime('%Y%m%d%H') for i in range(len(datelists))]
    return datelists


def load_as_1daray(timelist, dtype):
    base_path = "/work/hk03/bannai/PGML/data"
    PERSIANN_CNN = "EnsembleMean/PERSIAN"
    PERSIANN_CW = "EnsembleMean/PERSIAN_CW"
    PERSIANN_MTL = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy"
    PERSIANN_WES = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy(WES)"
    arr_1d = []
    for date in timelist:
        if dtype == "stage4":
            path = f"{base_path}/stage4_net/crop/{date}.npy"
        elif dtype == "pred_cnn":
            path = f"{base_path}/prediction/{PERSIANN_CNN}/{date}.npy"
        elif dtype == "pred_cw":
            path = f"{base_path}/prediction/{PERSIANN_CW}/{date}.npy"
        elif dtype == "pred_mtl":
            path = f"{base_path}/prediction/{PERSIANN_MTL}/{date}.npy"
        elif dtype == "pred_wes":
            path = f"{base_path}/prediction/{PERSIANN_WES}/{date}.npy"
        elif dtype == "persian_ccs":
            path = f"{base_path}/persiann_ccs/crop/{date}.npy"
        elif dtype == "imerg_ir":
            path = f"{base_path}/imerg/crop_IRprecipitation/{date+'00'}.npy"
        elif dtype == "imerg_uncal":
            path = f"{base_path}/imerg/crop_precipitationUncal/{date+'00'}.npy"
        elif dtype == "gsmap":
            path = f"{base_path}/gsmap/crop/gsmap_nrt.{date[:8]}.{date[8:]}00.npy"
        elif dtype == "era5":
            path = f"{base_path}/era5/crop_prcp/{date}00.npy"
        arr = np.load(path).reshape(-1)
        arr_1d.append(arr)
    arr_1d = np.array(arr_1d).reshape(-1)
    return arr_1d


def validate_datelist(timelist):
    base_path = "/work/hk03/bannai/PGML/data"
    PERSIANN_MTL = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy"
    val_date, invalid_date = [], []
    for date in timelist:
        try:
            path = f"{base_path}/stage4_net/crop/{date}.npy"
            arr = np.load(path)
            path = f"{base_path}/prediction/{PERSIANN_MTL}/{date}.npy"
            arr = np.load(path)
            path = f"{base_path}/persiann_ccs/crop/{date}.npy"
            arr = np.load(path)
            path = f"{base_path}/imerg/crop_IRprecipitation/{date+'00'}.npy"
            arr = np.load(path)
            path = f"{base_path}/imerg/crop_precipitationUncal/{date+'00'}.npy"
            arr = np.load(path)
            path = f"{base_path}/gsmap/crop/gsmap_nrt.{date[:8]}.{date[8:]}00.npy"
            arr = np.load(path)
            path = f"{base_path}/era5/crop_prcp/{date}00.npy"
            arr = np.load(path)
            val_date.append(date)
        except BaseException:
            invalid_date.append(date)
            # print(traceback.print_exc())
    return val_date, invalid_date


def read_image(date = '2013080506'):
    base_path = "/work/hk03/bannai/PGML/data"
    PERSIANN_CNN = "EnsembleMean/PERSIAN"
    PERSIANN_CW = "EnsembleMean/PERSIAN_CW"
    PERSIANN_MTL = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy"
    PERSIANN_WES = "EnsembleMean/PERSIAN_MTMI_CW_Weighting_Strategy(WES)"
    path_label = f"{base_path}/stage4_net/crop/{date}.npy"
    label = np.load(path_label)

    pred_cnn = np.load(f"{base_path}/prediction/{PERSIANN_CNN}/{date}.npy")
    pred_cw = np.load(f"{base_path}/prediction/{PERSIANN_CW}/{date}.npy")
    pred_mtl = np.load(f"{base_path}/prediction/{PERSIANN_MTL}/{date}.npy")
    pred_wes = np.load(f"{base_path}/prediction/{PERSIANN_WES}/{date}.npy")

    path_persian = f"{base_path}/persiann_ccs/crop/{date}.npy"
    persiann_ccs = np.load(path_persian)
    path_imerg_ir = f"{base_path}/imerg/crop_IRprecipitation/{date+'00'}.npy"
    imerg_ir = np.load(path_imerg_ir)
    path_imerg_uncal = f"{base_path}/imerg/crop_precipitationUncal/{date+'00'}.npy"
    imerg_uncl = np.load(path_imerg_uncal)
    path_gsmap = f"{base_path}/gsmap/crop/gsmap_nrt.{date[:8]}.{date[8:]}00.npy"
    imerg_gsmap = np.load(path_gsmap)
    path_era5 = f"{base_path}/era5/crop_prcp/{date}00.npy"
    era5 = np.load(path_era5)*1000
    return label, pred_cnn, pred_cw, pred_mtl, persiann_ccs, imerg_ir, imerg_uncl, imerg_gsmap, era5


def show_geoplots(images, vmax=8, diff_vmin=-20, dff_vmax=20, cbar=False):
    """Visuzalize spatial plot

    Args:
        images (_type_): _description_
        vmax (int, optional): _description_. Defaults to 8.
        diff_vmin (int, optional): _description_. Defaults to -20.
        dff_vmax (int, optional): _description_. Defaults to 20.
        cbar (bool, optional): _description_. Defaults to False.
    Note:
        image index:
        label, pred_cnn, pred_cw, pred_mtl, persiann_ccs, imerg_ir, imerg_uncl, imerg_gsmap, era5
    """
    label, pred_cnn, pred_mtl = images[0], images[1], images[3]
    persiann_ccs, era5 = images[4], images[8]
    # persiann_ccs, era5 = images[6], images[7]

    diff_cnn = pred_cnn - label
    diff_mtl = pred_mtl - label
    diff_persian = persiann_ccs - label
    diff_era5 = era5 - label

    plt.figure(figsize=(25, 10), dpi=300)
    ax = plt.subplot(2, 5, 1, projection=ccrs.PlateCarree())
    geoplot(ax, pred_cnn, "Blues", "PERSIANN-CNN", vmin=0, vmax=vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 2, projection=ccrs.PlateCarree())
    geoplot(ax, pred_mtl, "Blues", "PERSIANN-MTL", vmin=0, vmax=vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 3, projection=ccrs.PlateCarree())
    geoplot(ax, persiann_ccs, "Blues", "PERSIAN-CCS", vmin=0, vmax=vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 4, projection=ccrs.PlateCarree())
    geoplot(ax, era5, "Blues", "ERA5", vmin=0, vmax=vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 5, projection=ccrs.PlateCarree())
    geoplot(ax, label, "Blues", "Precp", vmin=0, vmax=vmax, cbar=cbar)

    ax = plt.subplot(2, 5, 6, projection=ccrs.PlateCarree())
    geoplot(ax, diff_cnn, "seismic", "Diff (PERSIANN-CNN)", vmin=diff_vmin, vmax=dff_vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 7, projection=ccrs.PlateCarree())
    geoplot(ax, diff_mtl, "seismic", "Diff (PERSIANN-MTL)", vmin=diff_vmin, vmax=dff_vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 8, projection=ccrs.PlateCarree())
    geoplot(ax, diff_persian, "seismic", "Diff (PERSIAN-CCS)", vmin=diff_vmin, vmax=dff_vmax, cbar=cbar)
    ax = plt.subplot(2, 5, 9, projection=ccrs.PlateCarree())
    geoplot(ax, diff_era5, "seismic", "Diff (ERA5)", vmin=diff_vmin, vmax=dff_vmax, cbar=cbar)


def create_timestamp(time_t = "2012-06-01 12"):
    time_t = datetime.datetime.strptime(time_t, "%Y-%m-%d %H")
    # Compute time-delta
    time_t_minus1 = time_t - datetime.timedelta(hours=1)
    time_t_minus1_15 = time_t_minus1 + datetime.timedelta(minutes=15)
    time_t_minus1_45 = time_t_minus1 + datetime.timedelta(minutes=45)
    # Format
    time_t = time_t.strftime("%Y%m%d%H")
    time_t_minus1 = time_t_minus1.strftime("%Y%m%d%H")
    time_t_minus1_15 = time_t_minus1_15.strftime("%Y%m%d%H%M")
    time_t_minus1_45 = time_t_minus1_45.strftime("%Y%m%d%H%M")
    return time_t, time_t_minus1, time_t_minus1_15, time_t_minus1_45


def create_image(time_t = "2013-08-16 09"):
    time_t, time_t_minus1, time_t_minus1_15, time_t_minus1_45 = create_timestamp(time_t)
    base_path = "/work/hk03/bannai/PGML/data"
    
    # Path to TIF
    path_stage4 = f"{base_path}/stage4_net/merged_tif/{time_t[:6]}/{time_t}.tif"
    path_goes_ch3_15 = f"{base_path}/goes/TIF_CH3/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
    path_goes_ch3_45 = f"{base_path}/goes/TIF_CH3/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
    path_goes_ch4_15 = f"{base_path}/goes/TIF_CH4/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
    path_goes_ch4_45 = f"{base_path}/goes/TIF_CH4/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
    path_era5_cw_minus1 = f"{base_path}/era5/TIF_CW/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
    path_era5_ci_minus1 = f"{base_path}/era5/TIF_CI/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
    path_era5_cw = f"{base_path}/era5/TIF_CW/{time_t[:6]}/{time_t}00.tif"
    path_era5_ci = f"{base_path}/era5/TIF_CI/{time_t[:6]}/{time_t}00.tif"
    path_era5_prcp = f"{base_path}/era5/TIF_PRCP/{time_t[:6]}/{time_t}00.tif"
    path_aoi = f"{base_path}/AOI/AOI_CONUS_Center.geojson"
    aoi = gpd.read_file(path_aoi)

    # Get array
    arr_stage4 = read_crop_as_array(path_stage4, aoi, 448, 448, True)
    arr_goes_ch3_15 = read_crop_as_array(path_goes_ch3_15, aoi, 448, 448, True)
    arr_goes_ch3_45 = read_crop_as_array(path_goes_ch3_45, aoi, 448, 448, True)
    arr_goes_ch4_15 = read_crop_as_array(path_goes_ch4_15, aoi, 448, 448, True)
    arr_goes_ch4_45 = read_crop_as_array(path_goes_ch4_45, aoi, 448, 448, True)
    arr_era5_cw_minus1 = read_crop_as_array(path_era5_cw_minus1, aoi, 448, 448, True)
    arr_era5_cw = read_crop_as_array(path_era5_cw, aoi, 448, 448, True)
    arr_era5_ci_minus1 = read_crop_as_array(path_era5_ci_minus1, aoi, 448, 448, True)
    arr_era5_ci = read_crop_as_array(path_era5_ci, aoi, 448, 448, True)

    feature = np.concatenate((arr_stage4,
                              arr_goes_ch3_15, arr_goes_ch3_45, 
                              arr_goes_ch4_15, arr_goes_ch4_45,
                              arr_era5_cw, arr_era5_cw_minus1,
                              arr_era5_ci, arr_era5_ci_minus1
                              ), axis=0)
    return feature

def check_validtime(target_times):
    base_path = "/work/hk03/bannai/PGML/data"    
    valid_times = []
    for target_time in tqdm(target_times):
        try:
            time_t, time_t_minus1, time_t_minus1_15, time_t_minus1_45 = create_timestamp(target_time.strftime("%Y-%m-%d %H"))  
            # Path to TIF
            path_stage4 = f"{base_path}/stage4_net/merged_tif/{time_t[:6]}/{time_t}.tif"
            path_goes_ch3_15 = f"{base_path}/goes/TIF_CH3/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
            path_goes_ch3_45 = f"{base_path}/goes/TIF_CH3/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
            path_goes_ch4_15 = f"{base_path}/goes/TIF_CH4/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
            path_goes_ch4_45 = f"{base_path}/goes/TIF_CH4/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
            path_era5_cw_minus1 = f"{base_path}/era5/TIF_CW/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
            path_era5_ci_minus1 = f"{base_path}/era5/TIF_CI/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
            path_era5_cw = f"{base_path}/era5/TIF_CW/{time_t[:6]}/{time_t}00.tif"
            path_era5_ci = f"{base_path}/era5/TIF_CI/{time_t[:6]}/{time_t}00.tif"
            path_era5_prcp = f"{base_path}/era5/TIF_PRCP/{time_t[:6]}/{time_t}00.tif"

            # Read test
            with rasterio.open(path_stage4) as src:
                pass
            with rasterio.open(path_goes_ch3_15) as src:
                pass
            with rasterio.open(path_goes_ch3_45) as src:
                pass
            with rasterio.open(path_goes_ch4_15) as src:
                pass
            with rasterio.open(path_goes_ch4_45) as src:
                pass
            with rasterio.open(path_era5_cw_minus1) as src:
                pass
            with rasterio.open(path_era5_ci_minus1) as src:
                pass
            with rasterio.open(path_era5_cw) as src:
                pass
            with rasterio.open(path_era5_ci) as src:
                pass
            
            # Collect valid_date
            valid_times.append(target_time)
            
        except:
            traceback.print_exc()
            pass
            
    return valid_times

# Crop dataset
def crop_tif(input_tif, output_tif, aoi):
    with rasterio.open(input_tif) as src:
        out_image, out_transform = rasterio.mask.mask(src, aoi["geometry"],
                                                      crop=True, nodata=-99999,
                                                     filled=False)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)


def resample_tif(input_tif, output_tif, h_pixel, w_pixel):
    with rasterio.open(input_tif) as dataset:
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(h_pixel),
                int(w_pixel)),
            resampling=Resampling.bilinear)
        # scale image transform
        out_transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2]))
        out_meta = dataset.meta
    out_meta.update({"driver": "GTiff",
                         "height": data.shape[1],
                         "width": data.shape[2],
                         "transform": out_transform})
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(data)


def read_crop_as_array(path, aoi, h_pixel=None, w_pixel=None, resample=False):
    crop_tif(path, "temp.tif", aoi)
    if resample is True:
        resample_tif("temp.tif", "temp.tif", h_pixel, w_pixel)
    with rasterio.open("temp.tif") as crs:
        arr = crs.read()
    os.remove("temp.tif")
    return arr


def geoplot(ax, data, cmap_name, title, vmin=None, vmax=None, cbar=False, x_ticks=True, y_ticks=True):
    sc = ax.imshow(data, origin='upper', extent=[-105-0.1, -90+0.1, 30-0.1, 45+0.1], transform=ccrs.PlateCarree(), cmap=cmap_name, vmin=vmin, vmax=vmax)
    if cbar is True:
        plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, linestyle="--") 
    gl.xlocator = mticker.FixedLocator([-105.0, -100.0, -95.0, -90.0]) 
    gl.ylocator = mticker.FixedLocator([30.0, 35.0, 40.0, 45.0]) 
    gl.xlabels_top = False
    gl.ylabels_right = False
    if y_ticks is False:
        gl.ylabels_left = False
    if x_ticks is False:
        gl.xlabels_bottom = False

