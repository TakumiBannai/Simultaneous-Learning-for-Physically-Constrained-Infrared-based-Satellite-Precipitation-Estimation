# %%
import os
import traceback
import datetime
import rasterio
import rasterio.mask
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.util
from tqdm.notebook import tqdm

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

def convert_patch(image, window_shape = (64, 64)):
    patch = skimage.util.view_as_blocks(image, window_shape).copy()
    patch = patch.reshape(-1, window_shape[0], window_shape[1]) # Flatten
    return patch


def reconst_patch(patch, original_image_shape = (448, 448), window_shape = (64, 64)):
    num_patch_h = int(original_image_shape[0] / window_shape[0])
    num_patch_w = int(original_image_shape[1] / window_shape[1])
    patch = patch.reshape(num_patch_h, num_patch_w, window_shape[0], window_shape[1])
    assert patch.ndim == 4, "Invalid n_dim"
    reconst = []
    for i in range(patch.shape[0]):
        reconst.append(np.hstack(patch[i, :, :, :]))
    reconst = np.vstack(reconst)
    return reconst


def check_validtime(target_times):
    valid_times = []
    for target_time in tqdm(target_times):
        try:
            time_t, time_t_minus1, time_t_minus1_15, time_t_minus1_45 = create_timestamp(target_time.strftime("%Y-%m-%d %H"))  
            # Path to TIF
            path_stage4 = f"../stage4_net/merged_tif/{time_t[:6]}/{time_t}.tif"
            path_goes_ch3_15 = f"../goes/TIF_CH3/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
            path_goes_ch3_45 = f"../goes/TIF_CH3/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
            path_goes_ch4_15 = f"../goes/TIF_CH4/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
            path_goes_ch4_45 = f"../goes/TIF_CH4/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
            path_era5_cw_minus1 = f"../era5/TIF_CW/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
            path_era5_ci_minus1 = f"../era5/TIF_CI/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
            path_era5_cw = f"../era5/TIF_CW/{time_t[:6]}/{time_t}00.tif"
            path_era5_ci = f"../era5/TIF_CI/{time_t[:6]}/{time_t}00.tif"
            path_era5_prcp = f"../era5/TIF_PRCP/{time_t[:6]}/{time_t}00.tif"

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
            # traceback.print_exc()
            pass
            
    return valid_times


def create_feature(time_t = "2013-08-16 09",path_size=[64, 64]):
    time_t, time_t_minus1, time_t_minus1_15, time_t_minus1_45 = create_timestamp(time_t)
    # Path to TIF
    path_stage4 = f"../stage4_net/merged_tif/{time_t[:6]}/{time_t}.tif"
    path_goes_ch3_15 = f"../goes/TIF_CH3/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
    path_goes_ch3_45 = f"../goes/TIF_CH3/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
    path_goes_ch4_15 = f"../goes/TIF_CH4/{time_t_minus1_15[:6]}/{time_t_minus1_15}.tif"
    path_goes_ch4_45 = f"../goes/TIF_CH4/{time_t_minus1_45[:6]}/{time_t_minus1_45}.tif"
    path_era5_cw_minus1 = f"../era5/TIF_CW/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
    path_era5_ci_minus1 = f"../era5/TIF_CI/{time_t_minus1[:6]}/{time_t_minus1}00.tif"
    path_era5_cw = f"../era5/TIF_CW/{time_t[:6]}/{time_t}00.tif"
    path_era5_ci = f"../era5/TIF_CI/{time_t[:6]}/{time_t}00.tif"
    path_era5_prcp = f"../era5/TIF_PRCP/{time_t[:6]}/{time_t}00.tif"
    path_aoi = f"../AOI/AOI_CONUS_Center.geojson"
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

    # Çreate patched feature
    feature_ = []
    for i in range(feature.shape[0]):
        image = feature[i]
        patch = convert_patch(image, (path_size[0], path_size[1]))
        feature_.append(patch)

    return np.array(feature_)


def create_features(valid_times, path_size=[64, 64]):
    features = []
    for time_data in valid_times:
        features.append(create_feature(time_data.strftime("%Y-%m-%d %H"), path_size))
    features = np.array(features)
    features = features.transpose(1, 0, 2, 3, 4) # Channel, n_data, height, width
    features = features.reshape(9, -1, path_size[0], path_size[1])
    return features


def save_feature(time_data, feature, rain_th):
    save_fname = time_data.strftime("%Y%m") + "/" + time_data.strftime("%Y%m%d%H")
    for i in range(feature.shape[1]):
        # with_rain
        stage4 = feature[0, i, :, :].reshape(-1)
        if any(stage4 >= rain_th):
            os.makedirs(save_fname, exist_ok=True)
            np.save(f"{save_fname}/{i+1}.npy", feature[:, i, :, :])


path_aoi = f"../AOI/AOI_CONUS_Center.geojson"
aoi = gpd.read_file(path_aoi)

# %%
target_times = pd.date_range("2012-06-01", "2012-09-01", freq="1h").to_list()
valid_times = check_validtime(target_times)

# Save files by patch
for time_data in tqdm(valid_times):
    feature = create_feature(time_data.strftime("%Y-%m-%d %H"), path_size=[64, 64])
    save_feature(time_data, feature, rain_th = 0)
print("2012. Done")

target_times = pd.date_range("2013-06-01", "2013-09-01", freq="1h").to_list()
valid_times = check_validtime(target_times)

# Save files by patch
for time_data in tqdm(valid_times):
    feature = create_feature(time_data.strftime("%Y-%m-%d %H"), path_size=[64, 64])
    save_feature(time_data, feature, rain_th = 0)
print("2013. Done")


