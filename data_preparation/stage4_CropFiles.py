# %%
import os
import glob
import traceback
import datetime
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.util
from tqdm.notebook import tqdm

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

path_aoi = f"../AOI/AOI_CONUS_Center.geojson"
aoi = gpd.read_file(path_aoi)

# %%
# Crop and save files
path_files = glob.glob("merged_tif/201308/*.tif")
path_files.sort()

for path_file in path_files:
    arr = read_crop_as_array(path_file, aoi, h_pixel=448, w_pixel=448, resample=True)
    fname = path_file.split("/")[-1].replace("tif", "npy")
    os.makedirs("crop", exist_ok=True)
    np.save("crop/" + fname, arr.squeeze())
    print("Done..", fname)


