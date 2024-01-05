# %%
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import rasterio
import rasterio.mask
from rasterio.transform import Affine
from fiona.crs import from_epsg
import geopandas as gpd

# %%
def read_data(y, m, d, t, ch):
    '''
    Usage;
    nc, data = read_data("2012", "06", "01", "0015", "ch3")
    '''
    input_path = f"raw/{y}/{m+d}/GridSat-CONUS.goes13.{y}.{m}.{d}.{t}.v01.nc"
    nc = netCDF4.Dataset(input_path)
    data = nc[ch][:] * nc[ch].scale_factor + nc[ch].add_offset
    return nc, data

# numpy to geoTIF conversion
def convert_np2tif(data, out_file,
                   with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                   crs):
    affine = Affine(with_of_pixel, 0, lon_upper_left, 0, -1*height_of_pixel, lat_upper_left)
    with rasterio.open(out_file, 'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=affine
                        ) as dst:
        dst.write(data.reshape(1,data.shape[0],data.shape[1]))


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

def get_geotif_params(lon, lat):
    lat_start, lat_end,lat_size = lat[0], lat[-1], lat.size
    lon_start, lon_end, lon_size = lon[0],lon[-1], lon.size
    # print("lat start from: ", lat_start, "end width: ", lat_end)
    # print("lon start from: ", lon_start, "end width: ", lon_end)
    # print("Shape-size for lat ", lat_size, "for lon ", lon_size)
    height_of_pixel = (-1*lat_start + lat_end)/lat_size
    with_of_pixel = (-1*lon_start + lon_end)/lon_size
    lon_upper_left = lon_start
    lat_upper_left = lat_end
    # print("Tif params -------")
    # print("height_of_pixel", height_of_pixel)
    # print("with_of_pixel", with_of_pixel)
    # print("lon_upper_left", lon_upper_left)
    # print("lat_upper_left", lat_upper_left)
    return height_of_pixel, with_of_pixel, lon_upper_left, lat_upper_left

def convertTIF_iteration(datetime_list, channel="ch4", save_dir="TIF_CH4"):
    for i in range(len(datetime_list)):
        try:
            nc, data = read_data(f"{datetime_list[i].year}",
                                 f"{datetime_list[i].month:02}",
                                 f"{datetime_list[i].day:02}",
                                 f"{datetime_list[i].hour:02}{datetime_list[i].minute:02}",
                                 channel)
            lon, lat = nc["lon"][:], nc["lat"][:]
            with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left = get_geotif_params(lon, lat)
            # array to tif (NOTE: input array dim and up-side-down)
            convert_np2tif(np.flipud(data[0]) ,f"./{save_dir}/{datetime_list[i].strftime('%Y%m%d%H%M')}.tif",
                           with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                           set_crs)
        except Exception as e:
            print(e)
            # print(datetime_list[i])

aoi = gpd.read_file('../AOI/AOI_CONUS_Center.geojson')
set_crs = from_epsg(4326)

# %%
datetime_list = pd.date_range("2013-06-01", "2013-09-01", freq="15min").to_list()
convertTIF_iteration(datetime_list, channel="ch4", save_dir="TIF_CH4")

# %%
# Orgaize Dir
import os
import glob

def organize_files(target_dir = "./TIF_CH4"):
    target_files = glob.glob(target_dir + "/*.tif")
    for i in range(len(target_files)):
        new_dir = target_dir + "/" + os.path.basename(target_files[i])[:6] # "YYYYMM" for New dir.
        new_file = os.path.join(new_dir, os.path.basename(target_files[i]))
        os.makedirs(new_dir, exist_ok=True)
        os.rename(target_files[i], new_file)

organize_files(target_dir = "./TIF_CH3")

# %%
# Crop Image
crop_tif("tif_con.tif", "tif_con_crop.tif", aoi)

# %%
import rasterio.plot
# Check original tif
with rasterio.open("tif_con.tif") as src:
    rasterio.plot.show(src)
    arr = src.read()

# Check cropped tif
with rasterio.open("tif_con_crop.tif") as src:
    rasterio.plot.show(src, cmap='jet')
    arr = src.read()

