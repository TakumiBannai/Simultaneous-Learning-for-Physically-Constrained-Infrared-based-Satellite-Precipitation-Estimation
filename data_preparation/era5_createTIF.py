# %%
import os
import shutil
import netCDF4
import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import matplotlib.animation as animation
import rasterio
import rasterio.mask
from rasterio.transform import Affine
from fiona.crs import from_epsg
import geopandas as gpd

# %%
def read_data(nc, time_step=0, scaled=False):
    val_mame = list(nc.variables.keys())[-1]
    # Get data as array
    arr = nc[val_mame][time_step]
    lat = nc["latitude"][:]
    lon = nc["longitude"][:]
    if scaled is True:
        # Scale and off-set
        scale = nc[val_mame].scale_factor
        offset = nc[val_mame].add_offset
        arr = arr * scale + offset
    label = get_time(nc["time"][time_step])
    return arr, lon, lat, label

# Time (UTC)
def get_time(target_time):
    """
    units: hours(UTC) since 1900-01-01 00:00:00.0
    """
    start_base = datetime.datetime(1900, 1, 1, 0, 0)
    add_hours = datetime.timedelta(hours=target_time.item())
    target_time = start_base + add_hours
    return target_time.strftime('%Y%m%d%H%M')

def longitude_conversion(arr, lon, lat):
    # Longitude-conversion (0~360 -> -180~180)
    lon_ = (lon + 180) % 360 - 180
    lon_ = np.hstack((lon_[720:], lon_[:720]))
    arr_ = np.hstack((arr[:, 720:], arr[:, :720]))
    return arr_, lon_, lat

def get_geotif_params(lon, lat):
    lat_start, lat_end,lat_size = lat[0], lat[-1], lat.size
    lon_start, lon_end, lon_size = lon[0],lon[-1], lon.size
    # print("lat start from: ", lat_start, "end width: ", lat_end)
    # print("lon start from: ", lon_start, "end width: ", lon_end)
    # print("Shape-size for lat ", lat_size, "for lon ", lon_size)
    height_of_pixel = (lat_start + -1*lat_end)/lat_size
    with_of_pixel = (-1*lon_start + lon_end)/lon_size
    lon_upper_left = lon_start
    lat_upper_left = lat_start
    # print("Tif params -------")
    # print("height_of_pixel", height_of_pixel)
    # print("with_of_pixel", with_of_pixel)
    # print("lon_upper_left", lon_upper_left)
    # print("lat_upper_left", lat_upper_left)
    return height_of_pixel, with_of_pixel, lon_upper_left, lat_upper_left

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



def convertTIF_iteration(input_path, save_dir, scaled=False):
    os.makedirs(save_dir, exist_ok=True)
    nc = netCDF4.Dataset(input_path)
    for time_step in range(len(nc['time'][:])):
        arr, lon, lat, title = read_data(nc, time_step, scaled)
        arr, lon, lat = longitude_conversion(arr, lon, lat)
        with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left = get_geotif_params(lon, lat)
        # array to tif (NOTE: input array dim-size and up-side-down)
        convert_np2tif(arr, f"./{save_dir}/{title}.tif",
                       with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                       set_crs)


aoi = gpd.read_file('../AOI/AOI_CONUS_Center.geojson')
set_crs = from_epsg(4326)

# %%
# Cloud water Run
input_path = "./raw/total_column_cloud_liquid_water_2012.nc"
convertTIF_iteration(input_path, "TIF_CW", False)
print("Done Cloud water")

# Cloud Ice Run
input_path = "./raw/total_column_cloud_ice_water_2012.nc"
convertTIF_iteration(input_path, "TIF_CI", False)
print("Done Cloud ice")

# Precp Run
input_path = "./raw/total_precipitation_2012.nc"
convertTIF_iteration(input_path, "TIF_PRCP", False)
print("Done Prcp")

# Water vapor Run
input_path = "./raw/total_column_water_vapour_2012.nc"
convertTIF_iteration(input_path, "TIF_WV", False)
print("Done TIF_WV")


