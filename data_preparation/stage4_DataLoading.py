# %%
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

def create_url(datetime="201208"):
    urls = []
    for subdomain in subdomain_list:
        url_head = "https://hydrology.nws.noaa.gov/aorc-historic/"
        url = url_head + f"AORC_{subdomain}_4km/{subdomain}_precip_partition/AORC_APCP_4KM_{subdomain}_{datetime}.zip" 
        urls.append(url)
    return urls

subdomain_list = ["ABRFC", "CBRFC", "CNRFC", "LMRFC", "MARFC", "MBRFC",
                      "NCRFC", "NERFC", "NWRFC", "OHRFC", "SERFC", "WGRFC"]

# %%
# Data downloading
for datetime in ["201206", "201207", "201208", "201306", "201307", "201308"]:
    urls = create_url(datetime)
    for url, subdomain in zip(urls, subdomain_list):
        sh(f"wget {url}")
        sh(f"mkdir raw/{subdomain}_{datetime}")
        sh(f"unzip AORC_APCP_4KM_{subdomain}_{datetime}.zip -d raw/{subdomain}_{datetime}")
        sh(f"rm -r AORC_APCP_4KM_{subdomain}_{datetime}.zip")

# %%
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
from rasterio.transform import Affine
from fiona.crs import from_epsg
import geopandas as gpd
from rasterio.merge import merge
import rasterio.plot

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


aoi = gpd.read_file('../AOI/AOI_CONUS_Center.geojson')
set_crs = from_epsg(4326)

def create_conus(datetime, day, hour):
    # Create Tif files for subdomain
    for subdomain in subdomain_list:
        # Get data
        input_path = f"./raw/{subdomain}_{datetime}/AORC_APCP_{subdomain}_{datetime}{day}{hour}.nc4"
        nc = netCDF4.Dataset(input_path)
        data = nc['APCP_surface'][:]
        lat = nc['latitude'][:]
        lon = nc['longitude'][:]
        # Set GeiTiff metadata
        with_of_pixel = with_of_pixel = (lat.max() - lat.min())/data.shape[1]
        height_of_pixel = (-1*lon.min() - -1*lon.max())/data.shape[2]
        lon_upper_left = lon.min()
        lat_upper_left = lat.max()
        # array to tif (NOTE: input array dim and up-side-down)
        convert_np2tif(np.flipud(data[0]) ,f"{subdomain}.tif",
                       with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                       set_crs)

    # Merge subdomain Tifs
    src_list = []
    for subdomain in subdomain_list:
        src_t = rasterio.open(f"{subdomain}.tif")
        src_list.append(src_t)
    data, out_trans = merge(src_list)

    # Update the metadata
    out_meta = src_t.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": data.shape[1],
                     "width": data.shape[2],
                     "transform": out_trans
                     }
                    )

    # Save
    with rasterio.open(f"merged_tif/{datetime + day + hour}.tif", "w", **out_meta) as dest:
        dest.write(data)

# %%
# Run iteration
datetime_list = pd.date_range("2013-06-1", "2013-08-31", freq="H").to_list()

for i in range(len(datetime_list)):
    datetime = f"{datetime_list[i].year}{datetime_list[i].month:02}"
    day = f"{datetime_list[i].day:02}"
    hour = f"{datetime_list[i].hour:02}"
    create_conus(datetime, day, hour)

# %%
# Orgaize Dir
import os
import glob

def organize_files(target_dir = "./merged_tif"):
    target_files = glob.glob(target_dir + "/*.tif")
    for i in range(len(target_files)):
        new_dir = target_dir + "/" + os.path.basename(target_files[i])[:6] # "YYYYMM" for New dir.
        new_file = os.path.join(new_dir, os.path.basename(target_files[i]))
        os.makedirs(new_dir, exist_ok=True)
        os.rename(target_files[i], new_file)

organize_files(target_dir = "./merged_tif")

# %%
# Check original tif
with rasterio.open("merged_tif/2012060106.tif") as src:
    rasterio.plot.show(src, vmin=0, vmax=10)


# %%
# Crop Image
crop_tif("merged_tif/2012060101.tif", "Crop.tif", aoi)

# Check cropped tif
with rasterio.open("Crop.tif") as src:
    rasterio.plot.show(src, cmap='jet', vmin=0, vmax=5)
    arr = src.read()

# %%


# %%


# %%

# Check original tif
with rasterio.open("Merged.tif") as src:
    rasterio.plot.show(src)
    arr = src.read()

# Check cropped tif
with rasterio.open("Crop.tif") as src:
    rasterio.plot.show(src, cmap='jet', vmin=0, vmax=20)
    arr = src.read()

# %%


# %%



