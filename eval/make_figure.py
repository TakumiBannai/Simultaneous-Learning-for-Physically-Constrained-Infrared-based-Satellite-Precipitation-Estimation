
"""
Make figures for the manuscript.
"""
# %%
import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
sys.path.append("../model/")
from util import *
from eval import *

# %%
# Load data as 1-d array
timelist = make_datelist(start="2013-08-01 00:00", end="2013-08-31 00:00", freq="1h")
val_datelist, invalid_datelist = validate_datelist(timelist)
print(len(timelist), len(val_datelist), len(invalid_datelist))

label = load_as_1daray(val_datelist, "stage4")
pred_cnn = load_as_1daray(val_datelist, "pred_cnn")
pred_mtl = load_as_1daray(val_datelist, "pred_mtl")
persian_ccs = load_as_1daray(val_datelist, "persian_ccs")
gsmap = load_as_1daray(val_datelist, "gsmap")
era5 = load_as_1daray(val_datelist, "era5")*1000

# %%
# Figure 2 ---------------------------------------------------
# Density scatter plot, histogram and residual boxplot
# drawing functions
def density_scatter(fig, ax, label, pred, title_name, vmax=1900,
                    c_bar=False, y_label=True, x_label=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # log10
    with np.errstate(divide='ignore'):
        label = np.log10(label)
        pred = np.log10(pred)
    # plot
    h, x, y, im = ax.hist2d(label, pred, bins=(55, 55),
               vmax=vmax,
               cmap=cm.jet, range=[[0.01, 1.5], [0.01, 1.5]])
    if c_bar is True:
        fig.colorbar(im, ax=ax)
    if x_label is True:
        ax.set_xlabel('Label: Precipitation ($log_{10}$[mm/h])')
    if y_label is True:
        ax.set_ylabel('Pred: Precipitation ($log_{10}$[mm/h])')
    ax.set_title('{a}'.format(a=title_name))
    ax.set_xticks(np.arange(0, 1.75, 0.25))
    ax.set_yticks(np.arange(0, 1.75, 0.25))
    # daiagonal line
    ident = [0.01, 1.5]
    ax.plot(ident, ident, ls="--", lw="1.2", c="gray")


def histogram(fig, ax, label, pred_cnn, pred_mtl, persian_ccs, era5):
    from matplotlib.lines import Line2D
    products = [label, pred_cnn, pred_mtl, persian_ccs, era5]
    pnames = ["Observation", "Baseline", "Proposed", "PERSIANN-CCS", "ERA5"]
    colors =["black", "steelblue", "seagreen", "goldenrod", "slateblue"]

    for product, pname, c in zip(products, pnames, colors):
        product = ma.masked_where(product <= 0.1, product).compressed()
        # Hist
        hist = ax.hist(np.log10(product), bins=np.arange(0.1, 2.0, 0.01),
                    label=f"{pname}", alpha=.95,
                    log=True, histtype="step",
                    lw=1.7, 
                    cumulative=False,
                    density=False,
                    color = c)
    # legend: 四角から線に変更
    # Create new legend handles but use only colors
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(fontsize=13, bbox_to_anchor=(0.5, 0.33),
              handles=new_handles, labels=labels)

    ax.set_title("Histogram")
    ax.set_xlabel("Precipitation ($log_{10}$[mm/h])")
    ax.set_ylabel("Number of samples")


def residual_box(fig, ax, pred_cnn, pred_mtl, persian_ccs, era5):
    products = [pred_cnn, pred_mtl, persian_ccs, era5]
    pnames = ["Baseline", "Proposed", "PERSIANN-CCS", "ERA5"]
    resid = []
    for product, pname in zip(products, pnames):
        resid.append(compute_resid_with_mask(label[label >= 0.1], product[label >= 0.1]))

    bp = ax.boxplot(resid, whis=(20, 80), showfliers=False, patch_artist=True, boxprops=dict(alpha=.85))
    ax.set_xticklabels(pnames)
    ax.set_title('Residual Boxplot')
    ax.grid()
    ax.set_ylabel("Residual [mm/h]")

    # boxの色の設定
    colors = ["steelblue", "seagreen", "goldenrod", "slateblue"]
    for b, c in zip(bp['boxes'], colors):
        b.set(color='black', linewidth=1)  # boxの外枠の色
        b.set_facecolor(c) # boxの色

# Figure setting
plt.rcParams["font.size"] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

fig, ax = plt.subplots(figsize=(21, 14), nrows=2, ncols=3, dpi=300)
density_scatter(fig, ax[0, 0], label, pred_cnn, "Baseline", x_label=False)
density_scatter(fig, ax[0, 1], label, pred_mtl, "Proposed", x_label=False, y_label=False, c_bar=False)
density_scatter(fig, ax[1, 0], label, persian_ccs, "PERSIANN-CCS")
density_scatter(fig, ax[1, 1], label, era5, "ERA5", y_label=False)
histogram(fig, ax[0, 2], label, pred_cnn, pred_mtl, persian_ccs, era5)
residual_box(fig, ax[1, 2], pred_cnn, pred_mtl, persian_ccs, era5)


# %%
# Figure 3 ---------------------------------------------------
# The spatio-temporal variation of precipitation 
# during the extreme event (August 3rd to 10th, 2013)
# Spatial pattern
# Preprocessing for Output
timelist = make_datelist(start="2013-08-03 00:00",
                         end="2013-08-10 23:30",
                         freq="1h")
product_mean = []
for img_index in range(0, 9):
    image_list = []
    for date in timelist:
        try:
            image = read_image(date)
            image_list.append(np.array(image[img_index]))
        except BaseException:
            nan_mat = np.empty((448, 448)) # 448by448 pixels
            nan_mat[:] = np.nan
            image_list.append(nan_mat)
    image_list = np.array(image_list)
    product_mean.append(np.nanmean(image_list, axis=0))

# Preprocessing for Input
target_times = pd.date_range(start="2013-08-03 00:00",
                         end="2013-08-10 23:30",
                         freq="1h").to_list()
valid_times = check_validtime(target_times)
feature = []
for i in tqdm(range(len(valid_times))):
    _ = create_image(valid_times[i].strftime("%Y-%m-%d %H"))
    feature.append(_)
feature = np.array(feature)

feature_mask = ma.masked_where(feature > 1.0e+5, feature)
mean_feature = np.nanmean(feature_mask, axis=0)

label, pred_cnn, pred_mtl = product_mean[0], product_mean[1], product_mean[3]
persiann_ccs, era5 = product_mean[4], product_mean[8]

# Preprocessinf for Temporal-pattern (sub-domain)
timelist_fmt = []
for i in range(len(timelist)):
    timelist_fmt.append(timelist[i][:4] + "-" + timelist[i][4:6]\
                        + "-" + timelist[i][6:8] + " " + timelist[i][8:] + ":00")

timelist_fmt = pd.to_datetime(timelist_fmt)
domain_mean = []
for img_index in range(0, 9):
    image_list = []
    for date in timelist:
        try:
            image = read_image(date)
            image = np.array(image[img_index])
            image = np.mean(image[170:300, 150:440])  # Target domain
            image_list.append(image)
        except BaseException:
            image_list.append(np.nan)
    image_list = np.array(image_list)
    domain_mean.append(image_list)

domain_mean = np.array(domain_mean).T
domain_mean = pd.DataFrame(domain_mean, columns=["Lable", "Baseline", "Baseline(CW)",
                                                 "Proposed", "PERSIANN-CCS",
                                                 "IMERG(IR)", "IMERG(Uncal)",
                                                 "GSMaP", "ERA5"])
domain_mean = domain_mean.set_index(timelist_fmt)

arr = domain_mean[["Lable", "Baseline", "Proposed", "PERSIANN-CCS", "ERA5"]].values

# %%
plt.rcParams["font.size"] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(37, 22), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
geoplot(ax[0, 0], pred_cnn, "Blues", "Baseline", vmin=0, vmax=1, x_ticks=False, y_ticks=True, cbar=False)
geoplot(ax[0, 1], pred_mtl, "Blues", "Proposed", vmin=0, vmax=1, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[0, 2], persiann_ccs, "Blues", "PERSIANN-CCS", vmin=0, vmax=1, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[0, 3], era5, "Blues", "ERA5", vmin=0, vmax=1, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[0, 4], mean_feature[0], "Blues", "Stage-IV", vmin=0, vmax=1, x_ticks=False, y_ticks=False, cbar=False)

geoplot(ax[1, 0], pred_cnn - label, "seismic", "", vmin=-2, vmax=2, x_ticks=False, y_ticks=True, cbar=False)
geoplot(ax[1, 1], pred_mtl - label, "seismic", "", vmin=-2, vmax=2, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[1, 2], persiann_ccs - label, "seismic", "", vmin=-2, vmax=2, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[1, 3], era5 - label, "seismic", "", vmin=-2, vmax=2, x_ticks=False, y_ticks=False, cbar=False)
geoplot(ax[1, 4], mean_feature[5], "Purples", "CW", vmin=None, vmax=None, x_ticks=False, y_ticks=False, cbar=False)

geoplot(ax[2, 0], mean_feature[2], "inferno", "GOES(6.7μm)", vmin=None, vmax=None, x_ticks=True, y_ticks=True, cbar=False)
geoplot(ax[2, 1], mean_feature[4], "inferno", "GOES(10.7μm)", vmin=None, vmax=None, x_ticks=True, y_ticks=True, cbar=False)

for a in ax[2, 2:]:
    a.remove()

ax_add = fig.add_subplot(3, 5, (13, 15))
ax_add.plot(timelist_fmt, arr[:, 0], label="Observation", lw=1.3, ls="-", alpha=.95, marker=".", markersize=5, c="black")
ax_add.plot(timelist_fmt, arr[:, 1], label="Baseline", lw=1.3, ls="-", alpha=.95, marker=".", markersize=5, c="steelblue")
ax_add.plot(timelist_fmt, arr[:, 2], label="Proposed", lw=1.3, ls="-", alpha=.95, marker=".", markersize=5, c="seagreen")
ax_add.plot(timelist_fmt, arr[:, 3], label="PERSIANN-CCS", lw=1.3, ls="-", alpha=.95, marker=".", markersize=5, c="goldenrod")
ax_add.plot(timelist_fmt, arr[:, 4], label="ERA5", lw=1.3, ls="-", alpha=.95, marker=".", markersize=5, c="slateblue")

ax_add.grid(ls="--", c="black", alpha=.6, lw=0.5)
ax_add.legend(bbox_to_anchor=(0.99, 1), fontsize=24)
ax_add.set_title("Timse-series variation")
# ax_add.set_xlabel("Time: 2013/8/3 ~ 2013/8/10")
ax_add.set_ylabel("Precipitation (mm/h)")

days = mdates.DayLocator()
daysFmt = mdates.DateFormatter('%m/%d')
ax_add.xaxis.set_major_locator(days)
ax_add.xaxis.set_major_formatter(daysFmt)

fig.tight_layout()

# fig.savefig("fig3.png",transparent=True)

# %%
# Figure 4 (Sensitivity analysis) ----------------------
data = pd.read_csv("output/ensemble_result_overall_mean.csv")
data.set_index("Model", inplace=True)
data.index = ["Base", "CW", "MTL", "Seq", "Seq_WES", "Weight", "Weight+Seq", "Weight+Seq_WES"]
data = data.T

plt.rcParams["font.size"] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


# %%
# Regression performance
def format_tick(tick_val, tick_pos):
    return f"{tick_val:.3f}"

def reg_plot(data):
    data = data[['Base', 'CW', 'MTL', 'Weight', 'Weight+Seq']]
    mae = data.T["MAE"]
    rmse = data.T["RMSE"]
    cc = data.T["CC"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax2 = ax.twinx()
    ax3 = ax.twinx()

    ax.plot(mae, label="MAE", marker="o", color="darkblue", markersize=10)
    ax2.plot(rmse, label="RMSE", marker="^", color="royalblue", markersize=10)
    ax3.plot(cc, label="CC", marker="s", color="steelblue", markersize=10)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()

    ax.legend(h1+h2+h3, l1+l2+l3, frameon=False,
              bbox_to_anchor=(0.75, -0.1), fontsize=14, ncol=3)

    ax.set_title("Regression performance")
    ax.set_ylabel("MAE", rotation="horizontal", size=14, color="darkblue")
    ax2.set_ylabel("RMSE", rotation="horizontal", size=14, color="royalblue")
    ax3.set_ylabel("CC", rotation="horizontal", size=14, color="steelblue")

    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)

    ax2.yaxis.set_label_position('left') 
    ax2.yaxis.set_ticks_position('left')

    ax.spines["left"].set_position(("axes", -0.15))
    ax2.spines["left"].set_position(("axes", -0.0))
    ax3.spines["right"].set_position(("axes", 1))

    ax.spines["left"].set_color("darkblue")
    ax3.spines["left"].set_color("royalblue")
    ax3.spines["right"].set_color("steelblue")
    
    ax.yaxis.set_label_coords(-0.15, 1.01)
    ax2.yaxis.set_label_coords(-0.05, 1.01)
    ax3.yaxis.set_label_coords(1.07, 1.04)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.grid(linestyle="--", linewidth=0.9)
    
    # 桁数の統一
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax3.yaxis.set_major_formatter(FuncFormatter(format_tick))

    # 目盛りラベルの色設定
    ax.tick_params(axis='y', labelcolor='darkblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')


reg_plot(data)


# %%
# Classification performance
def format_tick(tick_val, tick_pos):
    return f"{tick_val:.2f}"

def clasf_plot(data):
    data = data[['Base', 'CW', 'MTL', 'Weight', 'Weight+Seq']]

    pod = data.T["POD"]
    far = data.T["FAR"]
    csi = data.T["CSI"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax2 = ax.twinx()
    ax3 = ax.twinx()

    ax.plot(pod, label="POD", marker="o", color="darkgreen", markersize=10)
    ax2.plot(far, label="FAR", marker="^", color="limegreen", markersize=10)
    ax3.plot(csi, label="CSI", marker="s", color="yellowgreen", markersize=10)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()

    ax.legend(h1+h3+h2, l1+l3+l2, frameon=False,
              bbox_to_anchor=(0.75, -0.1), fontsize=14, ncol=3)

    ax.set_title("Classification performance")
    ax.set_ylabel("POD", rotation="horizontal", size=14, color="darkgreen")
    ax2.set_ylabel("FAR", rotation="horizontal", size=14, color="limegreen")
    ax3.set_ylabel("CSI", rotation="horizontal", size=14, color="yellowgreen")

    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)

    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')

    ax.yaxis.set_label_coords(-0.14, 1.01)
    ax2.yaxis.set_label_coords(1.07, 1.05)
    ax3.yaxis.set_label_coords(-0.05, 1.01)

    ax.spines["left"].set_position(("axes", -0.14))
    ax2.spines["left"].set_position(("axes", -0.0))
    ax3.spines["right"].set_position(("axes", 1.))

    ax.spines["left"].set_color("darkgreen")
    ax3.spines["right"].set_color("limegreen")
    ax3.spines["left"].set_color("yellowgreen")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # 桁数の統一
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax3.yaxis.set_major_formatter(FuncFormatter(format_tick))

    # 目盛りラベルの色設定
    ax.tick_params(axis='y', labelcolor='darkgreen')
    ax2.tick_params(axis='y', labelcolor='limegreen')
    ax3.tick_params(axis='y', labelcolor='yellowgreen')

    ax.grid(linestyle="--", linewidth=0.8)

clasf_plot(data)
