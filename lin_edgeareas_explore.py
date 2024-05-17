# %% Setup

import pandas as pd
import numpy as np
import importlib
import lin_edgeareas_module as lem
from matplotlib import pyplot as plt
from matplotlib import colormaps

this_dir = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"
version = "20240506"

# %% Setup

step_01 = 0.001
xdata_01 = np.arange(0, 1 + step_01, step_01)


# %% Import data
importlib.reload(lem)

# Get version info
vinfo = lem.get_version_info(version)

# Import edge areas
filename_template = os.path.join(this_dir, "inout", version, f"Edgearea_clean_%d.csv")
edgeareas = lem.read_combine_multiple_csvs(filename_template, version)
edgeareas = lem.add_missing_bins(edgeareas)

# Import land covers
landcovers = lem.import_landcovers(this_dir, version)


# %% Get derived information
importlib.reload(lem)

# Total forest area (from Lin's edgeareas files)
totalareas = edgeareas.groupby(["Year", "site"]).sum().drop(columns="edge").rename(columns={"sumarea": "forest_from_ea"})

# Total derived areas (from landcovers)
for lc in [x.replace("is_", "") for x in landcovers.columns if "is_" in x]:
    totalareas = lem.get_site_lc_area(lc, totalareas, landcovers)

# Total area
site_area = landcovers.groupby(["Year", "site"]).sum()
totalareas = totalareas.assign(sitearea=site_area.sumarea)

# (cropland_pasture)/(cropland+pasture+forest)
croppast_frac_croppastfor = totalareas.croppast / (totalareas.fforest + totalareas.croppast)
totalareas = totalareas.assign(croppast_frac_croppastfor=croppast_frac_croppastfor)


# %% Fit data
importlib.reload(lem)

# X variable
# xvar = "forest_from_ea"
# xvar = "fforest"
xvar = "croppast"
# xvar = "croppast_frac_croppastfor"

# Y variable
yvar = "bin_as_frac_allforest"

# Exclude sites?
sites_to_exclude = [4]

# Bootstrap resample to ensure even sampling across X-axis?
bootstrap = False

edgefits = []
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    ef = lem.EdgeFitType(edgeareas, totalareas, sites_to_exclude, b, bin, vinfo)
    ef.ef_fit(xvar, yvar, bootstrap)
    edgefits.append(ef)
    print(ef)
print("Done.")


# %% Plot with subplots for each bin's scatter and fits
importlib.reload(lem)

# Setup
sitecolors = list(colormaps["Set2"].colors[0:vinfo["Nsites"]])

# Portrait
nx = 2; figsizex = 11
ny = int(np.ceil(vinfo["Nbins"]/2)); figsizey = 22

# # Landscape
# ny = 2; figsizey = 11
# nx = int(np.ceil(vinfo["Nbins"]/2)); figsizex = 22

fig, axs = plt.subplots(
    ny, nx,
    figsize=(figsizex, figsizey),
    )
Nextra = ny*nx - vinfo["Nbins"]

for b, bin in enumerate(pd.unique(edgeareas.edge)):
    
    # Get dataframe with just this edge, indexed by Year-site
    ef = edgefits[b]
    
    # Visualize
    sitelist = [i[1] for i in ef.thisedge_df.index]
    plt.sca(fig.axes[b])
    for s, site in enumerate(np.unique(sitelist)):
        if site in sites_to_exclude:
            continue
        thisedgesite_df = ef.thisedge_df[ef.thisedge_df.index.get_level_values("site") == site]
        thisedgesite_df.plot(
            ax=fig.axes[b],
            x=xvar,
            y=yvar,
            color=sitecolors[s],
            label = site,
            kind="scatter",
            )
    
    # Add best fit
    plt.plot(ef.fit_xdata, ef.predict(ef.fit_xdata), "-k")
    
    # Add chart info
    plt.legend(title="Site")
    title_bin = f"Bin {bin}: {vinfo['bins'][b]} m: "
    title_fit = f"{ef.fit_type}: r2={np.round(ef.fit_result.rsquared, 3)}"
    plt.title(title_bin + title_fit)
    plt.xlabel(lem.get_axis_labels(xvar))
    plt.ylabel(lem.get_axis_labels(yvar))

# Get rid of unused axes
for x in np.arange(nx):
    for y in np.arange(ny):
        if not axs[y][x].has_data():
            fig.delaxes(axs[y][x])

fig.tight_layout()

# Add lines with adjustments to sum to 1
ydata_adj_yb = lem.adjust_predicted_fits(
    lem.predict_multiple_fits(xdata_01, edgeareas, edgefits)
    )
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    fig.axes[b].plot(xdata_01, ydata_adj_yb[:,b], '--k')

# Save
outpath = lem.get_figure_filepath(this_dir, version, edgefits[0], "fits_with_scatter")
plt.savefig(outpath)

plt.show()


# %% All fits and adjusted fits on one plot
importlib.reload(lem)

def get_color(vinfo, b):
    color = colormaps["jet_r"](b/(vinfo["Nbins"]-1))
    return color

ydata_yb = lem.predict_multiple_fits(xdata_01, edgeareas, edgefits, restrict_x=True)
ydata_adj_yb = lem.adjust_predicted_fits(
    lem.predict_multiple_fits(xdata_01, edgeareas, edgefits)
    )

for b, bin in enumerate(vinfo["bins"]):
    color = get_color(vinfo, b)
    plt.plot(xdata_01, ydata_yb[:,b], color=color)
for b, bin in enumerate(vinfo["bins"]):
    color = get_color(vinfo, b)
    plt.plot(xdata_01, ydata_adj_yb[:,b], "--", color=color)
plt.legend(vinfo["bins"])
plt.xlabel(lem.get_axis_labels(xvar))
plt.ylabel(lem.get_axis_labels(yvar))
plt.title("Raw (solid) and adjusted (dashed) predictions")

outpath = lem.get_figure_filepath(this_dir, version, edgefits[0], "fit_lines_1plot")
plt.savefig(outpath)

plt.show()
