# %% Setup

import pandas as pd
import numpy as np
import importlib
import lin_edgeareas_module as lem
from matplotlib import pyplot as plt
from matplotlib import colormaps

this_dir = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"
version = "20240506"


# %% Import data
importlib.reload(lem)

# Get version info
vinfo = lem.get_version_info(version)

# Import edge areas
filename_template = os.path.join(this_dir, "inout", version, f"Edgearea_clean_%d.csv")
edgeareas = lem.read_combine_multiple_csvs(filename_template, version)

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


# %% Visualize

# X variable
# xvar = "forest_from_ea"
xvar = "fforest"
# xvar = "croppast"
# xvar = "croppast_frac_croppastfor"

# Y variable
yvar = "bin_as_frac_allforest"

# Exclude sites?
sites_to_exclude = []


# Setup
importlib.reload(lem)
sitecolors = list(colormaps["Set2"].colors[0:vinfo["Nsites"]])
sites_to_include = [x for x in np.unique(edgeareas.site) if x not in sites_to_exclude]

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

edgefits = []

for b, bin in enumerate(pd.unique(edgeareas.edge)):
    
    # Get dataframe with just this edge, indexed by Year-site
    ef = lem.EdgeFitType(edgeareas, totalareas, sites_to_include, b, bin, vinfo)
    
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
    ef.ef_fit(xvar, yvar)
    plt.plot(ef.fit_xdata, ef.fit_result.best_fit, "-k")
    
    # Add chart info
    plt.legend(title="Site")
    title_bin = f"Bin {bin}: {vinfo['bins'][b]} m: "
    title_fit = f"{ef.fit_type}: r2={np.round(ef.fit_result.rsquared, 3)}"
    plt.title(title_bin + title_fit)
    plt.xlabel(lem.get_axis_labels(xvar))
    plt.ylabel(lem.get_axis_labels(yvar))
    
    # Save
    edgefits.append(ef)
    print(ef)

# Get rid of unused axes
for x in np.arange(nx):
    for y in np.arange(ny):
        if not axs[y][x].has_data():
            fig.delaxes(axs[y][x])

fig.tight_layout()

# Add lines with adjustments to sum to 1
tmp = totalareas[totalareas.index.isin(sites_to_include, level="site")]
tmp = tmp.div(tmp.sitearea, axis=0)
step = 0.001
xdata = np.arange(0, 1 + step, step)
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    ydata = edgefits[b].fit_result.eval(x=xdata)
    if b==0:
        ydata_yb = np.expand_dims(ydata, axis=1)
    else: 
        ydata_yb = np.concatenate((ydata_yb, np.expand_dims(ydata, axis=1)), axis=1)
ydata_yb[ydata_yb < 0] = 0
ydata_yb = ydata_yb / np.sum(ydata_yb, axis=1, keepdims=True)
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    fig.axes[b].plot(xdata, ydata_yb[:,b], '--k')

# Save
outfile = f"fits.{version}.{xvar}"
if sites_to_exclude:
    outfile += ".excl"
    for s, site in enumerate(sites_to_exclude):
        if s > 0:
            outfile += ","
        outfile += str(site)
outfile += ".pdf"
outpath = os.path.join(
    this_dir,
    "inout",
    version,
    outfile
    )
plt.savefig(outpath)

plt.show()
