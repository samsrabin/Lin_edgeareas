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


# %% Visualize

# X variable
# xvar = "forest_from_ea"
# xvar = "fforest"
xvar = "croppast"

# Y variable
yvar = "bin_as_frac_allforest"

# Exclude sites?
sites_to_exclude = [4]


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
fits = []
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    
    # Get dataframe with just this edge, indexed by Year-site
    thisedge_df = edgeareas[edgeareas.edge==bin].drop(columns="edge").set_index(["Year", "site"], verify_integrity=True)
    thisedge_df = thisedge_df[thisedge_df.index.isin(sites_to_include, level="site")]
    thisedge_df = thisedge_df.rename(columns={"sumarea": "bin"})
    
    # Join with areas of different land cover types
    thisedge_df = thisedge_df.join(totalareas)
    
    # Convert to fractional area
    thisedge_df = thisedge_df.div(thisedge_df.sitearea, axis=0)
    
    # Get edge bin area as fraction of total forest
    thisedge_df = thisedge_df.assign(bin_as_frac_allforest=thisedge_df.bin / thisedge_df.forest_from_ea)
    
    # Visualize
    sitelist = [i[1] for i in thisedge_df.index]
    plt.sca(fig.axes[b])
    for s, site in enumerate(np.unique(sitelist)):
        if site in sites_to_exclude:
            continue
        thisedgesite_df = thisedge_df[thisedge_df.index.get_level_values("site") == site]
        thisedgesite_df.plot(
            ax=fig.axes[b],
            x=xvar,
            y=yvar,
            color=sitecolors[s],
            label = site,
            kind="scatter",
            )
    
    # Add best fit
    xdata = thisedge_df[xvar].values
    ydata = thisedge_df[yvar].values
    isort = np.argsort(xdata)
    xdata = xdata[isort]
    ydata = ydata[isort]
    fit_type, fit = lem.fit(xdata, ydata)
    fits.append(fit)
    plt.plot(xdata, fit.best_fit, "-k")
    
    # Add chart info
    plt.legend(title="Site")
    title_bin = f"Bin {bin}: {vinfo['bins'][b]} m: "
    title_fit = f"{fit_type}: r2={np.round(fit.rsquared, 3)}"
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
tmp = totalareas[totalareas.index.isin(sites_to_include, level="site")]
tmp = tmp.div(tmp.sitearea, axis=0)
step = 0.001
xdata = np.arange(np.min(tmp[xvar]), np.max(tmp[xvar]) + step, step)
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    fit = fits[b]
    ydata = fit.eval(x=xdata)
    if b==0:
        ydata_yb = np.expand_dims(ydata, axis=1)
    else:
        ydata_yb = np.concatenate((ydata_yb, np.expand_dims(ydata, axis=1)), axis=1)
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