# %% Setup

# Logically it seems like, if a bin has zero area, all deeper bins should also be zero. It looks like this is wrong in the fits: compare bins 7 and 8 at x > 0.8 in fits.20240506.croppast.pdf.
### Maybe not? Remember that Lin is only counting as "edge" that forest next to crop or pasture. One could imagine nonforest-nonagriculture taking up all of e.g. 500-1000m band. But seems unlikely!
# Is this ever true in the real sites?
# Is this avoidable in the fits?
# If not, how to handle in the model? Just set all deeper bins to zero? (Show this in plots.)


import pandas as pd
import numpy as np
import importlib
import lin_edgeareas_module as lem
import lin_edgeareas_figs as lef
import os

this_dir = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"
# version = 20240506
# version = 20240605
version = 20240709

# bin_edges_out = None
# bin_edges_out = [30, 60, 120, 300]
bin_edges_out = [30, 60, 90, 120, 300, 500, 1000, 2000]

# %% Setup

version_str = str(version)

# For making plots of predicted values across entire 0-1 range of X-axis
step_01 = 0.001
xdata_01 = np.arange(0, 1 + step_01, step_01)


# %% Import data
importlib.reload(lem)

# Get version info
vinfo = lem.get_version_info(version, bin_edges_out)

# Import edge areas and land covers
if version == 20240506:
    filename_template = os.path.join(this_dir, "inout", version_str, f"Edgearea_clean_%d.csv")
    edgeareas = lem.read_combine_multiple_csvs(filename_template, version)
    edgeareas = lem.add_missing_bins(edgeareas)
    landcovers = lem.import_landcovers_20240506(this_dir, version_str)
elif version == 20240605:
    filename = os.path.join(this_dir, "inout", version_str, "Edge_landcover_forSam.csv")
    site_info, siteyear_info, edgeareas, landcovers = lem.read_20240605(this_dir, filename, version)
elif version == 20240709:
    filename = os.path.join(this_dir, "inout", version_str, "Edge_landcover_forSam_v2.csv")
    site_info, siteyear_info, edgeareas, landcovers = lem.read_20240605(this_dir, filename, version)
else:
    raise RuntimeError(f"Version {version} not recognized")

if vinfo["Nsites"] is None:
    vinfo["Nsites"] = len(np.unique(edgeareas["site"]))

# There should be no NaNs
if any(edgeareas.isna().sum()):
    raise RuntimeError("NaN(s) found in edgeareas")
if any(landcovers.isna().sum()):
    raise RuntimeError("NaN(s) found in landcovers")

# Combine bins, if needed
if vinfo["bin_mapping"] is not None:
    edgeareas = lem.combine_bins(edgeareas, vinfo)


# %% Get derived information
importlib.reload(lem)

# Total forest area (from Lin's edgeareas files)
totalareas = edgeareas.drop(columns="edge")
totalareas = totalareas.groupby(["Year", "site"])
totalareas = totalareas.sum().rename(columns={"sumarea": "forest_from_ea"})

# Total derived areas (from landcovers)
for lc in [x.replace("is_", "") for x in landcovers.columns if "is_" in x]:
    totalareas = lem.get_site_lc_area(lc, totalareas, landcovers)

# Total area
site_area = landcovers.groupby(["Year", "site"]).sum()
totalareas = totalareas.assign(sitearea=site_area.sumarea)

# There should be no NaNs
if any(totalareas.isna().sum()):
    if any(totalareas["sitearea"].isna()):
        print("Sites/gridIDs missing landcovers outside 51-59:")
        yearsites_missing = totalareas[totalareas["sitearea"].isna()]
        years_missing = np.array([x[0] for x in yearsites_missing.index])
        sites_missing = np.array([x[1] for x in yearsites_missing.index])
        for y in np.unique(years_missing):
            sites_missing_thisyear = sites_missing[years_missing==y]
            print(f"   Year {y}: {sites_missing_thisyear}")
    raise RuntimeError("NaN(s) found in totalareas")

# %% Fit data
importlib.reload(lem)
importlib.reload(lef)

# X variable
# xvar = "forest_from_ea"
# xvar = "fforest"
# xvar = "croppast"
xvar = "croppast_frac_croppastfor"

# Y variable
yvar = "bin_as_frac_allforest"

# Exclude sites?
# sites_to_exclude = [4]
sites_to_exclude = []

# Bootstrap resample to ensure even sampling across X-axis?
bootstrap = False

edgefits = []
for b, bin in enumerate(pd.unique(edgeareas.edge)):
    ef = lem.EdgeFitType(edgeareas, totalareas, sites_to_exclude, b, bin, vinfo)
    ef.ef_fit(xvar, yvar, bootstrap)
    edgefits.append(ef)
    print(ef)

# Get figure filename suffix
figfile_suffix = ".".join([
    xvar,
    yvar,
])
if sites_to_exclude:
    figfile_suffix = figfile_suffix + "." + ",".join(
        [str(x)for x in sites_to_exclude]
    )
if bootstrap:
    figfile_suffix = ".".join(
        [figfile_suffix, "bs"]
    )
if vinfo["bin_mapping"] is not None:
    figfile_suffix = ".".join(
        [figfile_suffix, "-".join(
            [str(x) for x in vinfo["bin_edges_out"]]
        )]
    )

# Save summary figure
lef.plot_fits_1plot(this_dir, version_str, figfile_suffix, xdata_01, vinfo, edgeareas, xvar, yvar, edgefits)

print("Done.")


# %% Plot with subplots for each bin's scatter and fits
importlib.reload(lef)

# Setup
lef.plot_scatter_each_bin(this_dir, version_str, xdata_01, vinfo, edgeareas, xvar, yvar, sites_to_exclude, bootstrap, edgefits, figfile_suffix)


# %% Query a single point
importlib.reload(lem)

x = 0.5

ydata = lem.predict_multiple_fits(np.array([x]), edgeareas, edgefits, restrict_x=False)

print("Fraction of forest:")
for b, bin in enumerate(vinfo["bins_out"]):
    y = ydata[0][b]
    y = np.round(100*y, 1)
    print(f"{bin}:    \t{y}%")

print("Fraction of gridcell (assuming just forest and $xaxis):")
for b, bin in enumerate(vinfo["bins_out"]):
    y = ydata[0][b]
    y *= (1 - x)
    y = np.round(100*y, 1)
    print(f"{bin}:    \t{y}%")
    

gridcell_ht = 5
print("Patch height:")
for b, bin in enumerate(vinfo["bins_out"]):
    y = ydata[0][b]
    y *= (1 - x)*5
    y = np.round(y, 1)
    print(f"{bin}:    \t{y}")