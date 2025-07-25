"""
Play with fits to Lin's edge data
"""

# %% Setup

# Logically it seems like, if a bin has zero area, all deeper bins should also be zero.
# It looks like this is wrong in the fits: compare bins 7 and 8 at x > 0.8 in
# fits.20240506.croppast.pdf.
### Maybe not? Remember that Lin is only counting as "edge" that forest next to crop or pasture.
# One could imagine nonforest-nonagriculture taking up all of e.g. 500-1000m band.
# But seems unlikely!
# Is this ever true in the real sites?
# Is this avoidable in the fits?
# If not, how to handle in the model? Just set all deeper bins to zero? (Show this in plots.)


import os
import importlib
import numpy as np
import lin_edgeareas_module as lem
import lin_edgeareas_figs as lef
from edge_fit_list_type import EdgeFitListType

# %% Options
# pylint: disable=pointless-string-statement

"""
Edges of the forest bins we want in the output. Exclude 0 and Inf. Number of bins will be
len(bin_edges_out) + 1.
"""
# bin_edges_out = None
# bin_edges_out = [30, 60, 90, 120, 300, 500, 1000, 2000]
# bin_edges_out = [30, 60, 120, 300]
bin_edges_out = [30, 60, 120, 150, 300]

"""
Version of the input data to use. This must be a directory in THIS_DIR.
"""
# VERSION = 20240506
# VERSION = 20240605
VERSION = 20240709

"""
Directory containing VERSION subdirectories.
"""
THIS_DIR = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"

"""
X variable (predictor)
"""
xvar_list = [
    "forest_from_ea",  # Forested fraction (sum of all edge area bins; "sumarea" column in .csv)
    # "fforest",  # "Forest-forest" (landcover #1.1)
    # "croppast",  # Crop + pasture fraction (landcovers #3.1, 3.2, 3.4)
    # "croppast_frac_croppastfor",  # croppast as fraction of croppast+fforest
]

"""
Y variable (what we want to predict)
"""
YVAR = "bin_as_frac_allforest"  # Fraction of forest in this bin

"""
Exclude sites?
Site 4 has issues in at least some input data versions
"""
# sites_to_exclude = [4]
sites_to_exclude = []

"""
Bootstrap resample to ensure even sampling across X-axis?
"""
BOOTSTRAP = False


# %% Setup

out_dir = os.path.join(THIS_DIR, "inout", str(VERSION))


# %% Import data
importlib.reload(lem)

# Get version info
vinfo = lem.get_version_info(VERSION, bin_edges_out)

# Import edge areas and land covers
if VERSION == 20240506:
    filename_template = os.path.join(
        THIS_DIR, "inout", str(VERSION), "Edgearea_clean_%d.csv"
    )
    edgeareas = lem.read_combine_multiple_csvs(
        filename_template, VERSION, bin_edges_out
    )
    landcovers = lem.import_landcovers_20240506(THIS_DIR, str(VERSION), bin_edges_out)
elif VERSION == 20240605:
    filename = os.path.join(
        THIS_DIR, "inout", str(VERSION), "Edge_landcover_forSam.csv"
    )
    site_info, siteyear_info, edgeareas, landcovers = lem.read_20240605(
        THIS_DIR, filename, VERSION
    )
elif VERSION == 20240709:
    filename = os.path.join(
        THIS_DIR, "inout", str(VERSION), "Edge_landcover_forSam_v2.csv"
    )
    site_info, siteyear_info, edgeareas, landcovers = lem.read_20240605(
        THIS_DIR, filename, VERSION
    )
else:
    raise RuntimeError(f"Version {VERSION} not recognized")

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

# Add any missing rows
edgeareas = lem.add_missing_bins(edgeareas, "edgeareas")


# %% Get derived information
importlib.reload(lem)

# Total forest area (from Lin's edgeareas files)
totalareas = edgeareas.drop(columns="edge")
totalareas = totalareas.groupby(["Year", "site"])
totalareas = totalareas.sum().rename(columns={"sumarea": "forest_from_ea"})
if any(totalareas.isna().sum()):
    raise RuntimeError("NaN(s) found in totalareas")

# Total derived areas (from landcovers)
for lc in [x.replace("is_", "") for x in landcovers.columns if "is_" in x]:
    totalareas = lem.get_site_lc_area(lc, totalareas, landcovers)
    if any(totalareas.isna().sum()):
        raise RuntimeError("NaN(s) found in totalareas")

# Total area
site_area = landcovers.groupby(["Year", "site"]).sum()
totalareas = totalareas.assign(sitearea=site_area.sumarea)

# Drop site-years with no observations
edgeareas = lem.drop_siteyears_without_obs(edgeareas, totalareas["sitearea"])
landcovers = lem.drop_siteyears_without_obs(landcovers, totalareas["sitearea"])

# There should be no NaNs
if any(totalareas.isna().sum()):
    if any(totalareas["sitearea"].isna()):
        print("Sites/gridIDs missing landcovers outside 51-59:")
        yearsites_missing = totalareas[totalareas["sitearea"].isna()]
        years_missing = np.array([x[0] for x in yearsites_missing.index])
        sites_missing = np.array([x[1] for x in yearsites_missing.index])
        for y in np.unique(years_missing):
            sites_missing_thisyear = sites_missing[years_missing == y]
            print(f"   Year {y}: {sites_missing_thisyear}")
    raise RuntimeError("NaN(s) found in totalareas")


# %% Fit data
importlib.reload(lem)
importlib.reload(lef)

for xvar in xvar_list:
    print(f"===== xvar: {xvar} =====")

    # Dictionary with fit info
    finfo = {
        "xvar": xvar,
        "yvar": YVAR,
        "bootstrap": BOOTSTRAP,
        "sites_to_exclude": sites_to_exclude,
    }

    # Fit every edge bin
    edgefits = EdgeFitListType(
        edgeareas=edgeareas,
        totalareas=totalareas,
        vinfo=vinfo,
        finfo=finfo,
    )

    # Print information about the fits
    edgefits.print_fitted_equations()
    print("----------------------------------------------------")

    # Get output filename suffix (info about the fits)
    OUTFILE_SUFFIX = lef.get_outfile_suffix(finfo, vinfo)

    # Save .cdl file with fit parameters (and print to screen)
    cdl_file = lem.get_output_filepath(
        out_dir, VERSION, edgefits[0], "params", OUTFILE_SUFFIX, extension="cdl"
    )
    if os.path.exists(cdl_file):
        os.remove(cdl_file)
    edgefits.print_cdl_lines(cdl_file)

    # Save summary figure
    lef.plot_fits_1plot(out_dir, str(VERSION), OUTFILE_SUFFIX, vinfo, edgefits)

    # Save plot with subplots for each bin's scatter and fits
    lef.plot_scatter_each_bin(
        out_dir=out_dir,
        version_str=str(VERSION),
        vinfo=vinfo,
        edgeareas=edgeareas,
        sites_to_exclude=finfo["sites_to_exclude"],
        edgefits=edgefits,
        figfile_suffix=OUTFILE_SUFFIX,
    )


# %% Query a single point
# importlib.reload(lem)

# x = 0.5

# ydata = lem.predict_multiple_fits(np.array([x]), edgeareas, edgefits, restrict_x=False)

# print("Fraction of forest:")
# for b, bin in enumerate(vinfo["bins_out"]):
#     y = ydata[0][b]
#     y = np.round(100*y, 1)
#     print(f"{bin}:    \t{y}%")

# print("Fraction of gridcell (assuming just forest and $xaxis):")
# for b, bin in enumerate(vinfo["bins_out"]):
#     y = ydata[0][b]
#     y *= (1 - x)
#     y = np.round(100*y, 1)
#     print(f"{bin}:    \t{y}%")


# gridcell_ht = 5
# print("Patch height:")
# for b, bin in enumerate(vinfo["bins_out"]):
#     y = ydata[0][b]
#     y *= (1 - x)*5
#     y = np.round(y, 1)
#     print(f"{bin}:    \t{y}")
