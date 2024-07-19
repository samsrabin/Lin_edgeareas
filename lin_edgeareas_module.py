"""
Various functions for exploring Lin's edge areas
"""

import os
import numpy as np
import pandas as pd
from lmfit import models
from lmfit import fit_report  # pylint: disable=unused-import

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

# For making plots of predicted values across entire 0-1 range of X-axis
STEP_01 = 0.001
XDATA_01 = np.arange(0, 1 + STEP_01, STEP_01)

def add_missing_bins(edgeareas):
    """
    Some site-years have bins missing because they had zero area. Add those zeroes.
    """
    edgeareas2 = pd.DataFrame()
    index_list = ["Year", "edge"]
    for site in edgeareas["site"].unique():
        df = edgeareas[edgeareas["site"] == site]
        for index in index_list:
            if index == "edge":
                # All site-years SHOULD have every edge, but don't necessarily
                cats = edgeareas["edge"].unique()
            else:
                # Only include years where this site has observations
                cats = df[index].unique()
            with pd.option_context("mode.chained_assignment", None):
                df[index] = pd.Categorical(df[index], categories=cats)
        df = df.groupby(index_list, as_index=False, observed=False).first()
        df["site"] = df["site"].fillna(site).astype(int)
        for index in index_list:
            df[index] = np.array(df[index])
        edgeareas2 = pd.concat([edgeareas2, df])
    edgeareas2["sumarea"] = edgeareas2["sumarea"].fillna(0)
    return edgeareas2





def get_site_lc_area(lc, totalareas, landcovers):
    lc_area = landcovers[landcovers["is_" + lc]].groupby(["Year", "site"]).sum()
    lc_area = lc_area.rename(columns={"sumarea": lc})
    totalareas = totalareas.join(lc_area[lc])
    totalareas = totalareas.fillna(value=0)
    return totalareas


def get_figure_filepath(this_dir, version, ef, title, figfile_suffix):
    outfile = f"{title}.{version}.{ef.fit_xvar}"
    if ef.sites_to_exclude:
        outfile += ".excl"
        for s, site in enumerate(ef.sites_to_exclude):
            if s > 0:
                outfile += ","
            outfile += str(site)
    if ef.fit_bootstrapped:
        outfile += ".bs"
    outfile += "." + figfile_suffix
    outfile += ".pdf"
    outpath = os.path.join(this_dir, "inout", version, outfile)

    return outpath


def bin_edges_to_str(bin_edges):
    # Check
    if any(x <= 0 for x in bin_edges) or any(np.isinf(bin_edges)):
        raise ValueError(
            "Include only positive, finite bin edges. 0 and Inf are implied."
        )

    n_bins = len(bin_edges) + 1
    bins = []
    for b, bin_edge in enumerate(bin_edges):
        if b == 0:
            bins.append(f"<{bin_edge}")
        else:
            bins.append(f"{bin_edges[b-1]}-{bin_edge}")
    bins.append(f">{bin_edge}")  # pylint: disable=undefined-loop-variable
    return n_bins, bins


def get_version_info(version, bin_edges_out):
    vinfo = {}

    # Specify bin edges in input file. Do not include 0 or Inf.
    if int(version) in [20240506, 20240605]:
        vinfo["bin_edges_in"] = [30, 60, 90, 120, 300, 500, 1000, 2000]
    elif int(version) in [20240709]:
        vinfo["bin_edges_in"] = [
            30,
            60,
            90,
            120,
            150,
            200,
            300,
            500,
            750,
            1000,
            1500,
            2000,
        ]
    else:
        raise RuntimeError(f"Version {version} not recognized")
    vinfo["bin_edges_in"].sort()
    if not np.array_equal(vinfo["bin_edges_in"], np.unique(vinfo["bin_edges_in"])):
        raise RuntimeError("bin_edges_in must all be unique")

    # Process input bins
    vinfo["Nbins_in"], vinfo["bins_in"] = bin_edges_to_str(vinfo["bin_edges_in"])

    if int(version) == 20240506:
        vinfo["Nsites"] = 4
    elif int(version) in [20240605, 20240709]:
        vinfo["Nsites"] = None
    else:
        raise RuntimeError(f"Version {version} not recognized")

    # Process output bins
    if bin_edges_out is None:
        vinfo["bin_mapping"] = None
        for key, value in vinfo.copy().items():
            if "_in" in key:
                vinfo[key.replace("_in", "_out")] = value
    else:
        bin_edges_out.sort()
        if not np.array_equal(bin_edges_out, np.unique(bin_edges_out)):
            raise RuntimeError("bin_edges_out must all be unique")
        if any(x not in vinfo["bin_edges_in"] for x in bin_edges_out):
            raise ValueError("bin_edges_out value(s) missing from bin_edges_in")
        vinfo["Nbins_out"], vinfo["bins_out"] = bin_edges_to_str(bin_edges_out)
        vinfo["bin_edges_out"] = bin_edges_out
        vinfo = map_bins_in2out(bin_edges_out, vinfo)

    return vinfo


def map_bins_in2out(bin_edges_out, vinfo):
    # The first input bin will always map to the first output bin
    vinfo["bin_mapping"] = [1]

    # Map the interior bins
    for bedge_in in vinfo["bin_edges_in"][1:]:
        index = None
        for b, bedge_out_lo in enumerate(bin_edges_out):
            if bedge_out_lo == max(bin_edges_out):
                bedge_out_hi = np.inf
            else:
                bedge_out_hi = bin_edges_out[b + 1]
            # if bedge_in >= bedge_out_lo and bedge_in <= bedge_out_hi:
            if bedge_out_lo <= bedge_in <= bedge_out_hi:
                index = b + 1
                break
        vinfo["bin_mapping"].append(index + 1)  # Because Lin started at 1

    # The highest input bin will always map to the highest output bin
    vinfo["bin_mapping"].append(vinfo["Nbins_out"])

    # Check mapping
    for b, bin_str_in in enumerate(vinfo["bins_in"]):
        m = vinfo["bin_mapping"][b] - 1
        bin_str_out = vinfo["bins_out"][m]

        bin_in_lo, bin_in_hi = get_bin_lo_hi_from_str(bin_str_in)
        bin_out_lo, bin_out_hi = get_bin_lo_hi_from_str(bin_str_out)

        err_msg = f" bound error with input bin {bin_str_in} mapped to output bin {bin_str_out}"
        assert bin_in_lo >= bin_out_lo, "Lower" + err_msg
        assert bin_in_hi <= bin_out_hi, "Upper" + err_msg

    return vinfo


def get_bin_lo_hi_from_str(bin_str):
    if "<" in bin_str:
        lo = -np.inf
        hi = float(bin_str.replace("<", ""))
    elif ">" in bin_str:
        lo = float(bin_str.replace(">", ""))
        hi = np.inf
    else:
        try:
            lo, hi = bin_str.split("-")
        except Exception as e:
            raise RuntimeError(f"Error splitting {bin_str}: Expected one hyphen") from e
        lo = float(lo)
        hi = float(hi)
    return lo, hi


def read_landcovers_legend(this_dir):
    landcovers_legend = pd.read_csv(
        os.path.join(this_dir, "MAPBIOMAS_Col6_Legenda_Cores.simple.csv")
    )
    return landcovers_legend


def import_landcovers_20240506(this_dir, version, bin_edges_out):

    # Import legend
    landcovers_legend = read_landcovers_legend(this_dir)

    # Import landcovers
    filename_template = os.path.join(
        this_dir, "inout", version, "Landcover_clean_%d.csv"
    )
    landcovers = read_combine_multiple_csvs(filename_template, version, bin_edges_out)
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})

    # Add labels
    landcovers = label_landcovers(landcovers_legend, landcovers)

    return landcovers


def label_landcovers(landcovers_legend, landcovers):
    landcovers = landcovers.assign(tmp=landcovers.landcover_num)
    landcovers = landcovers.set_index("tmp").join(
        landcovers_legend.set_index("landcover_num")
    )

    # Handle any types in landcovers but not landcovers_legend
    unknown_str = "unknown"
    landcovers["landcover_str"] = landcovers["landcover_str"].fillna(unknown_str)

    # There should be no remaining NaNs
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")

    # Regenerate index
    index = np.arange(len(landcovers.index))
    landcovers = landcovers.set_index(index)

    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")

    # Add legend-based info
    is_water = np.full(landcovers.landcover_num.shape, False)
    is_forest = is_water.copy()
    is_fforest = is_water.copy()
    is_crop = is_water.copy()
    is_pasture = is_water.copy()
    is_croppast = is_water.copy()
    is_agri = is_water.copy()
    is_unvegd = is_water.copy()
    is_vegd = is_water.copy()
    is_unobs = is_water.copy()
    is_unknown = is_water.copy()
    unknown_types = []
    for num in landcovers["landcover_num"].unique():
        # Get landcover string for this landcover code
        matching_landcovers = landcovers_legend[landcovers_legend.landcover_num == num]
        n_matches = matching_landcovers.shape[0]
        if n_matches == 0:
            unknown_types.append(num)
            landcover_str = unknown_str
        elif n_matches != 1:
            raise RuntimeError(
                f"Expected 1 landcover matching {num}; found {n_matches}"
            )
        else:
            landcover_str = matching_landcovers.landcover_str.values[0]

        # Classify based on landcover string
        where_this_landcover = np.where(landcovers["landcover_num"] == num)
        is_water[where_this_landcover] = "#5" in landcover_str
        is_forest[where_this_landcover] = "#1" in landcover_str
        is_fforest[where_this_landcover] = "#1.1" in landcover_str
        is_crop[where_this_landcover] = "#3.2" in landcover_str
        is_pasture[where_this_landcover] = "#3.1" in landcover_str
        is_croppast[where_this_landcover] = "3.4" in landcover_str
        is_agri[where_this_landcover] = "#3" in landcover_str
        is_unvegd[where_this_landcover] = "#4" in landcover_str
        is_unobs[where_this_landcover] = "#6" in landcover_str
        is_unknown[where_this_landcover] = n_matches == 0

    is_croppast = is_croppast | is_crop | is_pasture
    is_vegd = ~is_water & ~is_unvegd

    landcovers = landcovers.assign(
        is_water=is_water,
        is_forest=is_forest,
        is_fforest=is_fforest,
        is_crop=is_crop,
        is_pasture=is_pasture,
        is_croppast=is_croppast,
        is_agri=is_agri,
        is_unvegd=is_unvegd,
        is_vegd=is_vegd,
        is_unobs=is_unobs,
        is_unknown=is_unknown,
    )

    return landcovers





def read_combine_multiple_csvs(filename_template, version, bin_edges_out):
    vinfo = get_version_info(version, bin_edges_out)
    df_combined = []
    for f in 1 + np.arange(vinfo["Nsites"]):

        # Read site's .csv
        filename = filename_template % f
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        df = pd.read_csv(filename)

        # Add column for site
        df = df.assign(site=f)

        # Append to edgeareas df
        df_combined.append(df)

    df_combined = pd.concat(df_combined)
    return df_combined


def read_20240605(this_dir, filename_csv, version):

    first_edge_forest_lc = 51
    if version == 20240605:
        last_edge_forest_lc = 59
        first_edge_pasture_lc = None
        last_edge_pasture_lc = None
    elif version == 20240709:
        last_edge_forest_lc = 63
        first_edge_pasture_lc = 71
        last_edge_pasture_lc = 73
    else:
        raise RuntimeError(f"Version {version} not recognized")

    df = pd.read_csv(filename_csv)
    df = df.rename(
        columns={
            "totalareas": "sumarea",
            "gridID": "site",
            "year": "Year",
        }
    )

    # ecoregion should be unique per site
    site_info = df[["site", "ecoregion"]]
    groupvars = "site"
    if any(site_info.groupby(groupvars)["ecoregion"].nunique() > 1):
        print("ecoregion not unique per site. Skipping site_info.")
        site_info = None
    else:
        site_info = site_info.groupby(groupvars).mean()
        site_info = site_info.astype(int)
    df = df.drop(columns="ecoregion")

    # forestcover should be unique per site-year
    # 424 = 4.24%
    siteyear_info = df[["site", "Year", "forestcover"]]
    groupvars = ["site", "Year"]
    if any(siteyear_info.groupby(groupvars)["forestcover"].nunique() > 1):
        print("forestcover not unique per site-year. Skipping siteyear_info.")
        siteyear_info = None
    else:
        siteyear_info = siteyear_info.groupby(groupvars).mean()
        siteyear_info["forestcover"] *= 1e-4  # Convert to fraction
        siteyear_info = siteyear_info.rename(
            columns={"forestcover": "forestcover_frac"}
        )
    df = df.drop(columns="forestcover")

    # Get MapBiomas type of formação florestal
    landcovers_legend = read_landcovers_legend(this_dir)
    fforest_idx = np.where(
        ["formação florestal" in x.lower() for x in landcovers_legend["landcover_str"]]
    )[0]
    if len(fforest_idx) != 1:
        raise RuntimeError(
            f"Expected 1 formação florestal row in landcovers legend; found {len(fforest_idx)}"
        )
    fforest_idx = fforest_idx[0]
    fforest_num = landcovers_legend["landcover_num"][fforest_idx]

    # Which rows are edge classes?
    is_edge_class = (df["landcover"] >= first_edge_forest_lc) & (
        df["landcover"] <= last_edge_forest_lc
    )

    # Get DataFrame with just edge classes
    edgeareas = df[is_edge_class]
    edgeareas = edgeareas.rename(columns={"landcover": "edge"})
    edgeareas.edge -= (
        first_edge_forest_lc - 1
    )  # Change to edge bin numbers starting with 1

    # Get formação florestal area
    tmp_indices = ["site", "Year"]
    fforest = edgeareas.groupby(tmp_indices).sum()
    fforest = fforest.reset_index(level=tmp_indices)
    fforest = fforest.drop(columns="edge")
    fforest = fforest.assign(landcover=fforest_num)

    # Get landcovers
    landcovers = df[~is_edge_class]
    landcovers = pd.concat((landcovers, fforest))  # include formação florestal
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})

    # Combine edge pastures into non-edge pasture
    if first_edge_pasture_lc is not None:
        if last_edge_pasture_lc is None:
            raise ValueError(
                f"first_edge_pasture is {first_edge_pasture_lc} but last_edge_pasture is None"
            )

        # Get Mapbiomas pasture number
        mapbiomas_pasture_str = "#3.1. Pastagem"
        mapbiomas_pasture_lc = landcovers_legend["landcover_num"][
            landcovers_legend["landcover_str"] == mapbiomas_pasture_str
        ].values
        if len(mapbiomas_pasture_lc) != 1:
            raise RuntimeError(
                f"Expected 1 landcover_str matching {mapbiomas_pasture_str}; "
                + f"found {len(mapbiomas_pasture_lc)}"
            )
        mapbiomas_pasture_lc = mapbiomas_pasture_lc[0]

        # Combine
        lcnum = landcovers["landcover_num"]
        lcarea = landcovers["sumarea"]
        is_edge_pasture = (lcnum >= first_edge_pasture_lc) & (
            lcnum <= last_edge_pasture_lc
        )
        pasture_area_before = np.sum(lcarea[is_edge_pasture]) + np.sum(
            lcarea[lcnum == 15]
        )
        landcovers["landcover_num"][is_edge_pasture] = mapbiomas_pasture_lc
        tmp_indices = ["site", "Year", "landcover_num"]
        landcovers = landcovers.groupby(tmp_indices).sum()
        landcovers = landcovers.reset_index(level=tmp_indices)

        # Check
        pasture_area_after = np.sum(lcarea[lcnum == 15])
        if pasture_area_after != pasture_area_before:
            raise RuntimeError(
                "Pasture area mismatch after combining edge and deep pasture: "
                + str(pasture_area_after - pasture_area_before)
            )
        lcnum = landcovers["landcover_num"]
        is_edge_pasture = (lcnum >= first_edge_pasture_lc) & (
            lcnum <= last_edge_pasture_lc
        )
        if any(is_edge_pasture):
            raise RuntimeError("Edge pasture remains after combining")
    elif last_edge_pasture_lc is not None:
        raise ValueError(
            f"last_edge_pasture is {last_edge_pasture_lc} but first_edge_pasture is None"
        )

    # Check landcovers
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    landcovers = label_landcovers(landcovers_legend, landcovers)
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    print(landcovers.head())
    print(landcovers.tail())

    return site_info, siteyear_info, edgeareas, landcovers


def combine_bins(edgeareas, vinfo):
    edge2 = np.full_like(edgeareas["sumarea"].values, np.nan)
    for i, edge_out in enumerate(vinfo["bin_mapping"]):
        edge_in = i + 1
        edge2[np.where(edgeareas["edge"] == edge_in)] = edge_out
    if any(np.isnan(edge2)):
        raise RuntimeError("NaN(s) found in edge2")
    edgeareas2 = edgeareas.copy()
    edgeareas2["edge"] = edge2
    tmp_indices = ["edge", "site", "Year"]
    edgeareas2 = edgeareas2.groupby(tmp_indices).sum()
    edgeareas2 = edgeareas2.reset_index(level=tmp_indices)

    # Check
    tmp_indices = ["site", "Year"]
    edgeareas_tmp = edgeareas.groupby(tmp_indices).sum()
    edgeareas2_tmp = edgeareas2.groupby(tmp_indices).sum()
    assert np.array_equal(
        edgeareas_tmp["sumarea"], edgeareas2_tmp["sumarea"]
    ), "Error combining input to output bins"

    return edgeareas2
