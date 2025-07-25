"""
Various functions for exploring Lin's edge areas.
Provides utilities for bin management, output filepaths, and version info.
"""

import os
import numpy as np
import pandas as pd
from lmfit import fit_report  # pylint: disable=unused-import
from edge_fit_type import EdgeFitType


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements

# For making plots of predicted values across entire 0-1 range of X-axis
STEP_01 = 0.001
XDATA_01 = np.arange(0, 1 + STEP_01, STEP_01)


def add_missing_bins(df: pd.DataFrame, which_df: str):
    """
    Add zeroes for missing bins in site-years with zero area.
    Args:
        df (pd.DataFrame): Input DataFrame.
        which_df (str): 'edgeareas' or 'landcovers'.
    Returns:
        pd.DataFrame: DataFrame with missing bins filled as zero.
    """

    if np.any(np.isnan(df)):
        raise RuntimeError("add_missing_bins() received a DataFrame with NaNs")

    if which_df == "edgeareas":
        third_index = "edge"
    elif which_df == "landcovers":
        third_index = "landcover_num"
    else:
        raise RuntimeError(f"Unrecognized which_df: {which_df}")

    nsites = len(np.unique(df["site"]))
    nyears = len(np.unique(df["Year"]))
    n3rd = len(np.unique(df[third_index]))

    expected_edges = np.tile(np.unique(df[third_index].values), nsites * nyears)

    expected_sites = np.unique(df["site"].values)
    expected_sites = np.repeat(expected_sites, n3rd)
    expected_sites = np.tile(expected_sites, nyears)

    expected_years = np.unique(df["Year"].values)
    expected_years = np.repeat(expected_years, n3rd * nsites)

    sort_vars = ["Year", "site", third_index]
    df2 = df.sort_values(by=sort_vars)
    df2 = df2.set_index(sort_vars)
    index_names = tuple(sort_vars)
    new_multiindex = pd.MultiIndex.from_arrays(
        [expected_years, expected_sites, expected_edges],
        names=index_names,
    )
    df2 = df2.reindex(new_multiindex).reset_index()
    df2 = df2.fillna(0)
    return df2


def get_site_lc_area(lc: str, totalareas: pd.DataFrame, landcovers: pd.DataFrame):
    """
    Add land cover area for a given type to totalareas DataFrame.
    Args:
        lc (str): Land cover type.
        totalareas (pd.DataFrame): DataFrame to update.
        landcovers (pd.DataFrame): DataFrame with land cover data.
    Returns:
        pd.DataFrame: Updated totalareas DataFrame.
    """
    lc_area = landcovers[landcovers["is_" + lc]].groupby(["Year", "site"]).sum()
    lc_area = lc_area.rename(columns={"sumarea": lc})
    totalareas = totalareas.join(lc_area[lc])
    totalareas = totalareas.fillna(value=0)
    return totalareas


def get_output_filepath(
    out_dir: str,
    version,
    ef: EdgeFitType,
    title: str,
    outfile_suffix: str,
    *,
    extension="pdf",
):
    """
    Build output file path for figures or parameter files.
    Args:
        out_dir (str): Output directory.
        version (str/int): Data version.
        ef: EdgeFitType object.
        title (str): File title prefix.
        outfile_suffix (str): Suffix for file.
        extension (str): File extension (default 'pdf').
    Returns:
        str: Full output file path.
    """
    outfile = f"{title}.{version}"
    if ef.sites_to_exclude:
        outfile += ".excl"
        for s, site in enumerate(ef.sites_to_exclude):
            if s > 0:
                outfile += ","
            outfile += str(site)
    if ef.finfo["bootstrap"]:
        outfile += ".bs"
    outfile += "." + outfile_suffix
    outfile += "." + extension
    outpath = os.path.join(out_dir, outfile)

    return outpath


def bin_edges_to_str(bin_edges: list):
    """
    Convert bin edges to string labels for bins.
    Args:
        bin_edges (list): List of bin edge values.
    Returns:
        tuple: (number of bins, list of bin labels)
    """
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


def get_version_info(version, bin_edges_out: list):
    """
    Get version info and bin edge mapping for a given data version.
    Args:
        version (int/str): Data version.
        bin_edges_out (list): Output bin edges.
    Returns:
        dict: Version info dictionary with bin mapping.
    """
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


def map_bins_in2out(bin_edges_out: list, vinfo: dict):
    """
    Map input bins to output bins based on bin edges.
    Args:
        bin_edges_out (list): Output bin edges.
        vinfo (dict): Version info dict.
    Returns:
        dict: Updated version info with bin mapping.
    """
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


def get_bin_lo_hi_from_str(bin_str: str):
    """
    Get lower and upper bounds from a bin string label.
    Args:
        bin_str (str): Bin label string.
    Returns:
        tuple: (lower bound, upper bound)
    """
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


def read_landcovers_legend(this_dir: str):
    """
    Read the landcovers legend CSV file for MapBiomas classification.
    Args:
        this_dir (str): Directory containing the legend file.
    Returns:
        pd.DataFrame: DataFrame of landcover legend.
    """
    landcovers_legend = pd.read_csv(
        os.path.join(this_dir, "MAPBIOMAS_Col6_Legenda_Cores.simple.csv")
    )
    return landcovers_legend


def import_landcovers_20240506(this_dir: str, version, bin_edges_out: list):
    """
    Import and label landcovers for version 20240506.
    Args:
        this_dir (str): Base directory.
        version (str/int): Data version.
        bin_edges_out (list): Output bin edges.
    Returns:
        pd.DataFrame: Labeled landcovers DataFrame.
    """
    # Import legend
    landcovers_legend = read_landcovers_legend(this_dir)

    # Import landcovers
    filename_template = os.path.join(
        this_dir, "inout", str(version), "Landcover_clean_%d.csv"
    )
    landcovers = read_combine_multiple_csvs(filename_template, version, bin_edges_out)
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})

    # Add labels
    landcovers = label_landcovers(landcovers_legend, landcovers)

    return landcovers


def label_landcovers(landcovers_legend: pd.DataFrame, landcovers: pd.DataFrame):
    """
    Add string labels and classification flags to landcovers DataFrame.
    Args:
        landcovers_legend (pd.DataFrame): Legend DataFrame.
        landcovers (pd.DataFrame): Landcovers DataFrame.
    Returns:
        pd.DataFrame: Landcovers DataFrame with labels and flags.
    """
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


def print_and_write(line: str, file: str, newline=True):
    """
    Print a line and write it to a file, adding newline if needed.
    Args:
        line (str): Line to print and write.
        file (str): File path.
        newline (bool): Whether to add a newline (default True).
    """
    print(line)
    if newline and line[-1] != "\n":
        line += "\n"
    with open(file, "a", encoding="utf-8") as f:
        f.write(line)


def read_combine_multiple_csvs(filename_template: str, version, bin_edges_out: list):
    """
    Read and combine multiple CSV files for edge areas or landcovers.
    Args:
        filename_template (str): Template for file names.
        version (str/int): Data version.
        bin_edges_out (list): Output bin edges.
    Returns:
        pd.DataFrame: Combined DataFrame.
    """
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


def read_20240605(this_dir: str, filename_csv: str, version):
    """
    Read edge and landcover data for 20240605 and related versions.
    Args:
        this_dir (str): Base directory.
        filename_csv (str): CSV file path.
        version (str/int): Data version.
    Returns:
        tuple: (site_info, siteyear_info, edgeareas, landcovers)
    """

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

    # Add any missing rows
    landcovers = add_missing_bins(landcovers, "landcovers")

    # Check landcovers
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    landcovers = label_landcovers(landcovers_legend, landcovers)
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    print(landcovers.head())
    print(landcovers.tail())

    return site_info, siteyear_info, edgeareas, landcovers


def combine_bins(edgeareas: pd.DataFrame, vinfo: dict):
    """
    Combine input bins into output bins using bin mapping.
    Args:
        edgeareas (pd.DataFrame): Edge areas DataFrame.
        vinfo (dict): Version info dict with bin mapping.
    Returns:
        pd.DataFrame: Edge areas with combined bins.
    """
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


def drop_siteyears_without_obs(df: pd.DataFrame, sitearea: pd.Series):
    """
    Drop site-years with no observations (sitearea == 0).
    Args:
        df (pd.DataFrame): DataFrame to filter.
        sitearea (pd.Series): Series of site areas.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = df.set_index(sitearea.index.names)
    df = df.join(sitearea)
    missing_obs = df["sitearea"] == 0
    n_zero = len(df["sitearea"][missing_obs].index.unique())
    print(f"Dropping {n_zero} site-years with no observations")
    df = df[~missing_obs]
    df = df.drop(columns="sitearea")
    df = df.reset_index()
    return df
