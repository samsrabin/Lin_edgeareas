"""
Functions for making figures for Lin's edge areas analysis.
Includes plotting and labeling utilities for edge fits and scatter plots.
"""

import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import lin_edgeareas_module as lem
from lin_edgeareas_module import XDATA_01
from edge_fit_list_type import EdgeFitListType

IS_INTERACTIVE = not hasattr(main, "__file__")

# https://stackoverflow.com/questions/58788958/the-smallest-valid-alpha-value-in-matplotlib
MIN_ALPHA = 1 / 510

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches


def get_axis_labels(var: str):
    """
    Get axis label for a variable name.
    Args:
        var (str): Variable name.
    Returns:
        str: Axis label.
    """
    if var == "forest_from_ea":
        axis = "Forested fraction (sum all edge bins)"
    elif var == "bin_as_frac_allforest":
        axis = "Fraction of forest in this bin"
    elif var == "fforest":
        axis = "Forest-forest fraction"
    elif var == "croppast":
        axis = "Crop + pasture fraction"
    elif var == "croppast_frac_croppastfor":
        axis = "Crop + pasture area as fraction of crop+pasture+fforest"
    else:
        axis = var
    return axis


def get_color(vinfo: dict, b: int):
    """
    Get color for a given edge bin index.
    Args:
        vinfo (dict): Version info dict.
        b (int): Bin index.
    Returns:
        color: Color for plotting.
    """
    color = colormaps["jet_r"](b / (vinfo["Nbins_out"] - 1))
    return color


def get_outfile_suffix(finfo: dict, vinfo: dict):
    """
    Get filename suffix for the current figure. Includes information about the input data version
    and the fit options.

    Args:
        finfo (dict): Fit info dict.
        vinfo (dict): Version info dict.

    Returns:
        str: Suffix for output file.
    """
    figfile_suffix = ".".join(
        [
            finfo["xvar"],
            finfo["yvar"],
        ]
    )
    if finfo["sites_to_exclude"]:
        figfile_suffix = (
            figfile_suffix + "." + ",".join([str(x) for x in finfo["sites_to_exclude"]])
        )
    if finfo["bootstrap"]:
        figfile_suffix = ".".join([figfile_suffix, "bs"])
    if vinfo["bin_mapping"] is not None:
        figfile_suffix = ".".join(
            [figfile_suffix, "-".join([str(x) for x in vinfo["bin_edges_out"]])]
        )

    return figfile_suffix


def plot_fits_1plot(
    out_dir: str,
    version_str: [str, int],
    outfile_suffix: str,
    vinfo: dict,
    edgefits: EdgeFitListType,
):
    """
    Save summary figure of raw and adjusted predictions for all bins.
    Args:
        out_dir (str): Output directory.
        version_str (str): Data version string.
        outfile_suffix (str): Output file suffix.
        vinfo (dict): Version info dict.
        edgefits: EdgeFitListType object.
    """
    ydata_yb, ydata_adj_yb = edgefits.get_all_fits_and_adjs(restrict_x=False)
    plt.figure()

    for b in np.arange(len(vinfo["bins_out"])):
        color = get_color(vinfo, b)
        plt.plot(XDATA_01, ydata_yb[:, b], color=color)
    for b in np.arange(len(vinfo["bins_out"])):
        color = get_color(vinfo, b)
        plt.plot(XDATA_01, ydata_adj_yb[:, b], "--", color=color)

    # Get legend
    legend = []
    for i, ef in enumerate(edgefits):
        item = vinfo["bins_out"][i]
        item += f" (r2={np.round(ef.fit_result.rsquared, 3)})"
        legend.append(item)

    # Add info
    plt.legend(legend)
    plt.xlabel(get_axis_labels(edgefits.finfo["xvar"]))
    plt.ylabel(get_axis_labels(edgefits.finfo["yvar"]))
    plt.title("Raw (solid) and adjusted (dashed) predictions")

    outpath = lem.get_output_filepath(
        out_dir, version_str, edgefits[0], "fit_lines_1plot", outfile_suffix
    )
    plt.savefig(outpath)

    if IS_INTERACTIVE:
        plt.show()


def plot_scatter_each_bin(
    *,
    out_dir: str,
    version_str: str,
    vinfo: dict,
    edgeareas: pd.DataFrame,
    sites_to_exclude: list,
    edgefits: EdgeFitListType,
    figfile_suffix: str,
):
    """
    Save plot with subplots for each bin's scatter and fits.
    Args:
        out_dir (str): Output directory.
        version_str (str): Data version string.
        vinfo (dict): Version info dict.
        edgeareas (pd.DataFrame): Edge areas DataFrame.
        sites_to_exclude (list): Sites to exclude.
        edgefits: EdgeFitListType object.
        figfile_suffix (str): Output file suffix.
    """
    sitecolors = list(colormaps["Set2"].colors[0 : vinfo["Nsites"]])
    sep_sites = vinfo["Nsites"] <= 5 and not edgefits.finfo["bootstrap"]

    # # Portrait
    # nx = 2; figsizex = 11
    # ny = int(np.ceil(vinfo["Nbins_out"]/2)); figsizey = 22

    # Landscape
    ny = 3
    figsizey = 11
    nx = int(np.ceil(vinfo["Nbins_out"] / ny))
    figsizex = 15

    fig, axs = plt.subplots(
        ny,
        nx,
        figsize=(figsizex, figsizey),
    )

    for b, this_bin in enumerate(pd.unique(edgeareas.edge)):
        # Get dataframe with just this edge, indexed by Year-site
        ef = edgefits[b]

        # Visualize
        plt.sca(fig.axes[b])
        alpha = min(1, 8.635 / vinfo["Nsites"])
        alpha = max(alpha, MIN_ALPHA)
        if sep_sites:
            sitelist = [i[1] for i in ef.thisedge_df.index]
            for s, site in enumerate(np.unique(sitelist)):
                if site in sites_to_exclude:
                    continue
                thisedgesite_df = ef.thisedge_df[
                    ef.thisedge_df.index.get_level_values("site") == site
                ]

                thisedgesite_df.plot(
                    ax=fig.axes[b],
                    x=ef.finfo["xvar"],
                    y=ef.finfo["yvar"],
                    color=sitecolors[s],
                    label=site,
                    kind="scatter",
                )
        elif edgefits.finfo["bootstrap"]:
            heatmap, xedges, yedges = np.histogram2d(ef.bs_xdata, ef.bs_ydata, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.cla()
            plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="magma_r")

        else:
            ef.thisedge_df.plot(
                ax=fig.axes[b],
                x=ef.finfo["xvar"],
                y=ef.finfo["yvar"],
                alpha=alpha,
                kind="scatter",
            )

        # Add best fit
        xdata, ydata = sort_xy_data(ef.fit_xdata, ef.predict(ef.fit_xdata))
        plt.plot(xdata, ydata, "-k")

        # Add chart info
        if sep_sites:
            plt.legend(title="Site")
        title_bin = f"Bin {this_bin}: {vinfo['bins_out'][b]} m: "
        title_fit = f"{ef.fit_type}: r2={np.round(ef.fit_result.rsquared, 3)}"
        plt.title(title_bin + title_fit)
        plt.xlabel(get_axis_labels(ef.finfo["xvar"]))
        plt.ylabel(get_axis_labels(ef.finfo["yvar"]))

    # Get rid of unused axes
    for x in np.arange(nx):
        for y in np.arange(ny):
            if not axs[y][x].has_data():
                fig.delaxes(axs[y][x])

    fig.tight_layout()

    # Add lines with adjustments to sum to 1
    _, ydata_adj_yb = edgefits.get_all_fits_and_adjs(restrict_x=False)
    for b, this_bin in enumerate(pd.unique(edgeareas.edge)):
        xdata, ydata = sort_xy_data(XDATA_01, ydata_adj_yb[:, b])
        fig.axes[b].plot(xdata, ydata, "--k")

    # Save
    outpath = lem.get_output_filepath(
        out_dir, version_str, edgefits[0], "fits_with_scatter", figfile_suffix
    )
    if not sep_sites:
        outpath = outpath.replace("pdf", "png")
    plt.savefig(outpath)

    if IS_INTERACTIVE:
        plt.show()


def sort_xy_data(xdata: np.ndarray, ydata: np.ndarray):
    """
    Sort X and Y data according to X data sort order.
    Args:
        xdata (NumPy array): X data array.
        ydata (NumPy array): Y data array.
    Returns:
        tuple: Sorted (xdata, ydata).
    """
    if xdata.shape != ydata.shape:
        raise RuntimeError(
            f"Shapes of xdata ({xdata.shape}) and ydata ({ydata.shape}) differ"
        )
    isort = np.argsort(xdata)
    xdata = xdata[isort]
    ydata = ydata[isort]
    return xdata, ydata
