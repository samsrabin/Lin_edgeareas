import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import lin_edgeareas_module as lem

IS_INTERACTIVE = not hasattr(main, "__file__")

# For making plots of predicted values across entire 0-1 range of X-axis
step_01 = 0.001
XDATA_01 = np.arange(0, 1 + step_01, step_01)


def get_axis_labels(var):
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


def get_color(vinfo, b):
    color = colormaps["jet_r"](b / (vinfo["Nbins_out"] - 1))
    return color


def get_figfile_suffix(vinfo, yvar, sites_to_exclude, bootstrap, xvar):
    figfile_suffix = ".".join(
        [
            xvar,
            yvar,
        ]
    )
    if sites_to_exclude:
        figfile_suffix = (
            figfile_suffix + "." + ",".join([str(x) for x in sites_to_exclude])
        )
    if bootstrap:
        figfile_suffix = ".".join([figfile_suffix, "bs"])
    if vinfo["bin_mapping"] is not None:
        figfile_suffix = ".".join(
            [figfile_suffix, "-".join([str(x) for x in vinfo["bin_edges_out"]])]
        )

    return figfile_suffix


def plot_fits_1plot(
    this_dir, version_str, figfile_suffix, vinfo, edgeareas, xvar, yvar, edgefits
):
    ydata_yb = lem.predict_multiple_fits(XDATA_01, edgeareas, edgefits, restrict_x=True)
    ydata_adj_yb = lem.adjust_predicted_fits(
        lem.predict_multiple_fits(XDATA_01, edgeareas, edgefits)
    )

    for b, bin in enumerate(vinfo["bins_out"]):
        color = get_color(vinfo, b)
        plt.plot(XDATA_01, ydata_yb[:, b], color=color)
    for b, bin in enumerate(vinfo["bins_out"]):
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
    plt.xlabel(get_axis_labels(xvar))
    plt.ylabel(get_axis_labels(yvar))
    plt.title("Raw (solid) and adjusted (dashed) predictions")

    outpath = lem.get_figure_filepath(
        this_dir, version_str, edgefits[0], "fit_lines_1plot", figfile_suffix
    )
    plt.savefig(outpath)

    if IS_INTERACTIVE:
        plt.show()


def plot_scatter_each_bin(
    this_dir,
    version_str,
    vinfo,
    edgeareas,
    xvar,
    yvar,
    sites_to_exclude,
    bootstrap,
    edgefits,
    figfile_suffix,
):
    sitecolors = list(colormaps["Set2"].colors[0 : vinfo["Nsites"]])
    sep_sites = vinfo["Nsites"] <= 5 and not bootstrap

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
    Nextra = ny * nx - vinfo["Nbins_out"]

    for b, bin in enumerate(pd.unique(edgeareas.edge)):
        # Get dataframe with just this edge, indexed by Year-site
        ef = edgefits[b]

        # Visualize
        plt.sca(fig.axes[b])
        alpha = min(1, 8.635 / vinfo["Nsites"])
        alpha = max(
            alpha, 1 / 510
        )  # https://stackoverflow.com/questions/58788958/the-smallest-valid-alpha-value-in-matplotlib
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
                    x=xvar,
                    y=yvar,
                    color=sitecolors[s],
                    label=site,
                    kind="scatter",
                )
        elif bootstrap:
            heatmap, xedges, yedges = np.histogram2d(ef.bs_xdata, ef.bs_ydata, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.cla()
            plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="magma_r")

        else:
            ef.thisedge_df.plot(
                ax=fig.axes[b],
                x=xvar,
                y=yvar,
                alpha=alpha,
                kind="scatter",
            )

        # Add best fit
        plt.plot(ef.fit_xdata, ef.predict(ef.fit_xdata), "-k")

        # Add chart info
        if sep_sites:
            plt.legend(title="Site")
        title_bin = f"Bin {bin}: {vinfo['bins_out'][b]} m: "
        title_fit = f"{ef.fit_type}: r2={np.round(ef.fit_result.rsquared, 3)}"
        plt.title(title_bin + title_fit)
        plt.xlabel(get_axis_labels(xvar))
        plt.ylabel(get_axis_labels(yvar))

    # Get rid of unused axes
    for x in np.arange(nx):
        for y in np.arange(ny):
            if not axs[y][x].has_data():
                fig.delaxes(axs[y][x])

    fig.tight_layout()

    # Add lines with adjustments to sum to 1
    ydata_adj_yb = lem.adjust_predicted_fits(
        lem.predict_multiple_fits(XDATA_01, edgeareas, edgefits)
    )
    for b, bin in enumerate(pd.unique(edgeareas.edge)):
        fig.axes[b].plot(XDATA_01, ydata_adj_yb[:, b], "--k")

    # Save
    outpath = lem.get_figure_filepath(
        this_dir, version_str, edgefits[0], "fits_with_scatter", figfile_suffix
    )
    if not sep_sites:
        outpath = outpath.replace("pdf", "png")
    plt.savefig(outpath)

    if IS_INTERACTIVE:
        plt.show()
