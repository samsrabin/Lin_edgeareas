import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd
from lmfit import models, fit_report
from matplotlib import colormaps

class EdgeFitType:
    def __init__(self, edgeareas, totalareas, sites_to_exclude, b, bin, vinfo):
        
        sites_to_include = [x for x in np.unique(edgeareas.site) if x not in sites_to_exclude]
        
        # Get dataframe with just this edge, indexed by Year-site
        self.thisedge_df = edgeareas[edgeareas.edge==bin].drop(columns="edge").set_index(["Year", "site"], verify_integrity=True)
        self.thisedge_df = self.thisedge_df[self.thisedge_df.index.isin(sites_to_include, level="site")]
        self.thisedge_df = self.thisedge_df.rename(columns={"sumarea": "bin"})
        
        # Join with areas of different land cover types
        self.thisedge_df = self.thisedge_df.join(totalareas)
        if any(self.thisedge_df.isna().sum()):
            raise RuntimeError("NaN(s) found after joining thisedge_df and totalareas")
        
        # Convert to fractional area
        self.thisedge_df = self.thisedge_df.div(self.thisedge_df.sitearea, axis=0)
        
        # Get edge bin area as fraction of total forest
        self.thisedge_df = self.thisedge_df.assign(bin_as_frac_allforest=self.thisedge_df.bin / self.thisedge_df.forest_from_ea)
        
        # Save other info
        self.bin_index = b
        self.bin_number = bin
        self.bin_name = vinfo["bins"][b]
        self.sites_to_exclude = sites_to_exclude

        # Initialize other members
        self.fit_xdata = None
        self.fit_ydata_in = None
        self.fit_type = None
        self.fit_result = None
        self.fit_bootstrapped = None
        self.fit_xvar = None
        self.fit_yvar = None
    
    def __str__(self):
        prefix = f"EdgeFitType for bin {self.bin_number} ({self.bin_name} m): "
        if self.fit_result is None:
            output = prefix + "Not yet fit"
        else:
            output = prefix + f"{self.fit_type} r-squared {np.round(self.fit_result.rsquared, 3)}"
        return output

    def ef_fit(self, xvar, yvar, bootstrap):
        self.fit_xvar = xvar
        self.fit_yvar = yvar
        self.fit_bootstrapped = bootstrap
        
        # Get X and Y data for fitting
        if xvar == "croppast_frac_croppastfor":
            self.thisedge_df = self.thisedge_df.assign(croppast_frac_croppastfor=self.thisedge_df.croppast / (self.thisedge_df.fforest + self.thisedge_df.croppast))
            self.fit_xdata = self.thisedge_df.croppast_frac_croppastfor.values
        else:
            self.fit_xdata = self.thisedge_df[xvar].values
        self.fit_ydata_in = self.thisedge_df[yvar].values
        
        # Bootstrap across bins of X-axis to ensure even weighting
        if bootstrap:
            N_xbins = 10
            x_max = max(self.fit_xdata)
            x_min = min(self.fit_xdata)
            step = (x_max - x_min) / (N_xbins)
            bin_boundaries = np.arange(x_min, x_max+step, step)
            xdata = np.array([])
            ydata = np.array([])
            for b in np.arange(N_xbins):
                lo = bin_boundaries[b]
                hi = bin_boundaries[b+1]
                if b == N_xbins - 1:
                    cond_hi = self.fit_xdata <= hi
                else:
                    cond_hi = self.fit_xdata < hi
                cond = cond_hi & (self.fit_xdata >= lo)
                if not any(cond):
                    continue
                bin_data = self.fit_xdata[np.where(cond)[0]]
                rng = np.random.default_rng(seed=1987)
                chosen = rng.choice(np.arange(np.sum(cond)), 100)
                xdata = np.concatenate((xdata, self.fit_xdata[chosen]))
                ydata = np.concatenate((ydata, self.fit_ydata_in[chosen]))
        else:
            xdata = self.fit_xdata
            ydata = self.fit_ydata_in
        
        # Sort X and Y data (helpful for plotting)
        isort = np.argsort(self.fit_xdata)
        self.fit_xdata = self.fit_xdata[isort]
        self.fit_ydata_in = self.fit_ydata_in[isort]
        
        # Get best fit
        self.fit_type, self.fit_result = fit(xdata, ydata)
    
    def predict(self, xdata):
        return self.fit_result.eval(x=xdata)


def add_missing_bins(edgeareas):
    """
    Some site-years have bins missing because they had zero area. Add those zeroes.
    """
    edgeareas2 = pd.DataFrame()
    index_list = ["Year", "edge"]
    for s, site in enumerate(edgeareas["site"].unique()):
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


def adjust_predicted_fits(ydata_yb):
    ydata_yb[ydata_yb < 0] = 0
    ydata_yb = ydata_yb / np.sum(ydata_yb, axis=1, keepdims=True)
    return ydata_yb


class LognormalFitParams():
    def __init__(self, center=3.5, sigma=1, amplitude=6):
        self.center = center
        self.sigma = sigma
        self.amplitude = amplitude

def _fit_lognormal(xdata, ydata, params):
    model = models.LognormalModel()
    params = model.make_params(center=params.center, sigma=params.sigma, amplitude=params.amplitude)
    result = model.fit(ydata, params, x=xdata)

    # print("Lognormal:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_exponential(xdata, ydata):
    model = models.ExponentialModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Exponential:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_gaussian(xdata, ydata):
    model = models.GaussianModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Gaussian:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_skewgaussian(xdata, ydata):
    model = models.SkewedGaussianModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Skewed Gaussian:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_logistic(xdata, ydata):
    model = models.StepModel(form="logistic")
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Logistic:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_linear(xdata, ydata):
    model = models.LinearModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Linear:")
    # print(fit_report(result))
    # print(" ")
    return result

def _fit_quadratic(xdata, ydata):
    model = models.QuadraticModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    # print("Quadratic:")
    # print(fit_report(result))
    # print(" ")
    return result

def fit(xdata, ydata, lognormal_params=LognormalFitParams()):
    
    # Try all fits
    results = {}
    results["lognormal"] = _fit_lognormal(xdata, ydata, lognormal_params)
    results["logistic"] = _fit_logistic(xdata, ydata)
    results["exponential"] = _fit_exponential(xdata, ydata)
    results["linear"] = _fit_linear(xdata, ydata)
    results["quadratic"] = _fit_quadratic(xdata, ydata)
    results["gaussian"] = _fit_gaussian(xdata, ydata)
    results["skewgaussian"] = _fit_skewgaussian(xdata, ydata)
    
    # Find best fit
    best_fit = None
    best_result = None
    best_metric = np.inf
    for i, (fit, result) in enumerate(results.items()):
        if result.aic < best_metric:
            best_metric = result.aic
            best_fit = fit
            best_result = result
    
    return best_fit, best_result

def get_site_lc_area(lc, totalareas, landcovers):
    lc_area = landcovers[landcovers["is_" + lc]].groupby(["Year", "site"]).sum()
    lc_area = lc_area.rename(columns={"sumarea": lc})
    totalareas = totalareas.join(lc_area[lc])
    totalareas = totalareas.fillna(value=0)
    return totalareas


def get_figure_filepath(this_dir, version, ef, title):
    outfile = f"{title}.{version}.{ef.fit_xvar}"
    if ef.sites_to_exclude:
        outfile += ".excl"
        for s, site in enumerate(ef.sites_to_exclude):
            if s > 0:
                outfile += ","
            outfile += str(site)
    if ef.fit_bootstrapped:
        outfile += ".bs"
    outfile += ".pdf"
    outpath = os.path.join(
        this_dir,
        "inout",
        version,
        outfile
    )
        
    return outpath

def get_version_info(version):
    vinfo = {}
    if int(version) in [20240506, 20240605]:
        vinfo["Nsites"] = 4
        vinfo["bins"] = [
            "<30",
            "30-60",
            "60-90",
            "90-120",
            "120-300",
            "300-500",
            "500-1000",
            "1000-2000",
            ">2000"
            ]
        vinfo["Nbins"] = len(vinfo["bins"])
    else:
        raise RuntimeError(f"Version {version} not recognized")
    return vinfo

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


def read_landcovers_legend(this_dir):
    landcovers_legend = pd.read_csv(os.path.join(this_dir, "MAPBIOMAS_Col6_Legenda_Cores.simple.csv"))
    return landcovers_legend

def import_landcovers_20240506(this_dir, version):
    
    # Import legend
    landcovers_legend = read_landcovers_legend(this_dir)
    
    # Import landcovers
    filename_template = os.path.join(this_dir, "inout", version, f"Landcover_clean_%d.csv")
    landcovers = read_combine_multiple_csvs(filename_template, version)
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})
    
    # Add labels
    landcovers = label_landcovers(landcovers_legend, landcovers)
    
    return landcovers

def label_landcovers(landcovers_legend, landcovers):
    landcovers = landcovers.assign(tmp = landcovers.landcover_num)
    landcovers = landcovers.set_index("tmp").join(landcovers_legend.set_index("landcover_num"))
    
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
        Nmatches = matching_landcovers.shape[0]
        if Nmatches == 0:
            unknown_types.append(num)
            landcover_str = unknown_str
        elif Nmatches != 1:
            raise RuntimeError(f"Expected 1 landcover matching {num}; found {Nmatches}")
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
        is_unknown[where_this_landcover] = Nmatches == 0

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


def predict_multiple_fits(xdata, edgeareas, edgefits, restrict_x=False):
    for b, bin in enumerate(pd.unique(edgeareas.edge)):
        ydata = edgefits[b].predict(xdata)
        if restrict_x:
            ydata[xdata<min(edgefits[b].fit_xdata)] = np.nan
            ydata[xdata>max(edgefits[b].fit_xdata)] = np.nan
        if b==0:
            ydata_yb = np.expand_dims(ydata, axis=1)
        else: 
            ydata_yb = np.concatenate((ydata_yb, np.expand_dims(ydata, axis=1)), axis=1)
    return ydata_yb


def read_combine_multiple_csvs(filename_template, version):
    vinfo = get_version_info(version)
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

def read_20240605(this_dir, filename_csv):
    df = pd.read_csv(filename_csv)
    df = df.rename(columns={
        "totalareas": "sumarea",
        "gridID": "site",
        "year": "Year",
        })
    
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
        siteyear_info = siteyear_info.rename(columns={"forestcover": "forestcover_frac"})
    df = df.drop(columns="forestcover")
    
    # Get MapBiomas type of formação florestal
    landcovers_legend = read_landcovers_legend(this_dir)
    fforest_idx = np.where(["formação florestal" in x.lower() for x in landcovers_legend["landcover_str"]])[0]
    if len(fforest_idx) != 1:
        raise RuntimeError(f"Expected 1 formação florestal row in landcovers legend; found {len(fforest_idx)}")
    fforest_idx = fforest_idx[0]
    fforest_num = landcovers_legend["landcover_num"][fforest_idx]
    
    # Which rows are edge classes?
    first_edge_class = 51
    last_edge_class = 59
    is_edge_class = (df["landcover"] >= first_edge_class) & (df["landcover"] <= last_edge_class)
    
    # Get DataFrame with just edge classes
    edgeareas = df[is_edge_class]
    edgeareas = edgeareas.rename(columns={"landcover": "edge"})
    edgeareas.edge -= first_edge_class - 1  # Change to edge bin numbers 1-9
    
    # Get formação florestal area
    tmp_indices = ["site", "Year"]
    fforest = edgeareas.groupby(tmp_indices).sum()
    fforest = fforest.reset_index(level=tmp_indices)
    fforest = fforest.drop(columns="edge")
    fforest = fforest.assign(landcover=fforest_num)
    
    # Get landcovers
    landcovers = df[~is_edge_class]
    landcovers = pd.concat((landcovers, fforest)) # include formação florestal
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    landcovers = label_landcovers(landcovers_legend, landcovers)
    if any(landcovers.isna().sum()):
        raise RuntimeError("NaN(s) found in landcovers")
    print(landcovers.head())
    print(landcovers.tail())

    return site_info, siteyear_info, edgeareas, landcovers

def get_color(vinfo, b):
    color = colormaps["jet_r"](b/(vinfo["Nbins"]-1))
    return color

def plot_fits_1plot(this_dir, version_str, xdata_01, vinfo, edgeareas, xvar, yvar, edgefits):
    ydata_yb = predict_multiple_fits(xdata_01, edgeareas, edgefits, restrict_x=True)
    ydata_adj_yb = adjust_predicted_fits(
    predict_multiple_fits(xdata_01, edgeareas, edgefits)
    )

    # for b, bin in enumerate(vinfo["bins"]):
    #     color = get_color(vinfo, b)
    #     plt.plot(xdata_01, ydata_yb[:,b], color=color)
    for b, bin in enumerate(vinfo["bins"]):
        color = get_color(vinfo, b)
        plt.plot(xdata_01, ydata_adj_yb[:,b], "--", color=color)
    plt.legend(vinfo["bins"])
    plt.xlabel(get_axis_labels(xvar))
    plt.ylabel(get_axis_labels(yvar))
    plt.title("Raw (solid) and adjusted (dashed) predictions")

    outpath = get_figure_filepath(this_dir, version_str, edgefits[0], "fit_lines_1plot")
    plt.savefig(outpath)

    plt.show()