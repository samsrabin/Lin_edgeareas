import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd
from lmfit import models, fit_report

class EdgeFitType:
    def __init__(self, edgeareas, totalareas, sites_to_exclude, b, bin, vinfo):
        
        sites_to_include = [x for x in np.unique(edgeareas.site) if x not in sites_to_exclude]
        
        # Get dataframe with just this edge, indexed by Year-site
        self.thisedge_df = edgeareas[edgeareas.edge==bin].drop(columns="edge").set_index(["Year", "site"], verify_integrity=True)
        self.thisedge_df = self.thisedge_df[self.thisedge_df.index.isin(sites_to_include, level="site")]
        self.thisedge_df = self.thisedge_df.rename(columns={"sumarea": "bin"})
        
        # Join with areas of different land cover types
        self.thisedge_df = self.thisedge_df.join(totalareas)
        
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
    
    def __str__(self):
        prefix = f"EdgeFitType for bin {self.bin_number} ({self.bin_name} m): "
        if self.fit_result is None:
            output = prefix + "Not yet fit"
        else:
            output = prefix + f"{self.fit_type} r-squared {np.round(self.fit_result.rsquared, 3)}"
        return output

    def ef_fit(self, xvar, yvar):
        
        # Get X and Y data for fitting
        self.fit_xdata = self.thisedge_df[xvar].values
        self.fit_ydata_in = self.thisedge_df[yvar].values
        
        # Sort X and Y data (helpful for plotting)
        isort = np.argsort(self.fit_xdata)
        self.fit_xdata = self.fit_xdata[isort]
        self.fit_ydata_in = self.fit_ydata_in[isort]
        
        # Get best fit
        self.fit_type, self.fit_result = fit(self.fit_xdata, self.fit_ydata_in)
    
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


def get_figure_filepath(this_dir, version, xvar, sites_to_exclude, title):
    outfile = f"{title}.{version}.{xvar}"
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
        
    return outpath

def get_version_info(version):
    vinfo = {}
    if version == "20240506":
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
    else:
        axis = var
    return axis

def import_landcovers(this_dir, version):
    # Import legend
    landcovers_legend = pd.read_csv(os.path.join(this_dir, "MAPBIOMAS_Col6_Legenda_Cores.simple.csv"))
    
    # Import landcovers
    filename_template = os.path.join(this_dir, "inout", version, f"Landcover_clean_%d.csv")
    landcovers = read_combine_multiple_csvs(filename_template, version)
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})
    
    # Add labels
    landcovers = landcovers.assign(tmp = landcovers.landcover_num)
    landcovers = landcovers.set_index("tmp").join(landcovers_legend.set_index("landcover_num"))
    
    # Regenerate index
    index = np.arange(len(landcovers.index))
    landcovers = landcovers.set_index(index)
    
    # Add legend-based info
    is_water = []
    is_forest = []
    is_fforest = []
    is_crop = []
    is_pasture = []
    is_croppast = []
    is_agri = []
    is_unvegd = []
    is_vegd = []
    is_unobs = []
    for i, num in enumerate(landcovers.landcover_num):
        
        # Get landcover string for this landcover code
        matching_landcovers = landcovers_legend[landcovers_legend.landcover_num == num]
        Nmatches = matching_landcovers.shape[0]
        if Nmatches != 1:
            raise RuntimeError(f"Expected 1 landcover matching {num}; found {Nmatches}")
        landcover_str = matching_landcovers.landcover_str.values[0]
        
        # Classify based on landcover string
        is_water.append("#5" in landcover_str)
        is_forest.append("#1" in landcover_str)
        is_fforest.append("#1.1" in landcover_str)
        is_crop.append("#3.2" in landcover_str)
        is_pasture.append("#3.1" in landcover_str)
        is_croppast.append(is_crop[i] or is_pasture[i] or "3.4" in landcover_str)
        is_agri.append("#3" in landcover_str)
        is_unvegd.append("#4" in landcover_str)
        is_vegd.append(not is_water[i] and not is_unvegd[i])
        is_unobs.append("#6" in landcover_str)

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