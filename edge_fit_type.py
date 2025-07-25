"""
Class for one edge bin's fit
"""

import numpy as np
import pandas as pd
from fitting import fit

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements


class EdgeFitType:
    """
    Class for fitting and analyzing a single edge bin.
    Handles data preparation, fitting, and prediction for one bin.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        edgeareas: pd.DataFrame,
        totalareas: pd.DataFrame,
        sites_to_exclude: list,
        b: int,
        this_bin,
        vinfo: dict,
    ):
        """
        Initialize EdgeFitType for a single bin.
        Args:
            edgeareas, totalareas: DataFrames for edge and total areas.
            sites_to_exclude: List of sites to exclude.
            b: Bin index.
            this_bin: Bin number.
            vinfo: Version info dict.
        """

        sites_to_include = [
            x for x in np.unique(edgeareas.site) if x not in sites_to_exclude
        ]

        # Get dataframe with just this edge, indexed by Year-site
        self.thisedge_df = (
            edgeareas[edgeareas.edge == this_bin]
            .drop(columns="edge")
            .set_index(["Year", "site"], verify_integrity=True)
        )
        self.thisedge_df = self.thisedge_df[
            self.thisedge_df.index.isin(sites_to_include, level="site")
        ]
        self.binarea = self.thisedge_df["sumarea"].values
        self.thisedge_df = self.thisedge_df.rename(columns={"sumarea": "bin"})

        # Join with areas of different land cover types
        self.thisedge_df = self.thisedge_df.join(totalareas)
        if any(self.thisedge_df.isna().sum()):
            raise RuntimeError("NaN(s) found after joining thisedge_df and totalareas")

        # Get total forest area in this bin's gridcell
        self.all_forest_area = self.thisedge_df["forest_from_ea"].values.copy()

        # Convert to fractional area
        if np.any(np.isnan(self.thisedge_df)):
            raise RuntimeError("Unexpected NaN in self.thisedge_df")
        self.thisedge_df = self.thisedge_df.div(self.thisedge_df.sitearea, axis=0)
        if np.any(np.isnan(self.thisedge_df)):
            raise RuntimeError("Unexpected NaN in self.thisedge_df")

        # Get edge bin area as fraction of total forest
        bin_as_frac_allforest = self.thisedge_df.bin / self.thisedge_df.forest_from_ea
        bin_as_frac_allforest[self.thisedge_df.forest_from_ea == 0] = 0
        self.thisedge_df = self.thisedge_df.assign(
            bin_as_frac_allforest=bin_as_frac_allforest,
        )
        if np.any(np.isnan(self.thisedge_df)):
            raise RuntimeError("Unexpected NaN in self.thisedge_df")

        # Save other info
        self.bin_index = b
        self.bin_number = this_bin
        self.bin_name = vinfo["bins_out"][b]
        self.sites_to_exclude = sites_to_exclude

        # Initialize other members
        self.fit_xdata_orig = None
        self.fit_ydata_orig = None
        self.fit_xdata = np.array([])
        self.fit_ydata = np.array([])
        self.predicted_ydata = None
        self.fit_type = None
        self.fit_result = None
        self.bs_xdata = None
        self.bs_ydata = None
        self.predicted_adj_ydata = None
        self.finfo = None

    def __str__(self):
        """
        String representation of EdgeFitType, including fit type and r-squared.
        Returns:
            str: Summary of fit.
        """
        prefix = f"EdgeFitType for bin {self.bin_number} ({self.bin_name} m): "
        if self.fit_result is None:
            output = prefix + "Not yet fit"
        else:
            output = (
                prefix
                + f"{self.fit_type} r-squared {np.round(self.fit_result.rsquared, 3)}"
            )

        return output

    def ef_fit(self, finfo: dict):
        """
        Fit the edge bin using settings from fit info dictionary.
        Args:
            finfo (dict): Fit info dict.
        """
        self.finfo = finfo

        # Get X and Y data for fitting, plus anything for analysis
        if finfo["xvar"] == "croppast_frac_croppastfor":
            self.thisedge_df = self.thisedge_df.assign(
                croppast_frac_croppastfor=self.thisedge_df.croppast
                / (self.thisedge_df.fforest + self.thisedge_df.croppast)
            )
            self.fit_xdata_orig = self.thisedge_df.croppast_frac_croppastfor.values
        else:
            self.fit_xdata_orig = self.thisedge_df[finfo["xvar"]].values
        self.fit_ydata_orig = self.thisedge_df[finfo["yvar"]].values

        if np.any(np.isnan(self.fit_xdata_orig)):
            raise RuntimeError("Unexpected NaN in self.fit_xdata_orig")
        if np.any(np.isnan(self.fit_ydata_orig)):
            raise RuntimeError("Unexpected NaN in self.fit_ydata_orig")

        # Bootstrap across bins of X-axis to ensure even weighting
        if finfo["bootstrap"]:
            # Set up X-axis bins
            n_xbins = 10
            x_max = max(self.fit_xdata_orig)
            x_min = min(self.fit_xdata_orig)
            if x_max <= x_min:
                raise RuntimeError("x_max must be > x_min")
            step = (x_max - x_min) / (n_xbins)
            bin_boundaries = np.arange(x_min, x_max + step, step)

            # Which data points correspond to each X-axis bin?
            cond_list = []
            for b in np.arange(n_xbins):
                lo = bin_boundaries[b]
                hi = bin_boundaries[b + 1]
                if b == n_xbins - 1:
                    cond_hi = self.fit_xdata_orig <= hi
                else:
                    cond_hi = self.fit_xdata_orig < hi
                cond = cond_hi & (self.fit_xdata_orig >= lo)
                cond_list.append(cond)

            # How many samples should we take from each X-axis bin?
            n_choose = max(sum(x) for x in cond_list)

            # Take samples
            for cond in cond_list:
                if not any(cond):
                    continue
                where_cond = np.where(cond)[0]
                rng = np.random.default_rng(seed=1987)
                chosen = rng.choice(where_cond, n_choose)
                if len(chosen) != n_choose:
                    raise RuntimeError(
                        f"Expected {n_choose} samples; got {len(chosen)}"
                    )
                self.fit_xdata = np.concatenate(
                    (self.fit_xdata, self.fit_xdata_orig[chosen])
                )
                self.fit_ydata = np.concatenate(
                    (self.fit_ydata, self.fit_ydata_orig[chosen])
                )
            self.bs_xdata = self.fit_xdata
            self.bs_ydata = self.fit_ydata

            # Check
            if any(np.isnan(self.fit_xdata)) or any(np.isnan(self.fit_ydata)):
                raise RuntimeError("NaN after bootstrap sampling")
            for b in range(n_xbins):
                lo = bin_boundaries[b]
                hi = bin_boundaries[b + 1]
                if b == n_xbins - 1:
                    cond_hi = self.fit_xdata <= hi
                else:
                    cond_hi = self.fit_xdata < hi
                cond = cond_hi & (self.fit_xdata >= lo)
                n_found = np.sum(cond)
                if not any(cond_list[b]):
                    n_expected = 0
                else:
                    n_expected = n_choose
                if n_found != n_expected:
                    raise RuntimeError(
                        f"Expected {n_expected} points in {lo}-{hi}; found {n_found}"
                    )
        else:
            self.fit_xdata = self.fit_xdata_orig
            self.fit_ydata = self.fit_ydata_orig

        # Get best fit
        self.fit_type, self.fit_result = fit(self.fit_xdata, self.fit_ydata)
        # if self.fit_type == "lognormal":
        #     for p in ["amplitude", "center", "sigma"]:
        #         val = self.fit_result.params[p].value
        #         print("   ---------")
        #         print(f"   {p}: {val:.3g}")
        self.predicted_ydata = self.predict(self.fit_xdata)
        if np.any(np.isnan(self.predicted_ydata)):
            raise RuntimeError("Unexpected NaN in predicted_ydata")

    def predict(self, xdata: np.ndarray):
        """
        Predict values for given xdata using the fitted model.
        Args:
            xdata (NumPy array): Input data for prediction.
        Returns:
            NumPy array: Predicted values.
        """
        result = self.fit_result.eval(x=xdata)
        if np.any(np.isnan(result)):
            raise RuntimeError("Unexpected NaN in predict()")
        return result

    def get_bin_area_from_xy(self, data_in: np.ndarray):
        """
        Convert predicted values to bin area using forest area.
        Args:
            data_in (NumPy array): Predicted values.
        Returns:
            NumPy array: Bin area values.
        """
        if np.any(np.isnan(data_in)):
            raise RuntimeError("Unexpected NaN in data_in")

        if self.finfo["yvar"] == "bin_as_frac_allforest":
            data_out = data_in * self.all_forest_area
        else:
            raise RuntimeError(f"Unrecognized yvar: {self.finfo['yvar']}")
        return data_out

    def print_fitted_equation(self):
        """
        Print the fitted equation and parameters for the bin.
        """
        if self.fit_type in ["gaussian", "lognormal"]:
            if self.fit_type == "gaussian":
                equation = "y = A / (σ * √2π) * exp(-(x-µ)^2 / (2σ^2))"
            elif self.fit_type == "lognormal":
                equation = "y = A / (x * σ * √2π) * exp(-(ln(x)-µ)^2 / (2σ^2))"
            else:
                raise RuntimeError(f"Unrecognized fit type: {self.fit_type}")
            where = (
                "where:\n"
                + f"   A = amplitude = {self.param('amplitude')}\n"
                + f"   σ = sigma = {self.param('sigma')}\n"
                + f"   µ = center = {self.param('center')}"
            )
        elif self.fit_type == "quadratic":
            equation = "   y = ax^2 + bx + c"
            where = (
                "where:\n"
                + f"   a = {self.param('a')}\n"
                + f"   b = {self.param('b')}\n"
                + f"   c = {self.param('c')}"
            )
        else:
            raise RuntimeError(f"Unrecognized fit type: {self.fit_type}")
        print(f"Bin {self.bin_name}: {self.fit_type} fit")
        print(equation)
        print(where)

    def param(self, param_name: str):
        """
        Get the value of a fit parameter by name.
        Args:
            param_name (str): Name of parameter.
        Returns:
            float: Parameter value.
        """
        return self.fit_result.params[param_name].value
