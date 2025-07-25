"""
Class to contain a list of EdgeFit objects and do useful things with it.
Handles fitting, performance metrics, and output formatting for edge bins.
"""

import numpy as np
import pandas as pd
from edge_fit_type import EdgeFitType
import lin_edgeareas_module as lem
from lin_edgeareas_module import XDATA_01

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-instance-attributes


class EdgeFitListType:
    """
    Class to contain a list of EdgeFit objects and do useful things with it.
    """

    def __init__(self, *, edgeareas, totalareas, vinfo, finfo):

        # Initialize some variables
        self.finfo = finfo
        self.vinfo = vinfo
        self.km2_error = None
        self.km2_error_adj = None
        self.rmse = None
        self.rmse_adj = None
        self.nrmse = None
        self.nrmse_adj = None

        # Fit all bins
        self.edgefits = []
        if np.any(np.isnan(edgeareas)):
            raise RuntimeError("Unexpected NaN in edgeareas")
        if np.any(np.isnan(totalareas)):
            raise RuntimeError("Unexpected NaN in totalareas")
        self.fit(
            edgeareas=edgeareas,
            totalareas=totalareas,
            vinfo=vinfo,
            finfo=finfo,
        )

        # Get performance info
        self._get_performance_info()

        # Print fit info
        print(self)

    def __iter__(self):
        return (x for x in self.edgefits)

    def __getitem__(self, index):
        return self.edgefits[index]

    def __len__(self):
        return len(self.edgefits)

    def __str__(self):
        output = ""
        for b, ef in enumerate(self):
            output += "\n" + str(ef)
            if self.km2_error is not None:
                output += (
                    "\n   (Adjustment changes net error from "
                    f"{self.km2_error[b].astype(float):.2g} km2 "
                    f"[{self.pct_error[b].astype(float):.1f}%] to "
                    f"{self.km2_error_adj[b].astype(float):.2g} km2 "
                    f"[{self.pct_error_adj[b].astype(float):.1f}%])"
                )
        return output

    def _adjust_predicted_fits(self, ydata_yb: np.ndarray, restrict_x: bool):
        """
        Checks and adjusts predicted fit values for multiple bins:
        - Sets any negative predicted fits to zero.
        - Normalizes predicted fits so the sum across bins equals 1.

        Args:
            ydata_yb (NumPy array): Array of predicted fit values, expected to have multiple bins.
            restrict_x (bool): Whether the predicted fits got set to NaN for X values outside the
                training range. If False, checks that there are no NaN values in the predicted fits.

        Returns:
            NumPy array: The adjusted and normalized array of predicted fit values.

        Raises:
            RuntimeError: If input does not contain multiple bins, or if unexpected NaN values are
            found before or after adjustment.
        """
        # Checks
        if ydata_yb.ndim == 1 or ydata_yb.shape[1] == 1:
            raise RuntimeError(
                "It only makes sense to call adjust_predicted_fits() with multiple bins!"
            )
        if not restrict_x and np.any(np.isnan(ydata_yb)):
            raise RuntimeError("Unexpected NaN before adjusting predicted fits")

        # Don't allow negative areas
        ydata_yb[ydata_yb < 0] = 0

        # Ensure sum to 1
        axis = len(ydata_yb.shape) - 1
        ydata_y = np.sum(ydata_yb, axis=axis, keepdims=True)
        ydata_yb = ydata_yb / ydata_y

        # ???
        is_nan = np.squeeze(np.isnan(ydata_y))
        ydata_yb[is_nan, :] = 0

        # Check
        if np.any(np.isnan(ydata_yb)):
            raise RuntimeError("Unexpected NaN after adjusting predicted fits")

        return ydata_yb

    def fit(
        self,
        *,
        edgeareas: pd.DataFrame,
        totalareas: pd.DataFrame,
        vinfo: dict,
        finfo: dict,
    ):
        """
        Fit each edge forest bin.

        Arguments:
            edgeareas (pd.DataFrame): DataFrame containing edge area information.
            totalareas (pd.DataFrame): DataFrame containing total area information.
            vinfo (dict): Variable information dictionary.
            finfo (dict): Fit information dictionary.
        """
        bin_list = pd.unique(edgeareas.edge)
        for b, thisbin in enumerate(bin_list):
            print(f"Fitting bin {thisbin} ({b+1}/{len(bin_list)})...")
            ef = EdgeFitType(
                edgeareas=edgeareas,
                totalareas=totalareas,
                sites_to_exclude=finfo["sites_to_exclude"],
                b=b,
                this_bin=thisbin,
                vinfo=vinfo,
            )
            ef.ef_fit(finfo)
            self.edgefits.append(ef)

            # Check that all bins get the same X-axis inputs (every bin should be present in the
            # data for every site-year, even if that bin's area is zero)
            xdata = ef.fit_result.userkws["x"]
            if b == 0:
                xdata0 = xdata.copy()
            elif not np.array_equal(xdata, xdata0):
                raise RuntimeError(
                    f"X data used to fit {bin_list[b]} (index {b}) differs from that used to fit "
                    "bin {bin_list[0]} (index {0})"
                )

    def get_all_fits_and_adjs(self, xdata=XDATA_01, restrict_x=True):
        """
        Get predicted fits (before and after adjustment) for all bins.

        Arguments:
            xdata (NumPy array): Input data for prediction (default: XDATA_01, defined in
                lin_edgeareas_module).
            restrict_x (bool): Whether to restrict predictions to the range of fitted x values
                (default: True).

        Returns:
            Tuple of (predicted fits, adjusted fits) for all bins.
        """
        ydata_yb = self._predict_multiple_fits(xdata, restrict_x=restrict_x)
        ydata_adj_yb = self._adjust_predicted_fits(ydata_yb, restrict_x)
        return ydata_yb, ydata_adj_yb

    def _get_performance_info(self):
        """
        Computes and stores performance metrics for each bin's fit.

        The following metrics are calculated for both the fitted and adjusted data:
            - Root Mean Square Error (RMSE)
            - Normalized RMSE (NRMSE)
            - Total error in square kilometers (km2_error)
            - Percent error (% error)

        The method initializes arrays to store these metrics for each bin, then iterates over all
        bins to compute:
            - RMSE and NRMSE between fitted/adjusted bin areas and observed bin areas.
            - Total error in square kilometers and percent error for each bin.
            - Checks that the sum of adjusted areas matches the sum of observed areas.

        Raises:
            RuntimeError: If the x-data used for fitting differs between bins 0 and 1, or if the sum
                of adjusted areas does not match the sum of observed areas.
        """
        empty_array = np.full(self.nbins(), np.nan)
        self.rmse = empty_array.copy()
        self.rmse_adj = empty_array.copy()
        self.nrmse = empty_array.copy()
        self.nrmse_adj = empty_array.copy()
        self.km2_error = empty_array.copy()
        self.km2_error_adj = empty_array.copy()
        self.pct_error = empty_array.copy()
        self.pct_error_adj = empty_array.copy()

        if not np.array_equal(self[0].fit_xdata, self[1].fit_xdata):
            raise RuntimeError("fit_xdata unexpectedly differs between bins 0 and 1")
        ydata_yb, ydata_adj_yb = self.get_all_fits_and_adjs(xdata=None)
        adj_sum = 0
        obs_sum = 0
        for b, ef in enumerate(self):

            # Get m2
            obs = ef.binarea
            obs_km2 = 1e-6 * obs
            fit = ef.get_bin_area_from_xy(ydata_yb[:, b])
            adj = ef.get_bin_area_from_xy(ydata_adj_yb[:, b])
            obs_sum += np.sum(obs)
            adj_sum += np.sum(adj)

            # Get RMSE
            self.rmse[b] = rmse(fit, obs)
            self.rmse_adj[b] = rmse(adj, obs)

            # Get NRMSE
            self.nrmse[b] = self.rmse[b] / np.mean(obs)
            self.nrmse_adj[b] = self.rmse_adj[b] / np.mean(obs)

            # Get km2 error
            self.km2_error[b] = 1e-6 * np.sum(fit - obs)
            self.km2_error_adj[b] = 1e-6 * np.sum(adj - obs)

            # Get % error
            self.pct_error[b] = 100 * self.km2_error[b] / np.sum(obs_km2)
            self.pct_error_adj[b] = 100 * self.km2_error_adj[b] / np.sum(obs_km2)

        if not np.isclose(adj_sum, obs_sum):
            raise RuntimeError(f"adj_sum {adj_sum:.2e} != obs_sum {obs_sum:.2e}")

    def nbins(self):
        """
        Return the number of bins in the edge fit list.
        Returns:
            int: Number of bins.
        """
        return len(self)

    def _predict_multiple_fits(self, xdata: np.ndarray, restrict_x=False):
        """
        Predicts output values for multiple edge fits using the provided xdata.

        Iterates over each edge fit in the collection, applies the `predict` method of each fit to
        the input `xdata`, and aggregates the results into a 2D array where each column corresponds
        to predictions from one edge fit.

        If `xdata` is None, don't bother using `predict`. Instead, just use the X data used in the
        fit (and the resulting, already-saved predictions as Y).

        If `restrict_x` is True, sets predictions to NaN for x values outside the range of the fit's
        training data.

        Raises:
            RuntimeError: If any NaN values are encountered in the predictions after calling
            `edgefit.predict()`.

        Args:
            xdata (NumPy array): Input data for prediction. If None, uses each fit's training data.
            restrict_x (bool, optional): If True, restricts predictions to the range of each fit's
                training data.

        Returns:
            NumPy array: 2D array of predicted values, shape (len(xdata), number of edge fits).
        """
        for b, edgefit in enumerate(self):
            if xdata is None:
                xdata = edgefit.fit_xdata
                ydata = edgefit.fit_ydata
            else:
                ydata = edgefit.predict(xdata)
                if np.any(np.isnan(ydata)):
                    raise RuntimeError("Unexpected NaN after calling edgefit.predict()")
                if restrict_x:
                    is_too_low = xdata < min(edgefit.fit_xdata)
                    if np.any(is_too_low):
                        ydata[is_too_low] = np.nan
                    is_too_high = xdata > max(edgefit.fit_xdata)
                    if np.any(is_too_high):
                        ydata[is_too_high] = np.nan
            if b == 0:
                ydata_yb = np.expand_dims(ydata, axis=1)
            else:
                ydata_yb = np.concatenate(
                    (ydata_yb, np.expand_dims(ydata, axis=1)), axis=1
                )
        return ydata_yb

    def print_fitted_equations(self):
        """
        Prints the fitted equations for each edge fit in the collection.
        """
        for edgefit in self:
            print(" ")
            edgefit.print_fitted_equation()

    def print_cdl_lines(self, cdl_file: str):
        """
        Writes a CDL (Common Data form Language) file with lines needed for the FATES param file.
        Also prints those lines to stdout.

        The method generates CDL definitions for:
            - Dimensions: Number of forest edge bins.
            - Variables: Bin edges and fit parameters for Gaussian, lognormal, and quadratic fits,
              including units and descriptive long names.
            - Data: Populates the variables with fit results and bin edges.

        Parameters:
            cdl_file (str): Path to the file where CDL lines will be written.

        Raises:
            RuntimeError: If an unrecognized fit type is encountered in the edge fits.

        Side Effects:
            Appends CDL-formatted text to the specified file, describing the structure and data of
            forest edge fits.
        """
        # Dimensions
        lem.print_and_write("dimensions:", cdl_file)
        lem.print_and_write(
            f"        fates_edgeforest_bins = {self.nbins()} ;", cdl_file
        )

        # Variables
        lem.print_and_write("\n\nvariables:", cdl_file)
        ind0 = "        "
        ind1 = "                "
        lem.print_and_write(
            f"{ind0}double fates_edgeforest_bin_edges(fates_edgeforest_bins) ;",
            cdl_file,
        )
        lem.print_and_write(f'{ind1}fates_edgeforest_bin_edges:units = "m" ;', cdl_file)
        long_name = "Boundaries of forest edge bins (for each bin, include value closest to zero)"
        lem.print_and_write(
            f'{ind1}fates_edgeforest_bin_edges:long_name = "{long_name}" ;', cdl_file
        )
        cdl_dict = {}
        init_var = ["_"] * self.nbins()
        suffix_base = 'for calculating forest area in each edge bin (THISFIT fit)" ;'

        # Variables for gaussian fits
        suffix = suffix_base.replace("THISFIT", "gaussian")
        var = "fates_edgeforest_gaussian_amplitude"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Amplitudes {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_gaussian_sigma"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Sigmas {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_gaussian_center"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Centers {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        # Variables for lognormal fits
        suffix = suffix_base.replace("THISFIT", "lognormal")
        var = "fates_edgeforest_lognormal_amplitude"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Amplitudes {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_lognormal_sigma"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Sigmas {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_lognormal_center"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(f'{ind1}{var}:long_name = "Centers {suffix}', cdl_file)
        cdl_dict[var] = init_var.copy()

        # Variables for quadratic fits
        suffix = suffix_base.replace("THISFIT", "quadratic")
        var = "fates_edgeforest_quadratic_a"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(
            f'{ind1}{var}:long_name = "x^2 coefficient {suffix}', cdl_file
        )
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_quadratic_b"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(
            f'{ind1}{var}:long_name = "x^1 coefficient {suffix}', cdl_file
        )
        cdl_dict[var] = init_var.copy()

        var = "fates_edgeforest_quadratic_c"
        lem.print_and_write(f"{ind0}double {var}(fates_edgeforest_bins) ;", cdl_file)
        lem.print_and_write(f'{ind1}{var}:units = "unitless" ;', cdl_file)
        lem.print_and_write(
            f'{ind1}{var}:long_name = "x^0 coefficient {suffix}', cdl_file
        )
        cdl_dict[var] = init_var.copy()

        # Fill values
        cdl_dict["fates_edgeforest_bin_edges"] = [0] + self.vinfo["bin_edges_out"]
        for b, edgefit in enumerate(self):
            ft = edgefit.fit_type
            if ft in ["gaussian", "lognormal"]:
                param_list = ["amplitude", "sigma", "center"]
            elif ft == "quadratic":
                param_list = ["a", "b", "c"]
            else:
                raise RuntimeError(f"Unrecognized fit_type {edgefit.fit_type}")
            for param in param_list:
                this_dict_entry = f"fates_edgeforest_{ft}_{param}"
                cdl_dict[this_dict_entry][b] = edgefit.param(param)

        # Print values
        lem.print_and_write("\n\ndata:\n", cdl_file)
        join_str = ", "
        for key, value in cdl_dict.items():
            joined_str = join_str.join([str(x) for x in value])
            lem.print_and_write(f" {key} = {joined_str} ;\n", cdl_file)


def rmse(fit: np.ndarray, obs: np.ndarray):
    """
    Calculate the root mean square error (RMSE) between fit and observed values.

    Arguments:
        fit (NumPy array): Array of fitted values.
        obs (NumPy array): Array of observed values.

    Returns:
        float: The RMSE value.
    """
    if np.any(np.isnan(fit)):
        raise RuntimeError("Unexpected NaN(s) in fit")
    if np.any(np.isnan(obs)):
        raise RuntimeError("Unexpected NaN(s) in obs")
    return np.sum((fit - obs) ** 2) ** 0.5 / np.mean(obs)
