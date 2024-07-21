import numpy as np
import pandas as pd
from EdgeFitType import EdgeFitType
import lin_edgeareas_module as lem
import fitting
from lin_edgeareas_module import XDATA_01

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class EdgeFitListType:

    def __init__(self, edgeareas, totalareas, sites_to_exclude, vinfo, finfo):

        # Initialize some variables
        self.finfo = finfo
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
        self.fit(edgeareas, totalareas, sites_to_exclude, vinfo, finfo)

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
                output += f"\n   (Adjustment changes net error from {self.km2_error[b].astype(float):.2e} [{self.pct_error[b].astype(float):.1f}%] to {self.km2_error_adj[b].astype(float):.2e} [{self.pct_error_adj[b].astype(float):.1f}%])"
        return output

    def _adjust_predicted_fits(self, ydata_yb, restrict_x):
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
        is_nan = np.squeeze(np.isnan(ydata_y))
        ydata_yb[is_nan,:] = 0

        # Check
        if np.any(np.isnan(ydata_yb)):
            raise RuntimeError("Unexpected NaN after adjusting predicted fits")

        return ydata_yb

    def fit(self, edgeareas, totalareas, sites_to_exclude, vinfo, finfo):
        bin_list = pd.unique(edgeareas.edge)
        for b, thisbin in enumerate(bin_list):
            print(f"Fitting bin {thisbin} ({b+1}/{len(bin_list)})...")
            ef = EdgeFitType(edgeareas, totalareas, sites_to_exclude, b, thisbin, vinfo)
            ef.ef_fit(finfo)
            self.edgefits.append(ef)

            # Check that all bins get the same X-axis inputs (every bin should be present in the data for every site-year, even if that bin's area is zero)
            xdata = ef.fit_result.userkws["x"]
            if b == 0:
                xdata0 = xdata.copy()
            elif not np.array_equal(xdata, xdata0):
                raise RuntimeError(f"X data used to fit {bin_list[b]} (index {b}) differs from that used to fit bin {bin_list[0]} (index {0})")

    def get_all_fits_and_adjs(self, xdata=XDATA_01, restrict_x=True):
        ydata_yb = self._predict_multiple_fits(xdata, restrict_x=restrict_x)
        ydata_adj_yb = self._adjust_predicted_fits(ydata_yb, restrict_x)
        return ydata_yb, ydata_adj_yb

    def _get_performance_info(self):
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

            # Get km2
            obs = ef.binarea
            fit = ef.get_bin_area_from_xy(ydata_yb[:,b])
            adj = ef.get_bin_area_from_xy(ydata_adj_yb[:,b])
            obs_sum += np.sum(obs)
            adj_sum += np.sum(adj)

            # Get RMSE
            self.rmse[b] = rmse(fit, obs)
            self.rmse_adj[b] = rmse(adj, obs)

            # Get NRMSE
            self.nrmse[b] = self.rmse[b] / np.mean(obs)
            self.nrmse_adj[b] = self.rmse_adj[b] / np.mean(obs)

            # Get km2 error
            self.km2_error[b] = np.sum(fit - obs)
            self.km2_error_adj[b] = np.sum(adj - obs)

            # Get % error
            self.pct_error[b] = 100 * self.km2_error[b] / np.sum(obs)
            self.pct_error_adj[b] = 100 * self.km2_error_adj[b] / np.sum(obs)

        if adj_sum != obs_sum:
            raise RuntimeError(f"adj_sum {adj_sum:.2e} != obs_sum {obs_sum:.2e}")

    def nbins(self):
        return len(self)

    def _predict_multiple_fits(self, xdata, restrict_x=False):
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
                ydata_yb = np.concatenate((ydata_yb, np.expand_dims(ydata, axis=1)), axis=1)
        return ydata_yb

def rmse(fit, obs):
    if np.any(np.isnan(fit)):
        raise RuntimeError("Unexpected NaN(s) in fit")
    if np.any(np.isnan(obs)):
        raise RuntimeError("Unexpected NaN(s) in obs")
    return np.sum((fit - obs) ** 2) ** 0.5 / np.mean(obs)