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
        self.nrmse = None
        self.nrmse_adj = None

        # Fit all bins
        self.edgefits = []
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
        for ef in self:
            output += "\n" + str(ef)
        return output

    def _adjust_predicted_fits(self, ydata_yb):
        # Checks
        if ydata_yb.ndim == 1 or ydata_yb.shape[1] == 1:
            raise RuntimeError(
                "It only makes sense to call adjust_predicted_fits() with multiple bins!"
            )
        if np.any(np.isnan(ydata_yb)):
            raise RuntimeError("Unexpected NaN before adjusting predicted fits")

        # Don't allow negative areas
        ydata_yb[ydata_yb < 0] = 0

        # Ensure sum to 1
        axis = len(ydata_yb.shape) - 1
        ydata_y = np.sum(ydata_yb, axis=axis, keepdims=True)
        ydata_yb = ydata_yb / ydata_y
        ydata_yb[:,np.isnan(ydata_y)] = 0

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

    def get_all_fits_and_adjs(self, xdata=XDATA_01):
        ydata_yb = self._predict_multiple_fits(xdata, restrict_x=True)
        ydata_adj_yb = self._adjust_predicted_fits(ydata_yb)
        return ydata_yb, ydata_adj_yb

    def _get_performance_info(self):
        empty_array = np.full(self.nbins(), np.nan)
        self.rmse = empty_array.copy()
        self.rmse_adj = empty_array.copy()
        self.km2_error = empty_array.copy()
        self.km2_error_adj = empty_array.copy()
        self.pct_error = empty_array.copy()
        self.pct_error_adj = empty_array.copy()

        xdata = self[0].fit_xdata
        if not np.array_equal(self[0].fit_xdata, self[1].fit_xdata):
            raise RuntimeError("fit_xdata unexpectedly differs between bins 0 and 1")
        ydata_yb, ydata_adj_yb = self.get_all_fits_and_adjs(xdata=xdata)
        for b, ef in enumerate(self):

            # Get km2
            obs = ef.get_bin_area_from_xy(xdata)
            fit = ef.get_bin_area_from_xy(ydata_yb[:,b])
            adj = ef.get_bin_area_from_xy(ydata_adj_yb[:,b])

            # Get RMSE
            self.rmse[b] = rmse(fit, obs)
            self.rmse_adj[b] = rmse(adj, obs)

            # Get km2 error
            self.km2_error[b] = np.sum(fit - obs)
            self.km2_error_adj[b] = np.sum(adj - obs)

            # Get % error
            self.pct_error[b] = 100 * self.km2_error[b] / np.sum(obs)
            self.pct_error_adj[b] = 100 * self.km2_error_adj[b] / np.sum(obs)

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
                    ydata[xdata < min(edgefit.fit_xdata)] = np.nan
                    ydata[xdata > max(edgefit.fit_xdata)] = np.nan
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