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

        # Save some stuff
        self.finfo = finfo

        # Fit all bins
        self.edgefits = []
        self.fit(edgeareas, totalareas, sites_to_exclude, vinfo, finfo)

        # Print fit info
        print(self)


    def __iter__(self):
        return (x for x in self.edgefits)

    def __getitem__(self, index):
        return self.edgefits[index]

    def __str__(self):
        output = ""
        for ef in self:
            output += "\n" + str(ef)
        return output

    def _adjust_predicted_fits(self, ydata_yb):
        if ydata_yb.ndim == 1 or ydata_yb.shape[1] == 1:
            raise RuntimeError(
                "It only makes sense to call adjust_predicted_fits() with multiple bins!"
            )
        ydata_yb[ydata_yb < 0] = 0
        axis = len(ydata_yb.shape) - 1
        ydata_yb = ydata_yb / np.sum(ydata_yb, axis=axis, keepdims=True)
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

    def _predict_multiple_fits(self, xdata, restrict_x=False):
        for b, edgefit in enumerate(self):
            if xdata is None:
                xdata = edgefit.fit_xdata
                ydata = edgefit.fit_ydata
            else:
                ydata = edgefit.predict(xdata)
                if restrict_x:
                    ydata[xdata < min(edgefit.fit_xdata)] = np.nan
                    ydata[xdata > max(edgefit.fit_xdata)] = np.nan
            if b == 0:
                ydata_yb = np.expand_dims(ydata, axis=1)
            else:
                ydata_yb = np.concatenate((ydata_yb, np.expand_dims(ydata, axis=1)), axis=1)
        return ydata_yb
