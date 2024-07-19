import numpy as np
from fitting import fit

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class EdgeFitType:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, edgeareas, totalareas, sites_to_exclude, b, this_bin, vinfo):

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
        self.thisedge_df = self.thisedge_df.div(self.thisedge_df.sitearea, axis=0)

        # Get edge bin area as fraction of total forest
        self.thisedge_df = self.thisedge_df.assign(
            bin_as_frac_allforest=self.thisedge_df.bin / self.thisedge_df.forest_from_ea
        )

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
        self.fit_bootstrapped = None
        self.fit_xvar = None
        self.fit_yvar = None
        self.bs_xdata = None
        self.bs_ydata = None
        self.nrmse = None
        self.nrmse_adj = None
        self.get_net_km2_error = None
        self.km2_error = None
        self.km2_error_adj = None
        self.predicted_adj_ydata = None

    def __str__(self):
        prefix = f"EdgeFitType for bin {self.bin_number} ({self.bin_name} m): "
        if self.fit_result is None:
            output = prefix + "Not yet fit"
        else:
            output = (
                prefix
                + f"{self.fit_type} r-squared {np.round(self.fit_result.rsquared, 3)}"
            )


        if self.km2_error is not None:
            output += f"\n   (NRMSE {np.round(self.nrmse_adj/self.nrmse, 1)}x worse after adjustment) {np.max(self.binarea)}"
            decimals = 2
            km2_error = np.round(self.km2_error, decimals)
            km2_error_adj = np.round(self.km2_error_adj, decimals)
            total_bin_area = np.sum(self.binarea)
            pct_error = 100 * self.km2_error / total_bin_area
            pct_error_adj = 100 * self.km2_error_adj / total_bin_area
            decimals = 1
            pct_error = np.round(pct_error, decimals)
            pct_error_adj = np.round(pct_error_adj, decimals)
            output += f"\n   (Adjustment changes net error from {km2_error} [{pct_error}%] to {km2_error_adj} [{pct_error_adj}%])"

            # output += "\n   =========Troubleshooting========="
            # output += f"\n   Bin area, obs: {np.sum(self.binarea)}"
            # output += f"\n   Bin area, fit: {np.sum(self.fit_ydata_out*self.all_forest_area)}"
            # output += f"\n   Bin area, adj: {np.sum(self.fit_ydata_out_adj*self.all_forest_area)}"

        return output

    def ef_fit(self, xvar, yvar, bootstrap):
        self.fit_xvar = xvar
        self.fit_yvar = yvar
        self.fit_bootstrapped = bootstrap

        # Get X and Y data for fitting, plus anything for analysis
        if xvar == "croppast_frac_croppastfor":
            self.thisedge_df = self.thisedge_df.assign(
                croppast_frac_croppastfor=self.thisedge_df.croppast
                / (self.thisedge_df.fforest + self.thisedge_df.croppast)
            )
            self.fit_xdata_orig = self.thisedge_df.croppast_frac_croppastfor.values
        else:
            self.fit_xdata_orig = self.thisedge_df[xvar].values
        self.fit_ydata_orig = self.thisedge_df[yvar].values
        if bootstrap:
            print("Not yet able to get net km2 error when bootstrapping")
        elif self.fit_yvar == "bin_as_frac_allforest":
            def get_net_km2_error(self, fit_vals, obs_vals, indices=None):
                diff_area = (fit_vals - obs_vals) * self.all_forest_area
                if indices is not None:
                    diff_area = diff_area[indices]
                return diff_area
        else:
            print(f"You haven't told me how to get net km2 error for fit_yvar {self.fit_yvar}. Skipping.")

        # Bootstrap across bins of X-axis to ensure even weighting
        if bootstrap:
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
                self.fit_xdata = np.concatenate((self.fit_xdata, self.fit_xdata_orig[chosen]))
                self.fit_ydata = np.concatenate((self.fit_ydata, self.fit_ydata_orig[chosen]))
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

        # Sort X and Y data (helpful for plotting)
        isort = np.argsort(self.fit_xdata_orig)
        self.fit_xdata_orig = self.fit_xdata_orig[isort]
        self.fit_ydata_orig = self.fit_ydata_orig[isort]
        isort = np.argsort(self.fit_xdata_orig)
        self.fit_xdata = self.fit_xdata[isort]
        self.fit_ydata = self.fit_ydata[isort]

        # Get best fit
        self.fit_type, self.fit_result = fit(self.fit_xdata, self.fit_ydata)
        self.predicted_ydata = self.predict(self.fit_xdata)

    # Get RMSE
    def get_rmse(self, get_net_km2_error):
        self.nrmse = np.sum((self.fit_ydata - self.predicted_ydata) ** 2) ** 0.5 / np.mean(self.fit_ydata)
        self.nrmse_adj = (
            np.sum((self.fit_ydata - self.predicted_adj_ydata) ** 2)
        ) ** 0.5 / np.mean(self.fit_ydata)

        # Get km2 error
        if get_net_km2_error is not None:
            self.km2_error = get_net_km2_error(self.predicted_ydata, self.fit_ydata)
            self.km2_error_adj = get_net_km2_error(self.predicted_adj_ydata, self.fit_ydata)

    def predict(self, xdata):
        return self.fit_result.eval(x=xdata)
