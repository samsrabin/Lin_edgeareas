import numpy as np
from lmfit import models

def adjust_predicted_fits(ydata_yb):
    if ydata_yb.ndim ==1 or ydata_yb.shape[1] == 1:
        raise RuntimeError("It only makes sense to call adjust_predicted_fits() with multiple bins!")
    ydata_yb[ydata_yb < 0] = 0
    axis = len(ydata_yb.shape) - 1
    ydata_yb = ydata_yb / np.sum(ydata_yb, axis=axis, keepdims=True)
    return ydata_yb


class LognormalFitParams:
    # pylint: disable=too-few-public-methods
    def __init__(self, center=3.5, sigma=1, amplitude=6):
        self.center = center
        self.sigma = sigma
        self.amplitude = amplitude


def _fit_lognormal(xdata, ydata, params):
    model = models.LognormalModel()
    params = model.make_params(
        center=params.center, sigma=params.sigma, amplitude=params.amplitude
    )
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
    # results["lognormal"] = _fit_lognormal(xdata, ydata, lognormal_params)
    # results["logistic"] = _fit_logistic(xdata, ydata)
    # results["exponential"] = _fit_exponential(xdata, ydata)
    results["linear"] = _fit_linear(xdata, ydata)
    results["quadratic"] = _fit_quadratic(xdata, ydata)
    # results["gaussian"] = _fit_gaussian(xdata, ydata)
    # results["skewgaussian"] = _fit_skewgaussian(xdata, ydata)

    # Find best fit
    best_fit = None
    best_result = None
    best_metric = np.inf
    for i, (this_fit, result) in enumerate(
        results.items()
    ):  # pylint: disable=unused-variable
        if result.aic < best_metric:
            best_metric = result.aic
            best_fit = this_fit
            best_result = result

    return best_fit, best_result
