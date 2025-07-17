"""
Module for various fitting functionality
"""

import numpy as np
from lmfit import models
from lmfit.model import ModelResult

VERBOSE = False


class LognormalFitParams:
    """
    Holds parameters for lognormal fitting.
    Args:
        center (float): Center of the lognormal distribution.
        sigma (float): Standard deviation.
        amplitude (float): Amplitude of the curve.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, center: float = 3.5, sigma: float = 1, amplitude: float = 6):
        self.center: float = center
        self.sigma: float = sigma
        self.amplitude: float = amplitude


def _fit_lognormal(
    xdata: np.ndarray, ydata: np.ndarray, params: LognormalFitParams
) -> ModelResult:
    """
    Fit a lognormal model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
        params (LognormalFitParams): Lognormal fit parameters.
    Returns:
        ModelResult: Fitted lognormal model result.
    """
    model = models.LognormalModel()
    params = model.make_params(
        center=params.center, sigma=params.sigma, amplitude=params.amplitude
    )
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Lognormal:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_exponential(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit an exponential model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted exponential model result.
    """
    model = models.ExponentialModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Exponential:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_gaussian(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit a Gaussian model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted Gaussian model result.
    """
    model = models.GaussianModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Gaussian:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_skewgaussian(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit a skewed Gaussian model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted skewed Gaussian model result.
    """
    model = models.SkewedGaussianModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Skewed Gaussian:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_logistic(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit a logistic step model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted logistic model result.
    """
    model = models.StepModel(form="logistic")
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Logistic:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_linear(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit a linear model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted linear model result.
    """
    model = models.LinearModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Linear:")
        print(result.fit_report())
        print(" ")
    return result


def _fit_quadratic(xdata: np.ndarray, ydata: np.ndarray) -> ModelResult:
    """
    Fit a quadratic model to the data.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
    Returns:
        ModelResult: Fitted quadratic model result.
    """
    model = models.QuadraticModel()
    params = model.guess(ydata, x=xdata, seed=1987)
    result = model.fit(ydata, params, x=xdata)

    if VERBOSE:
        print("Quadratic:")
        print(result.fit_report())
        print(" ")
    return result


def fit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    lognormal_params: LognormalFitParams = LognormalFitParams(),
) -> tuple[str, ModelResult]:
    """
    Fit multiple models to X and Y data and select the best by AIC.
    Args:
        xdata (np.ndarray): X values.
        ydata (np.ndarray): Y values.
        lognormal_params (LognormalFitParams): Lognormal fit parameters.
    Returns:
        tuple: (best fit name, best fit ModelResult)
    """

    if np.any(np.isnan(xdata)):
        raise RuntimeError("Unexpected NaN in xdata")
    if np.any(np.isnan(ydata)):
        raise RuntimeError("Unexpected NaN in ydata")

    # Try all fits
    results: dict[str, ModelResult] = {}
    results["lognormal"] = _fit_lognormal(xdata, ydata, lognormal_params)
    results["logistic"] = _fit_logistic(xdata, ydata)
    results["exponential"] = _fit_exponential(xdata, ydata)
    results["linear"] = _fit_linear(xdata, ydata)
    results["quadratic"] = _fit_quadratic(xdata, ydata)
    results["gaussian"] = _fit_gaussian(xdata, ydata)
    results["skewgaussian"] = _fit_skewgaussian(xdata, ydata)

    # Find best fit
    best_fit: str | None = None
    best_result: ModelResult | None = None
    best_metric: float = np.inf
    for this_fit, result in results.items():
        if VERBOSE:
            print(f"   {this_fit}: r2 {result.rsquared:.3f} (AIC {result.aic:.1f})")
        if result.aic < best_metric:
            best_metric = result.aic
            best_fit = this_fit
            best_result = result

    return best_fit, best_result
