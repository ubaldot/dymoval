"""Statistical indices used for model validation.

Pure numpy helpers: they know nothing about :class:`dymoval.signal.Signal`,
:class:`dymoval.dataset.Dataset` or
:class:`dymoval.validation.ValidationSession`.
"""

from __future__ import annotations

import numpy as np

from .config import XCORR_STATISTIC_TYPE, XCorr_Statistic_type

__all__ = ["compute_statistic", "rsquared"]


def compute_statistic(
    data: np.ndarray,
    statistic: XCorr_Statistic_type = "mean",
    weights: np.ndarray | None = None,
) -> float:
    r"""Compute the statistic of a sequence of numbers.

    The elements of `data` can be weighted through the `weights` array.

    If `data.shape` dimension is greater than 1 then `data` will be flatten
    to a 1-D array.
    The  return values are normalized such that the function always return
    values between 0 and 1, with the exclusion of the statistic `quadratic`
    that may return values greater than 1.0.

    The statistic `S` is computed as it follows. Let :math:`w_i` is the
    `i`-th element of `weights` and :math:`x_i` is the `i`-th element
    of `data`.

    **mean**
    This is the classic weighted mean value, computed as:

    .. math::
        S = \frac{\sum_{i=1}^N w_ix_i}{\sum_{i=1}^N w_i}

    **abs_mean**
    Mean of absolute values, computed as:

    .. math::
        S = \frac{\sum_{i=1}^N w_i|x_i|}{\sum_{i=1}^N w_i}

    **max**
    Max of absolute values, computed as:

    .. math::
        S = \max_i\{|x_i|\}

    **std**
    Standard deviation, computed as:

    .. math::
        S =\sqrt{\sum_{i=1}^N w_i(x_i - \bar x)^2}

    where :math:`\bar x` is the weighted mean value computed above.

    **quadratic**
    This is a generic quadratic form of the form:

    .. math::
        S = \frac{1}{N \|W\|_{\infty}}x^TWx
        = \frac{\sum_{i=1}^N w_i|x_i|}{N\max_i \{|w_i|\}}


    This is particular useful since many famous statistics, such as Ljung-Box,
    Box-Pierce, Lagrange Multiplier, etc., can be rewritten in the above form
    through an appropriate choice of the weights.

    Parameters
    ----------
    data:
        Array containing values for which the `statistic` shall be computed.
    statistic:
        Kind of statistic to be computed.
    weights:
        An array of weights associated with the values in `data`.
        More precisely, `weights[i]` correspond to `data[i]`.
    """

    if data.ndim > 1:
        raise IndexError("'data' must be a 1-D np.ndarray.")

    if weights is None:
        weights = np.ones(data.size)
    else:
        # These are separate checks on purpose: chaining them with `elif`
        # means only the first one ever runs, so e.g. a 2-D `weights` with
        # no negative entry would sail through.
        if weights.ndim > 1:
            raise IndexError("'weights' must be a 1-D np.ndarray.")

        if data.size != weights.size:
            raise IndexError("'data' and 'weights' must have the same length.")

        if not np.all(np.isfinite(weights)):
            raise ValueError("All weights must be finite.")

        if np.min(weights) < 0:
            raise ValueError("All weights must be positive.")

        # Every statistic below divides by either the sum or the largest of
        # the weights, so an all-zero vector silently yields nan or inf.
        if np.max(weights) <= 0.0:
            raise ValueError("At least one weight must be strictly positive.")

    if statistic == "quadratic":
        # It holds x'Wx < lambda_max(W) |x|^2 = max(W) given that W is
        # diagonal with all positive entries. Furthermore, it
        # hold np.max(weights) = lambda_max(W).
        # To avoid that the metric depends on the number of lags, we normalize
        # by n.

        quadratic_form = data.T @ np.diag(weights) @ data

        result = quadratic_form / (np.max(np.abs(weights)) * len(data))
        # If the weights are all the same, then the metric reduces to |x|²/n,
        # which is similar to the abs_norm with all the weights equal to 1.

    elif statistic == "abs_mean":
        # This is similar to RMS. By using l1 norm we are more gentle with
        # respect to outliers.
        result = np.sum(weights * np.abs(data)) / np.sum(weights)

    elif statistic == "max":
        # To secure <1 you have to divide by |x|_inf|W|_inf
        result = np.max(np.abs(weights.T * data)) / (
            np.linalg.norm(weights, ord=np.inf)
        )
    elif statistic == "mean":
        # This should always be less than 1
        result = weights.T @ data / np.sum(weights)
    elif statistic == "std":
        # Compute the weighted average
        weighted_avg = weights.T @ data / np.sum(weights)
        # Compute the weighted variance
        weighted_variance = np.average(
            (data - weighted_avg) ** 2, weights=weights
        )
        result = np.sqrt(weighted_variance)  # Standard deviation
    else:
        raise ValueError(
            f"'statistic' must be one of [{XCORR_STATISTIC_TYPE}]"
        )
    return float(result)


def rsquared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Return the :math:`R^2` value of two signals.

    Signals can be MIMO.

    Parameters
    ----------
    x:
        First input signal. It must have shape :math:`N\times p`, where :math:`N` is the number
        of observation and :math:`p` is the signal dimension.
    y:
        Second input signal. It must have shape :math:`N\times p`, where :math:`N` is the number
        of observation and :math:`p` is the signal dimension.
    """

    if x.shape != y.shape:
        raise IndexError("Arguments must have the same shape.")
    eps = x - y
    # Compute r-square fit (%)
    x_mean = np.mean(x, axis=0)

    # Compute the R² index
    ss_res = np.sum(eps**2, axis=0)
    ss_tot = np.sum((x - x_mean) ** 2, axis=0)

    if np.any(ss_tot == 0.0):
        # R² measures how much of the variance of `x` the signal `y`
        # explains. A constant `x` has none, so the ratio is 0/0 and the
        # result would be nan or -inf depending on the residuals.
        raise ValueError(
            "R-squared is undefined for a constant reference signal: "
            "it has no variance to explain."
        )

    r2: np.ndarray = np.asarray((1.0 - ss_res / ss_tot) * 100)

    return r2
