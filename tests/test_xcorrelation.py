# -*- coding: utf-8 -*-
"""Tests for the XCorrelation class."""

from __future__ import annotations

import numpy as np
import pytest

import dymoval as dmv
from dymoval.config import ATOL


# ============================================================
# XCorrelation
# ============================================================
class Test_XCorrelation:
    def test_initializer(self, correlation_tensors: tuple) -> None:
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            _,
            Rx1y1_expected,
            _,
            X,
            Y,
            _,
            _,
            _,
        ) = correlation_tensors

        lags_expected = np.arange(-5, 6)

        x0 = X[:, 0].T
        y0 = Y[:, 0].T

        # SISO
        R = dmv.XCorrelation("foo", x0, y0).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 0].lags, lags_expected)

        # SIMO
        R = dmv.XCorrelation("foo", x0, Y).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 1].values, Rx0y1_expected, atol=1e-3)

        # MISO
        R = dmv.XCorrelation("foo", X, y0).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)

        # MIMO
        Rxy = dmv.XCorrelation("foo", X, Y)
        R = Rxy.R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 1].values, Rx0y1_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 1].values, Rx1y1_expected, atol=1e-3)

        for ii in range(2):
            for jj in range(2):
                np.testing.assert_allclose(R[ii, jj].lags, lags_expected)

        assert Rxy.kind == "cross-correlation"
        assert dmv.XCorrelation("foo", X, X).kind == "auto-correlation"

    def test_initializer_with_bandwidth_args(
        self, correlation_tensors: tuple
    ) -> None:
        (
            Rx0y0_expected,
            Rx1y0_expected,
            _,
            Rx0y1_expected_partial,
            _,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        lags_short = np.arange(-1, 2)
        lags_long = np.arange(-5, 6)

        Rxy = dmv.XCorrelation(
            "foo", X, Y, None, X_bandwidths, Y_bandwidths, sampling_period
        )
        R = Rxy.R

        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(
            R[0, 1].values, Rx0y1_expected_partial, atol=1e-3
        )
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)
        np.testing.assert_allclose(
            R[1, 1].values, Rx1y1_expected_partial, atol=1e-3
        )

        np.testing.assert_allclose(R[0, 0].lags, lags_long)
        np.testing.assert_allclose(R[0, 1].lags, lags_short)
        np.testing.assert_allclose(R[1, 0].lags, lags_long)
        np.testing.assert_allclose(R[1, 1].lags, lags_short)

    def test_initializer_with_not_all_args_passed(
        self, correlation_tensors: tuple
    ) -> None:
        # Downsampling needs the two bandwidths *and* the sampling period.
        # Supplying only some of them used to be silently ignored, which
        # gave a non-downsampled correlation with no indication that the
        # arguments had no effect.
        (
            *_,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        partial_arguments = [
            {"X_bandwidths": X_bandwidths, "sampling_period": sampling_period},
            {"Y_bandwidths": Y_bandwidths, "sampling_period": sampling_period},
            {"X_bandwidths": X_bandwidths, "Y_bandwidths": Y_bandwidths},
            {"X_bandwidths": X_bandwidths},
        ]

        for kwargs in partial_arguments:
            with pytest.raises(ValueError, match="all of them or none"):
                dmv.XCorrelation("foo", X, Y, None, **kwargs)

        # The sampling period on its own means "no downsampling" and is fine
        dmv.XCorrelation("foo", X, Y, None, sampling_period=sampling_period)

    def test_initializer_with_no_downsampling_args(
        self, correlation_tensors: tuple
    ) -> None:
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            _,
            Rx1y1_expected,
            _,
            X,
            Y,
            *_,
        ) = correlation_tensors

        R = dmv.XCorrelation("foo", X, Y, None).R

        lags_long = np.arange(-5, 6)

        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 1].values, Rx0y1_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 1].values, Rx1y1_expected, atol=1e-3)

        for ii in range(2):
            for jj in range(2):
                np.testing.assert_allclose(R[ii, jj].lags, lags_long)

    def test_initializer_with_nlags_arg(
        self, correlation_tensors: tuple
    ) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]
        X_bandwidths = correlation_tensors[8]
        Y_bandwidths = correlation_tensors[9]
        sampling_period = correlation_tensors[10]

        R = dmv.XCorrelation(
            name="foo",
            X=X,
            Y=Y,
            nlags=np.array([[5, 3], [6, 4]]),
            X_bandwidths=X_bandwidths,
            Y_bandwidths=Y_bandwidths,
            sampling_period=sampling_period,
        ).R

        np.testing.assert_allclose(R[0, 0].lags, np.arange(-2, 3))
        np.testing.assert_allclose(R[0, 1].lags, np.arange(-1, 2))
        np.testing.assert_allclose(R[1, 0].lags, np.arange(-3, 4))
        np.testing.assert_allclose(R[1, 1].lags, np.arange(-1, 2))

    def test_initializer_with_wrong_params(
        self, correlation_tensors: tuple
    ) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]
        Y_bandwidths = correlation_tensors[9]
        sampling_period = correlation_tensors[10]

        with pytest.raises(IndexError):
            dmv.XCorrelation(
                "foo",
                X,
                Y,
                None,
                np.array([2]),
                Y_bandwidths,
                sampling_period,
            )

    def test_estimate_whiteness(self, correlation_tensors: tuple) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        RXY = dmv.XCorrelation("foo", X, Y)
        w, W = RXY.estimate_whiteness()

        global_weights = np.ones((2, 2))
        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        for ii in range(2):
            for jj in range(2):
                local_weights[ii, jj] = np.ones(11)

        w_weighted, W_weighted = RXY.estimate_whiteness(
            local_weights=local_weights, global_weights=global_weights
        )

        np.testing.assert_allclose(w, w_weighted)
        np.testing.assert_allclose(W, W_weighted)

    def test_estimate_whiteness_raise(
        self, correlation_tensors: tuple
    ) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        RXY = dmv.XCorrelation("foo", X, Y)

        with pytest.raises(ValueError):
            RXY.estimate_whiteness(local_statistic="potato")

        with pytest.raises(IndexError):
            RXY.estimate_whiteness(local_weights=np.array([1, 2]))

        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        local_weights[0, 0] = np.ones(11)
        local_weights[0, 1] = np.ones(3)
        local_weights[1, 0] = np.ones(13)
        local_weights[1, 1] = np.ones(6)

        with pytest.raises(IndexError):
            RXY.estimate_whiteness(local_weights=local_weights)

        with pytest.raises(IndexError):
            RXY.estimate_whiteness(global_weights=np.array([1, 2]))

    def test_repr(self, correlation_tensors: tuple) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        text = repr(dmv.XCorrelation("foo", X, Y))

        assert "foo" in text
        assert "cross-correlation" in text


# ============================================================
# whiteness_level
# ============================================================
class Test_whiteness_level:
    def test_whiteness_level(self) -> None:
        x1 = np.array([0.1419, -0.4218, 0.9157, -0.7922, 0.9595])

        whiteness_expected = 0.3579244755541881
        whiteness_matrix_expected = np.array([[0.3579244755541881]])

        whiteness_actual, whiteness_matrix_actual = dmv.whiteness_level(x1)

        assert np.isclose(whiteness_expected, whiteness_actual, atol=ATOL)
        np.testing.assert_allclose(
            whiteness_matrix_expected, whiteness_matrix_actual
        )
