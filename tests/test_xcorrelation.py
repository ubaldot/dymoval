# -*- coding: utf-8 -*-
"""Tests for the XCorrelation class."""

from __future__ import annotations

import numpy as np
import pytest

import dymoval as dmv


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
        R = dmv.XCorrelation("foo", x0, y0, np.full((1, 1), 11)).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 0].lags, lags_expected)

        # SIMO
        R = dmv.XCorrelation("foo", x0, Y, np.full((1, 2), 11)).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 1].values, Rx0y1_expected, atol=1e-3)

        # MISO
        R = dmv.XCorrelation("foo", X, y0, np.full((2, 1), 11)).R
        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)

        # MIMO
        Rxy = dmv.XCorrelation("foo", X, Y, np.full((2, 2), 11))
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

        R = dmv.XCorrelation(
            "foo", X, Y, np.full((2, 2), 11, dtype=int)
        ).R

        lags_long = np.arange(-5, 6)

        np.testing.assert_allclose(R[0, 0].values, Rx0y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[0, 1].values, Rx0y1_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 0].values, Rx1y0_expected, atol=1e-3)
        np.testing.assert_allclose(R[1, 1].values, Rx1y1_expected, atol=1e-3)

        for ii in range(2):
            for jj in range(2):
                np.testing.assert_allclose(R[ii, jj].lags, lags_long)

    def test_default_lag_window(self) -> None:
        values = np.random.default_rng(42).normal(size=(100, 2))
        R = dmv.XCorrelation("default", values, values).R
        for ii in range(2):
            for jj in range(2):
                np.testing.assert_array_equal(
                    R[ii, jj].lags, np.arange(-20, 21)
                )

    def test_initializer_with_nlags_arg(
        self, correlation_tensors: tuple
    ) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        R = dmv.XCorrelation(
            name="foo",
            X=X,
            Y=Y,
            nlags=np.array([[5, 3], [6, 4]]),
        ).R

        np.testing.assert_allclose(R[0, 0].lags, np.arange(-2, 3))
        np.testing.assert_allclose(R[0, 1].lags, np.arange(-1, 2))
        np.testing.assert_allclose(R[1, 0].lags, np.arange(-3, 4))
        np.testing.assert_allclose(R[1, 1].lags, np.arange(-2, 3))

    @pytest.mark.parametrize(
        ("nlags", "error"),
        [
            (np.array([[0, 3], [5, 7]]), ValueError),
            (np.array([[-1, 3], [5, 7]]), ValueError),
            (np.array([[3.0, 3.0], [5.0, 7.0]]), TypeError),
        ],
    )
    def test_nlags_must_be_positive_integers(
        self,
        correlation_tensors: tuple,
        nlags: np.ndarray,
        error: type[Exception],
    ) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        with pytest.raises(error):
            dmv.XCorrelation("foo", X, Y, nlags=nlags)

    def test_correlation_score(self, correlation_tensors: tuple) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        RXY = dmv.XCorrelation("foo", X, Y)
        w, W = RXY.correlation_score()

        global_weights = np.ones((2, 2))
        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        for ii in range(2):
            for jj in range(2):
                local_weights[ii, jj] = np.ones(len(RXY.R[ii, jj].lags))

        w_weighted, W_weighted = RXY.correlation_score(
            local_weights=local_weights, global_weights=global_weights
        )

        np.testing.assert_allclose(w, w_weighted)
        np.testing.assert_allclose(W, W_weighted)

    def test_correlation_score_raise(self, correlation_tensors: tuple) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        RXY = dmv.XCorrelation("foo", X, Y)

        with pytest.raises(ValueError):
            RXY.correlation_score(local_statistic="potato")

        with pytest.raises(IndexError):
            RXY.correlation_score(local_weights=np.array([1, 2]))

        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        local_weights[0, 0] = np.ones(11)
        local_weights[0, 1] = np.ones(3)
        local_weights[1, 0] = np.ones(13)
        local_weights[1, 1] = np.ones(6)

        with pytest.raises(IndexError):
            RXY.correlation_score(local_weights=local_weights)

        with pytest.raises(IndexError):
            RXY.correlation_score(global_weights=np.array([1, 2]))

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

        assert np.isclose(whiteness_expected, whiteness_actual, atol=1e-9)
        np.testing.assert_allclose(
            whiteness_matrix_expected, whiteness_matrix_actual
        )
