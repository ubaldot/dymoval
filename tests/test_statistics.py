# -*- coding: utf-8 -*-
"""Tests for the validation statistics."""

from __future__ import annotations

import numpy as np
import pytest

import dymoval as dmv
from dymoval.config import ATOL
from dymoval.statistics import compute_statistic


# ============================================================
# rsquared
# ============================================================
class Test_rsquared:
    def test_rsquared_nominal(self) -> None:
        y1 = np.array(
            [
                0,
                0.5878,
                0.9511,
                0.9511,
                0.5878,
                0.0000,
                -0.5878,
                -0.9511,
                -0.9511,
                -0.5878,
                -0.0000,
            ]
        )
        y2 = np.array(
            [
                0,
                0.7053,
                1.1413,
                1.1413,
                0.7053,
                0.0000,
                -0.7053,
                -1.1413,
                -1.1413,
                -0.7053,
                -0.0000,
            ]
        )
        y1Calc = np.array(
            [
                0.1403,
                0.8620,
                1.0687,
                1.1633,
                0.9208,
                0.2390,
                -0.4537,
                -0.8314,
                -0.7700,
                -0.4187,
                0.1438,
            ]
        )
        y2Calc = np.array(
            [
                0.2233,
                1.0024,
                1.3110,
                1.3130,
                0.7553,
                0.0098,
                -0.5893,
                -1.0143,
                -0.8798,
                -0.3226,
                0.3743,
            ]
        )

        rsquared_expected_SISO = 91.2775
        rsquared_expected_MIMO = np.array([91.27830428, 91.89543218])

        assert np.isclose(
            rsquared_expected_SISO, dmv.rsquared(y1, y1Calc), atol=ATOL
        )
        np.testing.assert_allclose(
            rsquared_expected_MIMO,
            dmv.rsquared(np.array([y1, y2]).T, np.array([y1Calc, y2Calc]).T),
            atol=ATOL,
        )

    @pytest.mark.parametrize(
        "shape_x,shape_y",
        [
            ((10,), (5,)),
            ((8, 3), (10,)),
            ((10,), (10, 3)),
            ((5, 1), (10, 4)),
            ((8, 3), (4, 4)),
            ((10, 4), (15, 3)),
        ],
    )
    def test_rsquared_raise(self, shape_x: tuple, shape_y: tuple) -> None:
        rng = np.random.default_rng(17)

        with pytest.raises(IndexError):
            dmv.rsquared(rng.random(shape_x), rng.random(shape_y))


# ============================================================
# compute_statistic
# ============================================================
class Test_compute_statistic:
    test_data = np.array(
        [
            0.49065828,
            0.1754277,
            -0.37027646,
            0.26591682,
            -0.62597191,
            0.89125522,
            -0.14112183,
            -0.16938656,
            0.31309603,
            0.2876763,
        ]
    )

    @pytest.mark.parametrize(
        "statistic,expected",
        [
            ("mean", 0.11172735900000001),
            ("abs_mean", 0.37307871099999995),
            ("max", 0.89125522),
            ("quadratic", 0.18949074921298226),
            ("std", 0.420722885595575),
        ],
    )
    def test_unweighted(self, statistic: str, expected: float) -> None:
        assert np.isclose(
            compute_statistic(data=self.test_data, statistic=statistic),
            expected,
        )

    @pytest.mark.parametrize(
        "statistic,expected",
        [
            ("mean", 0.10053961584752258),
            ("abs_mean", 0.5967800071293234),
            ("max", 0.89125522),
            ("quadratic", 0.1228105035864878),
            ("std", 0.6498192687767278),
        ],
    )
    def test_weighted(self, statistic: str, expected: float) -> None:
        # Gaussian shaped weights
        x = np.arange(10)
        weights = np.exp(-((x - 4.5) ** 2) / 2)

        assert np.isclose(
            compute_statistic(
                data=self.test_data, statistic=statistic, weights=weights
            ),
            expected,
        )

    def test_raise(self) -> None:
        X = np.ones(10)

        with pytest.raises(IndexError):
            compute_statistic(np.ones((2, 2)))

        with pytest.raises(IndexError):
            compute_statistic(X, weights=np.ones(15))

        with pytest.raises(IndexError):
            compute_statistic(X, weights=np.ones((2, 2)))

        with pytest.raises(ValueError):
            compute_statistic(X, weights=-np.ones((2, 2)))

        with pytest.raises(ValueError):
            compute_statistic(X, statistic="quadraticcccc")
