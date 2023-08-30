"""Test feature extraction with light_curve package"""

import light_curve as licu
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from tape import Ensemble
from tape.analysis.feature_extractor import FeatureExtractor
from tape.utils import ColumnMapper


def test_stetsonk():
    stetson_k = licu.StetsonK()

    time = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0] * 2)
    flux = 1.0 + time**2.0
    err = np.full_like(time, 0.1, dtype=np.float32)
    band = np.r_[["g"] * 6, ["r"] * 6]

    extract_features = FeatureExtractor(stetson_k)
    result = extract_features(time=time, flux=flux, err=err, band=band, band_to_calc="g")
    assert result.shape == (1,)
    assert_array_equal(result.index, ["stetson_K"])
    assert_allclose(result.values, 0.84932, rtol=1e-5)
    assert_array_equal(result.dtypes, np.float64)


def test_stetsonk_with_ensemble():
    n = 5

    object1 = {
        "id": np.full(n, 1),
        "time": np.arange(n, dtype=np.float64),
        "flux": np.linspace(1.0, 2.0, n),
        "err": np.full(n, 0.1),
        "band": np.full(n, "g"),
    }
    object2 = {
        "id": np.full(2 * n, 2),
        "time": np.arange(2 * n, dtype=np.float64),
        "flux": np.r_[np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n)],
        "err": np.full(2 * n, 0.01),
        "band": np.r_[np.full(n, "g"), np.full(n, "r")],
    }
    rows = {column: np.concatenate([object1[column], object2[column]]) for column in object1}

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens = Ensemble(client=False).from_source_dict(rows, cmap)

    stetson_k = licu.Extractor(licu.AndersonDarlingNormal(), licu.InterPercentileRange(0.25), licu.StetsonK())
    result = ens.batch(
        stetson_k,
        band_to_calc="g",
    )

    assert result.shape == (2, 3)
    assert_array_equal(result.columns, ["anderson_darling_normal", "inter_percentile_range_25", "stetson_K"])
    assert_allclose(result, [[0.114875, 0.625, 0.848528]] * 2, atol=1e-5)
