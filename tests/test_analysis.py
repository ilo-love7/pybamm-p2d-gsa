# -*- coding: utf-8 -*-
"""分析模块纯函数测试。"""
import numpy as np
import pandas as pd

from p2d_gsa.analysis import _fill_nan, _saltelli_subY, build_whitelist

CFG = {"statistics": {"whitelist_st_threshold": 0.05}}


def test_fill_nan_replaces_with_median():
    y = np.array([1.0, 3.0, np.nan, 5.0])
    out, fill = _fill_nan(y, None)
    assert np.isfinite(out).all()
    assert np.isclose(fill, 3.0)
    assert np.isclose(out[2], 3.0)


def test_fill_nan_all_bad():
    y = np.array([np.nan, np.nan])
    out, fill = _fill_nan(y, None)
    assert np.allclose(out, 0.0)


def test_saltelli_subY_block_slicing():
    d = 3
    step = d + 2
    n_sub = 2
    y = np.arange(n_sub * step, dtype=float)
    sub = _saltelli_subY(y, d, 100, n_sub)
    # 取出前两个完整块，共 2*5=10 行
    assert len(sub) == n_sub * step
    assert np.allclose(sub, y[: n_sub * step])


def test_build_whitelist_threshold():
    df = pd.DataFrame({
        "parameter": ["a", "b", "c"],
        "ST": [0.01, 0.2, 0.04],
    })
    wl = build_whitelist(df, CFG)
    assert list(wl["parameter"]) == ["a", "c"]
    assert (wl["ST"] < 0.05).all()
