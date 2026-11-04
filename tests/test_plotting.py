# -*- coding: utf-8 -*-
"""plotting 绘图模块单元测试（Agg 后端，验证文件生成）。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from p2d_gsa import plotting


def _morris_df(n=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "parameter": [f"p{i}" for i in range(n)],
        "mu_star": np.sort(rng.uniform(0, 1, n)),
        "mu_star_conf": rng.uniform(0, 0.1, n),
        "sigma": rng.uniform(0, 1, n),
    })


def _sobol_df(n=10):
    rng = np.random.default_rng(1)
    st = np.sort(rng.uniform(0, 1, n))
    return pd.DataFrame({
        "parameter": [f"p{i}" for i in range(n)],
        "S1": st * 0.7,
        "S1_conf": rng.uniform(0, 0.1, n),
        "ST": st,
        "ST_conf": rng.uniform(0, 0.1, n),
    })


def _conv_df():
    rows = []
    for p in ("p0", "p1", "p2"):
        for n in (32, 64, 128, 256):
            rows.append({"parameter": p, "N": n, "fraction": 1.0,
                         "ST": 1.0 / n + 0.1})
    return pd.DataFrame(rows)


def test_all_plot_functions(tmp_path):
    out = str(tmp_path)
    m = _morris_df()
    s = _sobol_df()

    p1 = plotting.plot_morris_bar(m, "sc", out)
    assert os.path.exists(p1)
    p2 = plotting.plot_morris_scatter(m, "sc", out)
    assert os.path.exists(p2)
    p3 = plotting.plot_sobol_st(s, "sc", out)
    assert os.path.exists(p3)
    p4 = plotting.plot_sobol_s1_vs_st(s, "sc", out)
    assert os.path.exists(p4)
    p5 = plotting.plot_convergence(_conv_df(), "sc", out)
    assert os.path.exists(p5)

    st_matrix = pd.DataFrame(
        np.random.default_rng(2).uniform(0, 1, (8, 3)),
        columns=["s1", "s2", "s3"],
        index=[f"p{i}" for i in range(8)])
    p6 = plotting.plot_st_heatmap(st_matrix, out)
    assert os.path.exists(p6)

    whitelist = s[["parameter", "ST", "ST_conf"]].head(6)
    p7 = plotting.plot_whitelist(whitelist, 0.05, out)
    assert os.path.exists(p7)
