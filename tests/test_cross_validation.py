# -*- coding: utf-8 -*-
"""Morris/Sobol 交叉验证测试。"""
import os

import pandas as pd

from p2d_gsa.cross_validation import (
    cross_validate, plot_cross_validation, build_consensus,
    plot_consensus_atlas,
)


def _dfs(morris_values, sobol_values, params=("a", "b", "c", "d")):
    morris = pd.DataFrame({"parameter": list(params),
                           "mu_star": morris_values})
    sobol = pd.DataFrame({"parameter": list(params), "ST": sobol_values})
    return morris, sobol


def test_cross_validate_perfect_agreement():
    morris, sobol = _dfs([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1])
    result = cross_validate(morris, sobol, top_n=2)
    assert result["spearman_rho"] == 1.0
    assert result["kendall_tau"] == 1.0
    assert result["top_overlap_count"] == 2
    # 归一化后两方法完全一致，间隙近似为 0
    import numpy as np
    assert np.isclose(result["table"]["abs_norm_gap"].abs().max(), 0.0)


def test_cross_validate_partial_disagreement():
    # 排序部分不一致
    morris, sobol = _dfs([4, 3, 2, 1], [0.1, 0.4, 0.2, 0.3])
    result = cross_validate(morris, sobol, top_n=2)
    assert -1.0 < result["spearman_rho"] < 1.0
    assert 0 <= result["top_overlap_count"] <= 2
    assert set(result["table"]["parameter"]) == {"a", "b", "c", "d"}


def test_cross_validate_normalized_columns():
    morris, sobol = _dfs([40, 20, 10, 0], [0.8, 0.4, 0.2, 0.0])
    table = cross_validate(morris, sobol)["table"]
    assert table["mu_star_norm"].max() == 1.0
    assert table["st_norm"].max() == 1.0
    assert table["mu_star_norm"].min() == 0.0


def test_plot_cross_validation(tmp_path):
    morris, sobol = _dfs([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1])
    result = cross_validate(morris, sobol)
    out = os.path.join(tmp_path, "fig13.png")
    plot_cross_validation(result, "test_scenario", out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_build_consensus_ranking():
    morris, sobol = _dfs([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1])
    r1 = cross_validate(morris, sobol)
    # 两个工况（用不同的 Sobol 排序制造差异）
    morris2, sobol2 = _dfs([4, 3, 2, 1], [0.1, 0.4, 0.2, 0.3])
    r2 = cross_validate(morris2, sobol2)
    consensus = build_consensus({"sc1": r1, "sc2": r2})
    assert set(consensus["parameter"]) == {"a", "b", "c", "d"}
    # 按共识分降序
    scores = consensus["consensus_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert (consensus["consensus_score"] >= 0).all()


def test_plot_consensus_atlas(tmp_path):
    morris, sobol = _dfs([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1])
    r1 = cross_validate(morris, sobol)
    consensus = build_consensus({"sc1": r1})
    out = os.path.join(tmp_path, "fig14.png")
    plot_consensus_atlas({"sc1": r1}, consensus, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
