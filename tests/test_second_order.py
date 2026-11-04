# -*- coding: utf-8 -*-
"""二阶 S2 分析测试。"""
import os

import numpy as np
from SALib.sample import sobol as sobol_sample

from p2d_gsa import second_order
from p2d_gsa.second_order import (
    analyze_s2, s2_pair_table, plot_s2_heatmap, plot_s2_top_pairs,
)

SMALL_PROBLEM = {
    "num_vars": 3, "names": ["a", "b", "c"],
    "bounds": [[0.0, 1.0]] * 3,
}
SMALL_CFG = {"statistics": {"bootstrap_n": 100, "confidence": 0.95}}


def _make_design_and_y():
    x = sobol_sample.sample(SMALL_PROBLEM, 8, calc_second_order=True)
    # 含 a*b 交互的简单函数
    y = x[:, 0] + 0.5 * x[:, 0] * x[:, 1] + 0.2 * x[:, 2]
    return x, y


def test_analyze_s2_shapes():
    _, y = _make_design_and_y()
    res = analyze_s2(SMALL_PROBLEM, y, SMALL_CFG)
    assert np.asarray(res["S2"]).shape == (3, 3)
    assert np.asarray(res["S2_conf"]).shape == (3, 3)


def test_s2_pair_table(monkeypatch):
    monkeypatch.setattr(second_order, "SHORT_NAMES", ["a", "b", "c"])
    _, y = _make_design_and_y()
    res = analyze_s2(SMALL_PROBLEM, y, SMALL_CFG)
    table = s2_pair_table(res)
    # 3 个参数对
    assert len(table) == 3
    assert set(table.columns) == {"param_i", "param_j", "S2", "S2_conf"}
    # 降序排列
    assert table["S2"].tolist() == sorted(table["S2"], reverse=True)
    # a×b 交互最强
    assert table.iloc[0]["param_i"] == "a"
    assert table.iloc[0]["param_j"] == "b"


def test_s2_pair_table_top_n(monkeypatch):
    monkeypatch.setattr(second_order, "SHORT_NAMES", ["a", "b", "c"])
    _, y = _make_design_and_y()
    res = analyze_s2(SMALL_PROBLEM, y, SMALL_CFG)
    assert len(s2_pair_table(res, top_n=2)) == 2


def test_s2_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(second_order, "SHORT_NAMES", ["a", "b", "c"])
    _, y = _make_design_and_y()
    res = analyze_s2(SMALL_PROBLEM, y, SMALL_CFG)
    table = s2_pair_table(res)

    hm = os.path.join(tmp_path, "hm.png")
    bp = os.path.join(tmp_path, "bp.png")
    plot_s2_heatmap(res, "sc", "out", hm)
    plot_s2_top_pairs(table, "sc", "out", bp)
    assert os.path.getsize(hm) > 0
    assert os.path.getsize(bp) > 0
