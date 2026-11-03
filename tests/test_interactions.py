# -*- coding: utf-8 -*-
"""交互量化模块单元测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from p2d_gsa.interactions import (
    SOBOL_SOURCES, build_interaction_table, interaction_frame,
    main_output_interaction_matrix,
)


def make_sobol_df():
    return pd.DataFrame({
        "parameter": ["a", "b", "c"],
        "S1": [0.40, -0.05, 0.0],
        "S1_conf": [0.01, 0.01, 0.0],
        "ST": [0.60, 0.20, 0.0],
        "ST_conf": [0.02, 0.02, 0.0],
    })


def test_interaction_frame_clips_negative_s1():
    out = interaction_frame(make_sobol_df())
    # 负 S1 截断为 0
    assert out["S1_clipped"].tolist() == [0.40, 0.0, 0.0]
    # interaction = ST - clip(S1)
    assert out["interaction"].iloc[0] == pytest.approx(0.20)
    assert out["interaction"].iloc[1] == pytest.approx(0.20)
    # 交互占比
    assert out["interaction_share"].iloc[0] == pytest.approx(0.20 / 0.60)
    # ST=0 时占比为 0
    assert out["interaction_share"].iloc[2] == 0.0
    assert (out["interaction"] >= 0).all()


def test_build_interaction_table(tmp_path):
    # 写入与 SOBOL_SOURCES 同名的假 CSV
    for _, _, fn in SOBOL_SOURCES:
        make_sobol_df().to_csv(tmp_path / fn, index=False)
    table = build_interaction_table(str(tmp_path))
    # 4 个来源 × 3 参数
    assert len(table) == len(SOBOL_SOURCES) * 3
    assert set(table["scenario"]) == {"s1_discharge_1c",
                                      "s2_fastcharge_2c", "s3_cold_charge"}
    expected_cols = {"scenario", "output", "parameter", "S1_clipped", "ST",
                     "interaction", "interaction_share"}
    assert expected_cols.issubset(table.columns)


def test_main_output_matrix(tmp_path):
    for _, _, fn in SOBOL_SOURCES:
        make_sobol_df().to_csv(tmp_path / fn, index=False)
    table = build_interaction_table(str(tmp_path))
    matrix = main_output_interaction_matrix(table)
    # 3 个工况列，参数为行
    assert list(matrix.columns) == ["s1_discharge_1c", "s2_fastcharge_2c",
                                   "s3_cold_charge"]
    assert list(matrix.index) == ["a", "b", "c"]
    assert np.isfinite(matrix.to_numpy()).all()
