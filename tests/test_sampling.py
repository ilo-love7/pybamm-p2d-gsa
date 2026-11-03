# -*- coding: utf-8 -*-
"""抽样设计与落盘测试（小样本）。"""
import os

import numpy as np

from p2d_gsa.parameters import build_problem
from p2d_gsa.sampling import (
    generate_morris, generate_saltelli, save_design, load_design,
)


def test_morris_shape():
    problem = build_problem()
    d = problem["num_vars"]
    cfg = {"num_trajectories": 6, "num_levels": 4, "optimal_trajectories": 4}
    x = generate_morris(problem, cfg, seed=1)
    # optimal 轨迹数 * (d+1)
    assert x.shape == (4 * (d + 1), d)


def test_saltelli_shape_and_bounds():
    problem = build_problem()
    d = problem["num_vars"]
    n = 16
    x = generate_saltelli(problem, {"base_n": n, "calc_second_order": False}, seed=1)
    # calc_second_order=False 时 N*(d+2)
    assert x.shape == (n * (d + 2), d)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = save_design("morris", x, str(tmp_path))
    assert os.path.exists(path)
    back = load_design("morris", str(tmp_path))
    assert np.allclose(back, x)
