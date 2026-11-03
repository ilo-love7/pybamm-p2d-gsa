# -*- coding: utf-8 -*-
"""参数空间定义与映射测试。"""
import numpy as np

from p2d_gsa.parameters import (
    PARAM_SPECS, FACTOR_OVERRIDES, get_factors, build_problem, denormalize,
)

PERTURB = {"lower_factor": 0.8, "upper_factor": 1.2}


def test_param_specs_count_and_unique():
    assert len(PARAM_SPECS) == 24
    shorts = [s.short for s in PARAM_SPECS]
    assert len(set(shorts)) == 24
    keys = [s.pybamm_key for s in PARAM_SPECS]
    assert len(set(keys)) == 24


def test_get_factors_override():
    lo, hi = get_factors("c0_neg", PERTURB)
    assert (lo, hi) == FACTOR_OVERRIDES["c0_neg"]
    assert hi <= 1.10
    lo2, hi2 = get_factors("neg_thickness", PERTURB)
    assert (lo2, hi2) == (0.8, 1.2)


def test_build_problem():
    p = build_problem()
    assert p["num_vars"] == 24
    assert len(p["names"]) == 24 and len(p["bounds"]) == 24
    assert all(b == [0.0, 1.0] for b in p["bounds"])


def test_denormalize_endpoints():
    baselines = {s.short: 100.0 for s in PARAM_SPECS}
    x0 = np.zeros(24)
    x1 = np.ones(24)
    a0 = denormalize(x0, baselines, PERTURB)
    a1 = denormalize(x1, baselines, PERTURB)
    for spec in PARAM_SPECS:
        if spec.short == "c0_neg":
            assert np.isclose(a0[spec.pybamm_key], 80.0)
            assert np.isclose(a1[spec.pybamm_key], 110.0)
        else:
            assert np.isclose(a0[spec.pybamm_key], 80.0)
            assert np.isclose(a1[spec.pybamm_key], 120.0)
