# -*- coding: utf-8 -*-
"""PyBaMM DFN + 集总热模型封装与单次仿真。"""
from __future__ import annotations

import pybamm

from .parameters import PARAM_SPECS, denormalize


def build_model():
    """构建 DFN + lumped thermal 模型（每个进程构建一次）。"""
    return pybamm.lithium_ion.DFN(options={"thermal": "lumped"})


def _aggregate(values, how):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if how == "last":
        return float(arr[-1])
    if how == "max":
        return float(np.nanmax(arr))
    if how == "min":
        return float(np.nanmin(arr))
    raise ValueError(f"未知聚合方式 {how}")


def run_one(x_unit, scenario_cfg, baselines, perturb_cfg, model=None,
            parameter_set="Chen2020"):
    """
    运行一次仿真。

    Parameters
    ----------
    x_unit : 归一化参数向量 [0,1]
    scenario_cfg : 该工况配置（experiment/outputs/aggregations）
    baselines : 参数基线
    perturb_cfg : 扰动上下界因子

    Returns
    -------
    dict 指标名 -> 值；失败抛异常由调用方捕获
    """
    if model is None:
        model = build_model()

    actual = denormalize(x_unit, baselines, perturb_cfg)

    pvals = pybamm.ParameterValues(parameter_set)
    pvals.update(actual)
    # 初始温度跟随环境温度，保证热初始条件一致
    pvals["Initial temperature [K]"] = actual[
        next(s.pybamm_key for s in PARAM_SPECS if s.short == "ambient_t")
    ]

    experiment = pybamm.Experiment([scenario_cfg["experiment"]])
    sim = pybamm.Simulation(model, parameter_values=pvals, experiment=experiment)
    sol = sim.solve(calc_esoh=False)

    results = {}
    for out_name, var_name in scenario_cfg["outputs"].items():
        arr = sol[var_name].entries
        how = scenario_cfg["aggregations"][out_name]
        results[out_name] = _aggregate(arr, how)
    return results
