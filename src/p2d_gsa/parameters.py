# -*- coding: utf-8 -*-
"""参数空间定义：24 个跨域标量参数，统一乘法扰动。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pybamm

if TYPE_CHECKING:
    import pandas


@dataclass(frozen=True)
class ParamSpec:
    short: str
    pybamm_key: str
    category: str


# 24 个标量参数，覆盖几何/孔隙/输运/动力学/热/初始浓度六大类
PARAM_SPECS = [
    # --- 几何 ---
    ParamSpec("neg_thickness", "Negative electrode thickness [m]", "geometry"),
    ParamSpec("pos_thickness", "Positive electrode thickness [m]", "geometry"),
    ParamSpec("sep_thickness", "Separator thickness [m]", "geometry"),
    ParamSpec("neg_particle_r", "Negative particle radius [m]", "geometry"),
    ParamSpec("pos_particle_r", "Positive particle radius [m]", "geometry"),
    ParamSpec("electrode_height", "Electrode height [m]", "geometry"),
    # --- 孔隙 / 体积分数 ---
    ParamSpec("neg_amf", "Negative electrode active material volume fraction", "porosity"),
    ParamSpec("pos_amf", "Positive electrode active material volume fraction", "porosity"),
    ParamSpec("neg_porosity", "Negative electrode porosity", "porosity"),
    ParamSpec("sep_porosity", "Separator porosity", "porosity"),
    # --- 输运 ---
    ParamSpec("neg_diffusivity", "Negative particle diffusivity [m2.s-1]", "transport"),
    ParamSpec("pos_diffusivity", "Positive particle diffusivity [m2.s-1]", "transport"),
    ParamSpec("neg_conductivity", "Negative electrode conductivity [S.m-1]", "transport"),
    ParamSpec("transference", "Cation transference number", "transport"),
    ParamSpec("neg_bruggeman", "Negative electrode Bruggeman coefficient (electrolyte)", "transport"),
    # --- 动力学 ---
    ParamSpec("neg_alpha", "Negative electrode charge transfer coefficient", "kinetics"),
    ParamSpec("pos_alpha", "Positive electrode charge transfer coefficient", "kinetics"),
    # --- 热 ---
    ParamSpec("h_transfer", "Total heat transfer coefficient [W.m-2.K-1]", "thermal"),
    ParamSpec("neg_cp", "Negative electrode specific heat capacity [J.kg-1.K-1]", "thermal"),
    ParamSpec("neg_thermal_k", "Negative electrode thermal conductivity [W.m-1.K-1]", "thermal"),
    ParamSpec("cooling_area", "Cell cooling surface area [m2]", "thermal"),
    ParamSpec("ambient_t", "Ambient temperature [K]", "thermal"),
    # --- 初始浓度 ---
    ParamSpec("c0_elyte", "Initial concentration in electrolyte [mol.m-3]", "initial"),
    ParamSpec("c0_neg", "Initial concentration in negative electrode [mol.m-3]", "initial"),
]

# 逐参数扰动因子覆盖（默认使用全局 lower/upper）。
# c0_neg 基线已达负极最大浓度的 90%，上界须 < c_max（安全因子 1.109），收窄到 1.10。
FACTOR_OVERRIDES = {
    "c0_neg": (0.8, 1.10),
}


def get_factors(short, perturb_cfg):
    """返回某参数的 (lower_factor, upper_factor)。"""
    if short in FACTOR_OVERRIDES:
        return FACTOR_OVERRIDES[short]
    return perturb_cfg["lower_factor"], perturb_cfg["upper_factor"]


def baseline_parameters(parameter_set: str = "Chen2020"):
    """返回参数集对象与每个参数的基线值。"""
    values = pybamm.ParameterValues(parameter_set)
    baselines = {}
    for spec in PARAM_SPECS:
        v = values[spec.pybamm_key]
        if callable(v):
            raise TypeError(f"参数 {spec.pybamm_key} 不是标量，无法直接扰动")
        baselines[spec.short] = float(v)
    return values, baselines


def build_problem():
    """构建 SALib problem（归一化 [0,1] 空间）。"""
    names = [s.short for s in PARAM_SPECS]
    return {
        "num_vars": len(PARAM_SPECS),
        "names": names,
        "bounds": [[0.0, 1.0] for _ in PARAM_SPECS],
    }


def denormalize(x_unit, baselines, perturb_cfg):
    """把归一化向量 [0,1] 映射为实际参数值（乘法线性扰动，支持逐参数范围）。"""
    actual = {}
    for spec, u in zip(PARAM_SPECS, x_unit):
        lo, hi = get_factors(spec.short, perturb_cfg)
        factor = lo + (hi - lo) * u
        actual[spec.pybamm_key] = baselines[spec.short] * factor
    return actual


def parameter_table(baselines, perturb_cfg) -> "pandas.DataFrame":
    """生成参数空间表（含基线与上下界）。"""
    import pandas as pd

    rows = []
    for s in PARAM_SPECS:
        b = baselines[s.short]
        lo, hi = get_factors(s.short, perturb_cfg)
        rows.append({
            "short": s.short,
            "pybamm_key": s.pybamm_key,
            "category": s.category,
            "baseline": b,
            "lower_factor": lo,
            "upper_factor": hi,
            "lower": b * lo,
            "upper": b * hi,
        })
    return pd.DataFrame(rows)
