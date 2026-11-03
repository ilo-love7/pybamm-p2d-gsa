# -*- coding: utf-8 -*-
"""Morris/Sobol 分析、bootstrap CI、收敛性与白名单。"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze

from .parameters import PARAM_SPECS

SHORT_NAMES = [s.short for s in PARAM_SPECS]


def load_run_results(method, scenario, cfg):
    """读取跑批结果，按 index 对齐返回各输出列；失败为 NaN。"""
    ckpt = cfg["runtime"]["checkpoint_dir"]
    path = os.path.join(ckpt, f"run_{method}_{scenario}.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    out_cols = list(cfg["scenarios"][scenario]["outputs"].keys())
    df = df.sort_values("index").set_index("index")
    n_design = {
        "morris": cfg["morris"]["optimal_trajectories"] * (len(PARAM_SPECS) + 1),
        "saltelli": cfg["sobol"]["base_n"] * (len(PARAM_SPECS) + 2),
    }[method]
    df = df.reindex(range(n_design))
    Y = {c: df[c].to_numpy(dtype=float) for c in out_cols}
    status = df["status"].to_numpy()
    return Y, status


def _fill_nan(Y, n_fail):
    """SALib 不接受 NaN：用有限值中位数填充（失败率应 <5%）。"""
    finite = Y[np.isfinite(Y)]
    fill = np.median(finite) if finite.size else 0.0
    out = Y.copy()
    out[~np.isfinite(out)] = fill
    return out, fill


def analyze_morris(problem, X, Y, cfg, seed=2026):
    """Morris 分析，返回含 mu/mu_star/sigma 及置信的 DataFrame。"""
    Yf, fill = _fill_nan(Y, None)
    res = morris_analyze.analyze(
        problem, X, Yf,
        num_resamples=cfg["statistics"]["bootstrap_n"],
        conf_level=cfg["statistics"]["confidence"],
        num_levels=cfg["morris"]["num_levels"],
        seed=seed,
    )
    return pd.DataFrame({
        "parameter": SHORT_NAMES,
        "mu": res["mu"],
        "mu_star": res["mu_star"],
        "sigma": res["sigma"],
        "mu_star_conf": res["mu_star_conf"],
    }).sort_values("mu_star", ascending=False).reset_index(drop=True)


def _saltelli_subY(Y, d, N, N_sub):
    """
    从交错布局 Saltelli Y 中提取前 N' 个 base 点。
    每个 base 点占 step=d+2 行（A, AB_0..AB_{d-1}, B），按完整块切分。
    """
    step = d + 2
    parts = [Y[i * step:(i + 1) * step] for i in range(N_sub)]
    return np.concatenate(parts)


def analyze_sobol(problem, Y, cfg, seed=2026):
    """Sobol 分析，返回 S1/ST 及 bootstrap 95% CI。"""
    Yf, fill = _fill_nan(Y, None)
    res = sobol_analyze.analyze(
        problem, Yf,
        calc_second_order=cfg["sobol"].get("calc_second_order", False),
        num_resamples=cfg["statistics"]["bootstrap_n"],
        conf_level=cfg["statistics"]["confidence"],
        seed=seed,
    )
    return pd.DataFrame({
        "parameter": SHORT_NAMES,
        "S1": res["S1"],
        "S1_conf": res["S1_conf"],
        "ST": res["ST"],
        "ST_conf": res["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)


def sobol_convergence(problem, Y, cfg):
    """递增 base N 子集，观察 S1/ST 收敛。"""
    # 先填充失败点，与主分析口径一致，避免 NaN 传播导致指标全 0
    Y, _ = _fill_nan(Y, None)
    d = problem["num_vars"]
    N = cfg["sobol"]["base_n"]
    fractions = [0.25, 0.5, 0.75, 1.0]
    rows = []
    for frac in fractions:
        N_sub = int(N * frac)
        subY = _saltelli_subY(Y, d, N, N_sub)
        sub_problem = dict(problem)
        res = sobol_analyze.analyze(
            sub_problem, subY,
            calc_second_order=False, num_resamples=200,
            conf_level=cfg["statistics"]["confidence"], seed=2026,
        )
        for i, p in enumerate(SHORT_NAMES):
            rows.append({"fraction": frac, "N": N_sub, "parameter": p,
                         "S1": res["S1"][i], "ST": res["ST"][i]})
    return pd.DataFrame(rows)


def build_whitelist(sobol_df, cfg):
    """ST < 阈值的不敏感参数白名单。"""
    thr = cfg["statistics"]["whitelist_st_threshold"]
    wl = sobol_df[sobol_df["ST"] < thr].copy()
    return wl.sort_values("ST", ascending=True).reset_index(drop=True)
