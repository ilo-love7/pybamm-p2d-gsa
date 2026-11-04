# -*- coding: utf-8 -*-
"""
二阶 Sobol S2 分析：在 calc_second_order=true 设计上提取成对二阶指数，
输出 Top 参数对与 S2 热力图。零训练，复用 SALib 方差分析。
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze

from .analysis import _fill_nan
from .parameters import PARAM_SPECS

SHORT_NAMES = [s.short for s in PARAM_SPECS]


def load_s2_y(scenario: str, cfg) -> Dict:
    """读取二阶跑批结果，按 index 对齐到完整二阶设计。"""
    ckpt = cfg["runtime"]["checkpoint_dir"]
    path = os.path.join(ckpt, f"run_s2_{scenario}.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.sort_values("index").set_index("index")
    n_design = cfg["sobol_s2"]["base_n"] * (2 * len(PARAM_SPECS) + 2)
    df = df.reindex(range(n_design))
    out_cols = list(cfg["scenarios"][scenario]["outputs"].keys())
    Y = {c: df[c].to_numpy(dtype=float) for c in out_cols}
    status = df["status"].to_numpy()
    return Y, status


def analyze_s2(problem, Y, cfg, seed=2026):
    """Sobol 二阶分析，返回含 S2/S2_conf 的完整 SALib 结果。"""
    Yf, _ = _fill_nan(Y, None)
    return sobol_analyze.analyze(
        problem, Yf, calc_second_order=True,
        num_resamples=cfg["statistics"]["bootstrap_n"],
        conf_level=cfg["statistics"]["confidence"], seed=seed,
    )


def s2_pair_table(res, top_n: int = 15) -> pd.DataFrame:
    """提取上三角参数对的 S2 并降序排列。"""
    d = len(SHORT_NAMES)
    rows = []
    for i in range(d):
        for j in range(i + 1, d):
            rows.append({
                "param_i": SHORT_NAMES[i], "param_j": SHORT_NAMES[j],
                "S2": float(res["S2"][i, j]),
                "S2_conf": float(res["S2_conf"][i, j]),
            })
    table = pd.DataFrame(rows).sort_values(
        "S2", ascending=False).reset_index(drop=True)
    return table.head(top_n).reset_index(drop=True)


def plot_s2_heatmap(res, scenario, output, fig_path: str) -> None:
    d = len(SHORT_NAMES)
    s2 = np.asarray(res["S2"], dtype=float)
    s2 = np.clip(s2, 0, None)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(s2, cmap="magma", aspect="auto")
    ax.set_xticks(range(d), labels=SHORT_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(d), labels=SHORT_NAMES, fontsize=7)
    for i in range(d):
        for j in range(d):
            if s2[i, j] > 0.03:
                ax.text(j, i, f"{s2[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white")
    ax.set_title(f"Second-order Sobol S2 map — {scenario} ({output})")
    fig.colorbar(im, ax=ax, label="Pairwise S2")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_s2_top_pairs(pair_df: pd.DataFrame, scenario, output,
                      fig_path: str) -> None:
    sub = pair_df.iloc[::-1]
    labels = [f"{a} × {b}" for a, b in zip(sub["param_i"], sub["param_j"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, sub["S2"], xerr=sub["S2_conf"], color="#7a5195", alpha=0.9)
    ax.set_xlabel(r"Second-order Sobol index $S_2$")
    ax.set_title(f"Top pairwise interactions — {scenario} ({output})")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
