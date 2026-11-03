# -*- coding: utf-8 -*-
"""
交互效应量化（无需新仿真）：
Sobol 总效应 ST = S1 + 所有交互项之和，故 ST−S1 即该参数与其他参数的交互总量。
在 calc_second_order=false 设计下，这是不区分具体配对、可直接由现有结果计算的
交互量化路径；完整成对二阶 S2 需重新抽样（calc_second_order=true）。
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# (scenario, output, sobol结果文件名)
SOBOL_SOURCES: List[Tuple[str, str, str]] = [
    ("s1_discharge_1c", "capacity_ah",
     "sobol_s1_discharge_1c_capacity_ah.csv"),
    ("s1_discharge_1c", "time_to_cutoff_s",
     "sobol_s1_discharge_1c_time_to_cutoff_s.csv"),
    ("s2_fastcharge_2c", "max_temperature_c",
     "sobol_s2_fastcharge_2c_max_temperature_c.csv"),
    ("s3_cold_charge", "min_neg_phi_v",
     "sobol_s3_cold_charge_min_neg_phi_v.csv"),
]


def interaction_frame(sobol_df: pd.DataFrame) -> pd.DataFrame:
    """在单个 Sobol 结果上增加 S1 截断、交互量与交互占比列。"""
    df = sobol_df.copy()
    s1 = df["S1"].clip(lower=0.0)
    st = df["ST"].clip(lower=0.0)
    df["S1_clipped"] = s1
    df["interaction"] = (st - s1).clip(lower=0.0)
    # 交互占总效应比例（ST=0 时记 0）
    df["interaction_share"] = np.where(
        st > 1e-12, df["interaction"] / st.where(st > 1e-12), 0.0)
    return df


def build_interaction_table(results_dir: str) -> pd.DataFrame:
    """读取全部 Sobol 结果，返回长表（含工况/输出/交互量）。"""
    frames = []
    for scenario, output, fn in SOBOL_SOURCES:
        path = os.path.join(results_dir, fn)
        df = interaction_frame(pd.read_csv(path))
        df["scenario"] = scenario
        df["output"] = output
        frames.append(df)
    cols = ["scenario", "output", "parameter", "S1_clipped", "ST",
            "interaction", "interaction_share"]
    return pd.concat(frames, ignore_index=True)[cols]


def main_output_interaction_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """各工况主输出（每工况第一条来源）的交互量宽表，用于热力图。"""
    main_rows = []
    seen = set()
    for scenario, output, _ in SOBOL_SOURCES:
        if scenario in seen:
            continue
        seen.add(scenario)
        sub = table[(table["scenario"] == scenario) &
                    (table["output"] == output)]
        main_rows.append(sub)
    long = pd.concat(main_rows, ignore_index=True)
    return long.pivot(index="parameter", columns="scenario",
                     values="interaction")


def plot_interaction_heatmap(matrix: pd.DataFrame, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(matrix.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns,
                  rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix.to_numpy()[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    ax.set_title("Interaction effect (ST - S1) across scenarios")
    fig.colorbar(im, ax=ax, label="Interaction (ST - S1)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_interaction_share_bars(table: pd.DataFrame, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 每工况主输出，取交互量 top 8 参数
    seen = set()
    panels = []
    for scenario, output, _ in SOBOL_SOURCES:
        if scenario in seen:
            continue
        seen.add(scenario)
        sub = table[(table["scenario"] == scenario) &
                    (table["output"] == output)]
        sub = sub.sort_values("interaction", ascending=False).head(8)
        panels.append((scenario, sub))

    fig, axes = plt.subplots(1, len(panels), figsize=(15, 5), sharey=False)
    for ax, (scenario, sub) in zip(axes, panels):
        ax.barh(sub["parameter"][::-1], sub["interaction"][::-1],
                color="teal")
        ax.set_xlabel("Interaction (ST - S1)")
        ax.set_title(scenario)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Top interaction effects by scenario (main output)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_effect_atlas(table: pd.DataFrame, fig_path: str,
                      top_n: int = 12) -> None:
    """综合参数图谱：把主输出 ST 分解为 S1 主效应 + 交互，按类别标注。"""
    from .parameters import PARAM_SPECS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat_map = {s.short: s.category for s in PARAM_SPECS}
    seen = set()
    panels = []
    for scenario, output, _ in SOBOL_SOURCES:
        if scenario in seen:
            continue
        seen.add(scenario)
        sub = table[(table["scenario"] == scenario) &
                    (table["output"] == output)]
        sub = sub.sort_values("ST", ascending=False).head(top_n)
        panels.append((scenario, sub))

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 6), sharey=False)
    for ax, (scenario, sub) in zip(axes, panels):
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub["S1_clipped"], color="#2c7fb8", label="Main (S1)")
        ax.barh(y, sub["interaction"], left=sub["S1_clipped"],
                color="#f03b20", label="Interaction")
        labels = [f"{p} [{cat_map.get(p, '?')[:3]}]"
                  for p in sub["parameter"]]
        ax.set_yticks(y, labels=labels)
        ax.set_xlabel("Total effect ST (decomposed)")
        ax.set_title(scenario)
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(loc="lower right")
    fig.suptitle("Parameter effect atlas: main effect vs interaction")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
