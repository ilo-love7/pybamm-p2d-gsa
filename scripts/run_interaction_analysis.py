# -*- coding: utf-8 -*-
"""交互效应分析：计算 ST−S1 交互量，输出长表、主输出矩阵与配图。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pandas as pd

from p2d_gsa.interactions import (
    build_interaction_table, main_output_interaction_matrix,
    plot_effect_atlas, plot_interaction_heatmap, plot_interaction_share_bars,
)


def main():
    results_dir = "results"
    table = build_interaction_table(results_dir)
    matrix = main_output_interaction_matrix(table)

    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_interaction_heatmap(matrix, os.path.join(
        fig_dir, "fig8_interaction_heatmap.png"))
    plot_interaction_share_bars(table, os.path.join(
        fig_dir, "fig9_interaction_share_bars.png"))
    plot_effect_atlas(table, os.path.join(
        fig_dir, "fig10_parameter_effect_atlas.png"))

    table.to_csv("results/interaction_effects_all.csv",
                 index=False, encoding="utf-8-sig")
    matrix.to_csv("results/interaction_matrix_main_outputs.csv",
                  encoding="utf-8-sig")

    # 各工况交互量 top 5 打印
    seen = set()
    for scenario, output, _ in [
        ("s1_discharge_1c", "capacity_ah", None),
        ("s2_fastcharge_2c", "max_temperature_c", None),
        ("s3_cold_charge", "min_neg_phi_v", None),
    ]:
        sub = table[(table["scenario"] == scenario) &
                    (table["output"] == output)]
        top = sub.sort_values("interaction", ascending=False).head(5)
        print(f"=== {scenario}/{output} top5 交互量 ===")
        print(top[["parameter", "S1_clipped", "ST", "interaction",
                   "interaction_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
