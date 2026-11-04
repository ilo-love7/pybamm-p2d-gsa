# -*- coding: utf-8 -*-
"""
二阶 S2 分析：对三个工况主输出提取成对二阶指数，
输出 Top 参数对 CSV、S2 热力图与 Top 成对条形图。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pandas as pd

from p2d_gsa.config import load_config
from p2d_gsa.parameters import build_problem
from p2d_gsa.second_order import (
    load_s2_y, analyze_s2, s2_pair_table, plot_s2_heatmap, plot_s2_top_pairs,
)


def main(top_n=15):
    cfg = load_config()
    problem = build_problem()
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    for sc in cfg["scenarios"].keys():
        main_out = list(cfg["scenarios"][sc]["outputs"].keys())[0]
        Y, status = load_s2_y(sc, cfg)
        n_fail = int((status == "fail").sum())
        print(f"=== {sc} ({main_out}) 失败={n_fail}/{len(status)} ===")

        res = analyze_s2(problem, Y[main_out], cfg)
        pair_df = s2_pair_table(res, top_n=top_n)
        pair_df.to_csv(f"results/s2_pairs_{sc}.csv", index=False,
                       encoding="utf-8-sig")
        plot_s2_heatmap(res, sc, main_out,
                        os.path.join(fig_dir, f"fig15_s2_heatmap_{sc}.png"))
        plot_s2_top_pairs(pair_df, sc, main_out,
                          os.path.join(fig_dir, f"fig16_s2_top_pairs_{sc}.png"))
        print(pair_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
