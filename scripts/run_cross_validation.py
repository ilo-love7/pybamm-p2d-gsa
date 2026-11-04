# -*- coding: utf-8 -*-
"""
Morris/Sobol 交叉验证：对三个工况的主输出比较两类方法的参数排序一致性，
输出汇总 CSV、对齐表与交叉验证图。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pandas as pd

from p2d_gsa.config import load_config
from p2d_gsa.cross_validation import (
    cross_validate, plot_cross_validation, build_consensus,
    plot_consensus_atlas,
)


def main(top_n=6):
    cfg = load_config()
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    summary_rows = []
    cv_results = {}
    for sc in cfg["scenarios"].keys():
        main_out = list(cfg["scenarios"][sc]["outputs"].keys())[0]
        morris_df = pd.read_csv(f"results/morris_{sc}_{main_out}.csv")
        sobol_df = pd.read_csv(f"results/sobol_{sc}_{main_out}.csv")

        result = cross_validate(morris_df, sobol_df, top_n=top_n)
        cv_results[sc] = result
        result["table"].to_csv(
            f"results/cross_validation_{sc}.csv", index=False,
            encoding="utf-8-sig")
        plot_cross_validation(
            result, f"{sc}_{main_out}",
            os.path.join(fig_dir, f"fig13_cross_validation_{sc}.png"))

        summary_rows.append({
            "scenario": sc, "main_output": main_out,
            "spearman_rho": result["spearman_rho"],
            "spearman_p": result["spearman_p"],
            "kendall_tau": result["kendall_tau"],
            "top_n": top_n, "top_overlap_count": result["top_overlap_count"],
            "top_overlap": "; ".join(result["top_overlap"]),
        })

    # 跨方法、跨工况共识图谱
    consensus = build_consensus(cv_results)
    consensus.to_csv("results/parameter_consensus.csv", index=False,
                     encoding="utf-8-sig")
    plot_consensus_atlas(cv_results, consensus,
                         os.path.join(fig_dir, "fig14_consensus_atlas.png"))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/cross_validation_summary.csv", index=False,
                   encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("\n=== 跨方法共识参数（前 10） ===")
    print(consensus.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
