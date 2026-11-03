# -*- coding: utf-8 -*-
"""全量分析编排：Morris/Sobol 指标、参数图谱、白名单、配图（包内模块）。"""
from __future__ import annotations

import os

import pandas as pd

from .parameters import build_problem, PARAM_SPECS
from .sampling import load_design
from .analysis import (
    load_run_results, analyze_morris, analyze_sobol,
    sobol_convergence, build_whitelist,
)
from . import plotting


def run_analysis(cfg):
    problem = build_problem()
    ckpt = cfg["runtime"]["checkpoint_dir"]
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    scenarios = list(cfg["scenarios"].keys())

    Xm = load_design("morris", ckpt)

    st_columns = {}
    fail_summary = []
    whitelist_sets = {}

    for sc in scenarios:
        print(f"=== {sc} ===", flush=True)
        Ym, stm = load_run_results("morris", sc, cfg)
        for out_name, Y in Ym.items():
            mdf = analyze_morris(problem, Xm, Y, cfg)
            mdf.to_csv(f"results/morris_{sc}_{out_name}.csv",
                       index=False, encoding="utf-8-sig")
            plotting.plot_morris_bar(mdf, f"{sc}_{out_name}", fig_dir)
            plotting.plot_morris_scatter(mdf, f"{sc}_{out_name}", fig_dir)
        fail_summary.append(("morris", sc, int((stm == "fail").sum()), len(stm)))

        Ys_all, sts = load_run_results("saltelli", sc, cfg)
        for out_name, Y in Ys_all.items():
            sdf = analyze_sobol(problem, Y, cfg)
            sdf.to_csv(f"results/sobol_{sc}_{out_name}.csv",
                       index=False, encoding="utf-8-sig")
            plotting.plot_sobol_st(sdf, f"{sc}_{out_name}", fig_dir)
            plotting.plot_sobol_s1_vs_st(sdf, f"{sc}_{out_name}", fig_dir)
            conv = sobol_convergence(problem, Y, cfg)
            conv.to_csv(f"results/convergence_{sc}_{out_name}.csv",
                        index=False, encoding="utf-8-sig")
            plotting.plot_convergence(conv, f"{sc}_{out_name}", fig_dir)
            wl = build_whitelist(sdf, cfg)
            wl.to_csv(f"results/whitelist_{sc}_{out_name}.csv",
                      index=False, encoding="utf-8-sig")
            whitelist_sets[f"{sc}_{out_name}"] = set(wl["parameter"])
        fail_summary.append(("saltelli", sc, int((sts == "fail").sum()), len(sts)))

        main_out = list(cfg["scenarios"][sc]["outputs"].keys())[0]
        sdf_main = analyze_sobol(problem, Ys_all[main_out], cfg)
        st_columns[sc] = sdf_main.set_index("parameter").loc[
            [s.short for s in PARAM_SPECS], "ST"]

    st_matrix = pd.DataFrame(st_columns)
    st_matrix.to_csv("results/st_matrix_all.csv", encoding="utf-8-sig")
    plotting.plot_st_heatmap(st_matrix, fig_dir)

    global_wl = set([s.short for s in PARAM_SPECS])
    for s in whitelist_sets.values():
        global_wl &= s
    gwl_df = pd.DataFrame({"parameter": sorted(global_wl)})
    gwl_df.to_csv("results/whitelist_global.csv", index=False,
                  encoding="utf-8-sig")
    plotting.plot_whitelist(
        st_matrix.loc[sorted(global_wl)].assign(
            ST=st_matrix.loc[sorted(global_wl)].max(axis=1)).reset_index().rename(
            columns={"index": "parameter"})[["parameter", "ST"]],
        cfg["statistics"]["whitelist_st_threshold"], fig_dir)

    print("\n=== 失败率 ===", flush=True)
    tot_f = tot_n = 0
    for method, sc, f, n in fail_summary:
        print(f"{method:8} {sc:20} fail={f:3}/{n} rate={f/n:.4f}", flush=True)
        tot_f += f
        tot_n += n
    print(f"总失败率={tot_f/tot_n:.4f}", flush=True)
    print(f"\n全局白名单参数({len(global_wl)}): {sorted(global_wl)}", flush=True)
