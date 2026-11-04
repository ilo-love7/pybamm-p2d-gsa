# -*- coding: utf-8 -*-
"""
Morris 与 Sobol 交叉验证：比较两类全局敏感性方法的参数排序一致性，
量化秩相关、Top-N 重要参数重合与归一化效应差异。零训练。
"""
from __future__ import annotations

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def build_consensus(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    汇总各工况 cross_validate 结果，返回每个参数的跨工况共识重要性与分歧度。
    consensus 为 Morris/Sobol 归一化值的均值，再对各工况取最大。
    """
    frames = []
    for sc, result in results.items():
        t = result["table"].copy()
        t["scenario"] = sc
        t["consensus"] = 0.5 * (t["mu_star_norm"] + t["st_norm"])
        frames.append(t[["parameter", "scenario", "mu_star_norm",
                         "st_norm", "consensus", "abs_norm_gap"]])
    long = pd.concat(frames, ignore_index=True)
    grouped = long.groupby("parameter").agg(
        consensus_score=("consensus", "max"),
        mean_consensus=("consensus", "mean"),
        mean_method_gap=("abs_norm_gap", "mean"),
    ).reset_index()
    return grouped.sort_values("consensus_score",
                               ascending=False).reset_index(drop=True)


def plot_consensus_atlas(results: Dict[str, Dict], consensus: pd.DataFrame,
                         fig_path: str) -> None:
    # 各工况共识热力矩阵（参数 × 工况）
    long = pd.concat([
        result["table"].assign(scenario=sc).assign(
            consensus=lambda d: 0.5 * (d["mu_star_norm"] + d["st_norm"]))
        for sc, result in results.items()
    ], ignore_index=True)
    matrix = long.pivot(index="parameter", columns="scenario",
                        values="consensus")
    order = consensus["parameter"].tolist()
    matrix = matrix.loc[order]

    fig, ax = plt.subplots(figsize=(10, 9))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="magma", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), labels=matrix.columns,
                  rotation=25, ha="right")
    ax.set_yticks(range(matrix.shape[0]), labels=matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0.02:
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white")
    ax.set_title("Cross-method consensus importance map (Morris + Sobol)")
    fig.colorbar(im, ax=ax, label="Consensus normalized sensitivity")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cross_validate(morris_df: pd.DataFrame, sobol_df: pd.DataFrame,
                   top_n: int = 6) -> Dict:
    """对齐 Morris mu_star 与 Sobol ST，返回秩相关与 Top-N 重合结果。"""
    merged = morris_df[["parameter", "mu_star"]].merge(
        sobol_df[["parameter", "ST"]], on="parameter")
    spearman_rho, spearman_p = spearmanr(merged["mu_star"], merged["ST"])
    kendall_tau, _ = kendalltau(merged["mu_star"], merged["ST"])

    merged["mu_star_norm"] = merged["mu_star"] / merged["mu_star"].max()
    merged["st_norm"] = merged["ST"] / merged["ST"].max()
    merged["abs_norm_gap"] = (
        merged["mu_star_norm"] - merged["st_norm"]).abs()

    top_morris = set(merged.nlargest(top_n, "mu_star")["parameter"])
    top_sobol = set(merged.nlargest(top_n, "ST")["parameter"])
    overlap = top_morris & top_sobol

    return {
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "top_n": top_n,
        "top_overlap_count": len(overlap),
        "top_overlap": sorted(overlap),
        "table": merged.sort_values("mu_star", ascending=False)
        .reset_index(drop=True),
    }


def plot_cross_validation(result: Dict, scenario: str, fig_path: str) -> None:
    table = result["table"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 左：归一化效应对比（按 mu_star 排序的前 12 项）
    sub = table.head(12).iloc[::-1]
    y = np.arange(len(sub))
    h = 0.38
    ax = axes[0]
    ax.barh(y + h/2, sub["mu_star_norm"], height=h,
            color="#3b6fb6", label=r"Morris $\mu^*$ (normalized)")
    ax.barh(y - h/2, sub["st_norm"], height=h,
            color="#4f8a5b", label=r"Sobol $S_T$ (normalized)")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["parameter"], fontsize=8)
    ax.set_xlabel("Normalized sensitivity")
    ax.set_title(f"Normalized effect comparison — {scenario}")
    ax.legend(fontsize=8)

    # 右：mu_star_norm vs st_norm 散点
    ax2 = axes[1]
    ax2.scatter(table["mu_star_norm"], table["st_norm"],
                color="#c0504d", alpha=0.8)
    for _, r in table.iterrows():
        ax2.annotate(r["parameter"], (r["mu_star_norm"], r["st_norm"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"Morris $\mu^*$ (normalized)")
    ax2.set_ylabel(r"Sobol $S_T$ (normalized)")
    ax2.set_title(
        f"Rank agreement (Spearman={result['spearman_rho']:.2f}, "
        f"Top-{result['top_n']} overlap={result['top_overlap_count']})")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
