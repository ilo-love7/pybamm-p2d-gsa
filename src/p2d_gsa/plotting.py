# -*- coding: utf-8 -*-
"""报告配图：Morris/Sobol 参数图谱、收敛与白名单。英文标签。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_morris_bar(morris_df, scenario, out_dir, top=15):
    df = morris_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["parameter"], df["mu_star"], xerr=df["mu_star_conf"],
            color="#3b6fb6", alpha=0.85)
    ax.set_xlabel(r"Morris $\mu^*$ (mean absolute elementary effect)")
    ax.set_title(f"Morris screening — {scenario}")
    return _save(fig, os.path.join(out_dir, f"morris_mustar_{scenario}.png"))


def plot_morris_scatter(morris_df, scenario, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(morris_df["mu_star"], morris_df["sigma"],
                fmt="o", color="#c0504d", alpha=0.8)
    for _, r in morris_df.iterrows():
        ax.annotate(r["parameter"], (r["mu_star"], r["sigma"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = max(morris_df["mu_star"].max(), morris_df["sigma"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\mu^*$ (magnitude)")
    ax.set_ylabel(r"$\sigma$ (non-linearity / interaction)")
    ax.set_title(f"Morris magnitude vs spread — {scenario}")
    return _save(fig, os.path.join(out_dir, f"morris_scatter_{scenario}.png"))


def plot_sobol_st(sobol_df, scenario, out_dir, top=15):
    df = sobol_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["parameter"], df["ST"], xerr=df["ST_conf"],
            color="#4f8a5b", alpha=0.85)
    ax.set_xlabel(r"Total-order Sobol index $S_T$")
    ax.set_title(f"Total-order sensitivity (95% CI) — {scenario}")
    return _save(fig, os.path.join(out_dir, f"sobol_st_{scenario}.png"))


def plot_sobol_s1_vs_st(sobol_df, scenario, out_dir, top=12):
    df = sobol_df.head(top).iloc[::-1]
    y = np.arange(len(df))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + h/2, df["S1"], height=h, xerr=df["S1_conf"],
            color="#3b6fb6", label=r"First-order $S_1$")
    ax.barh(y - h/2, df["ST"], height=h, xerr=df["ST_conf"],
            color="#c0504d", label=r"Total-order $S_T$")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Sobol index")
    ax.set_title(f"First vs total order — {scenario}")
    ax.legend()
    return _save(fig, os.path.join(out_dir, f"sobol_s1_st_{scenario}.png"))


def plot_convergence(conv_df, scenario, out_dir, top=6):
    top_params = conv_df[conv_df["fraction"] == 1.0].sort_values(
        "ST", ascending=False)["parameter"].head(top).tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    for p in top_params:
        sub = conv_df[conv_df["parameter"] == p].sort_values("N")
        ax.plot(sub["N"], sub["ST"], marker="o", label=p)
    ax.set_xlabel("Base sample size N")
    ax.set_ylabel(r"Total-order $S_T$")
    ax.set_title(f"Sobol convergence — {scenario}")
    ax.legend(fontsize=8)
    return _save(fig, os.path.join(out_dir, f"convergence_{scenario}.png"))


def plot_st_heatmap(st_matrix, out_dir):
    """参数图谱：参数(行) × 工况(列) 的 ST 热力图。"""
    fig, ax = plt.subplots(figsize=(9, 9))
    data = st_matrix.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_xticks(range(st_matrix.shape[1]))
    ax.set_xticklabels(st_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(st_matrix.shape[0]))
    ax.set_yticklabels(st_matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if data[i, j] < data.max()*0.6 else "black")
    fig.colorbar(im, ax=ax, label=r"$S_T$")
    ax.set_title("Total-order sensitivity map (parameter x scenario)")
    return _save(fig, os.path.join(out_dir, "st_heatmap_all.png"))


def plot_whitelist(whitelist_df, threshold, out_dir):
    df = whitelist_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(df)+1.5)))
    xerr = df["ST_conf"] if "ST_conf" in df.columns else None
    ax.barh(df["parameter"], df["ST"], xerr=xerr,
            color="#8a8f98", alpha=0.9)
    ax.axvline(threshold, color="red", ls="--", lw=1,
               label=f"threshold ST={threshold}")
    ax.set_xlabel(r"Total-order $S_T$")
    ax.set_title("Non-influential whitelist (ST below threshold)")
    ax.legend()
    return _save(fig, os.path.join(out_dir, "whitelist.png"))
