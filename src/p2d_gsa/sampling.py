# -*- coding: utf-8 -*-
"""Morris / Saltelli 抽样与样本落盘。"""
from __future__ import annotations

import os

import numpy as np
from SALib.sample import morris, saltelli


def generate_morris(problem, morris_cfg, seed=123):
    """生成 Morris 样本（先抽 num_trajectories，再选 optimal_trajectories 最优）。"""
    X = morris.sample(
        problem,
        N=morris_cfg["num_trajectories"],
        num_levels=morris_cfg["num_levels"],
        optimal_trajectories=morris_cfg.get("optimal_trajectories"),
        local_optimization=True,
        seed=seed,
    )
    return np.asarray(X)


def generate_saltelli(problem, sobol_cfg, seed=123):
    """生成 Saltelli 样本。"""
    np.random.seed(seed)
    X = saltelli.sample(
        problem,
        N=sobol_cfg["base_n"],
        calc_second_order=sobol_cfg.get("calc_second_order", False),
    )
    return np.asarray(X)


def save_design(method, X, checkpoint_dir):
    path = os.path.join(checkpoint_dir, f"design_{method}.npy")
    np.save(path, X)
    return path


def load_design(method, checkpoint_dir):
    path = os.path.join(checkpoint_dir, f"design_{method}.npy")
    return np.load(path)
