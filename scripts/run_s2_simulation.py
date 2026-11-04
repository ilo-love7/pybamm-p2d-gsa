# -*- coding: utf-8 -*-
"""
二阶 S2 专用：生成 calc_second_order=true 的 Saltelli 设计并对 3 工况跑仿真。
支持断点续跑（复用 runner.run_design），可中断后重跑。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from p2d_gsa.config import load_config
from p2d_gsa.parameters import build_problem, baseline_parameters
from p2d_gsa.sampling import generate_saltelli, save_design, load_design
from p2d_gsa.runner import run_design


def main():
    cfg = load_config()
    ckpt = cfg["runtime"]["checkpoint_dir"]
    os.makedirs(ckpt, exist_ok=True)
    problem = build_problem()

    design_path = os.path.join(ckpt, "design_s2.npy")
    if os.path.exists(design_path):
        X = load_design("s2", ckpt)
        print(f"复用已有二阶设计 {X.shape}")
    else:
        X = generate_saltelli(problem, cfg["sobol_s2"])
        save_design("s2", X, ckpt)
        print(f"生成二阶设计 {X.shape} -> {design_path}")

    parameter_set = cfg["model"]["parameter_set"]
    _, baselines = baseline_parameters(parameter_set)
    for sc in cfg["scenarios"].keys():
        path, total, succ, fail = run_design(
            "s2", sc, X, cfg, baselines, parameter_set)
        print(f"s2/{sc}: total={total} success={succ} fail={fail} "
              f"fail_rate={fail/max(total,1):.4f}", flush=True)


if __name__ == "__main__":
    main()
