# -*- coding: utf-8 -*-
"""
P2D 全局参数敏感性分析平台 · config 驱动统一入口。

子命令：
  sample   生成 Morris / Saltelli 设计并落盘
  run      运行仿真（支持断点续跑，可按 method/scenario 过滤）
  analyze  计算敏感性指标、bootstrap CI、收敛、白名单并出图
  all      依次执行 sample -> run -> analyze
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import yaml  # noqa: E402

from p2d_gsa.parameters import baseline_parameters, build_problem  # noqa: E402
from p2d_gsa.sampling import (  # noqa: E402
    generate_morris, generate_saltelli, save_design, load_design,
)
from p2d_gsa.runner import run_design  # noqa: E402


def load_cfg(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def cmd_sample(cfg, args):
    problem = build_problem()
    Xm = generate_morris(problem, cfg["morris"])
    Xs = generate_saltelli(problem, cfg["sobol"])
    p1 = save_design("morris", Xm, cfg["runtime"]["checkpoint_dir"])
    p2 = save_design("saltelli", Xs, cfg["runtime"]["checkpoint_dir"])
    print(f"Morris {Xm.shape} -> {p1}")
    print(f"Saltelli {Xs.shape} -> {p2}")


def cmd_run(cfg, args):
    parameter_set = cfg["model"]["parameter_set"]
    _, baselines = baseline_parameters(parameter_set)
    methods = [args.method] if args.method else ("morris", "saltelli")
    scenarios = [args.scenario] if args.scenario else list(cfg["scenarios"].keys())
    for method in methods:
        X = load_design(method, cfg["runtime"]["checkpoint_dir"])
        for sc in scenarios:
            path, total, succ, fail = run_design(
                method, sc, X, cfg, baselines, parameter_set)
            print(f"{method}/{sc}: total={total} success={succ} fail={fail} "
                  f"fail_rate={fail/max(total,1):.4f}")


def cmd_analyze(cfg, args):
    # 复用分析编排脚本逻辑
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "scripts"))
    import analyze_all
    analyze_all.main()


def main():
    parser = argparse.ArgumentParser(description="P2D GSA platform")
    parser.add_argument("--config", default="configs/gsa_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sample")
    sp.set_defaults(func=cmd_sample)

    rp = sub.add_parser("run")
    rp.add_argument("--method", choices=["morris", "saltelli"])
    rp.add_argument("--scenario")
    rp.set_defaults(func=cmd_run)

    ap = sub.add_parser("analyze")
    ap.set_defaults(func=cmd_analyze)

    allp = sub.add_parser("all")
    allp.set_defaults(func=lambda c, a: (cmd_sample(c, a), cmd_run(c, a),
                                         cmd_analyze(c, a)))

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
