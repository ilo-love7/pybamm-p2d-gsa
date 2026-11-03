# -*- coding: utf-8 -*-
"""全量仿真：Morris + Saltelli × 3 工况，支持断点续跑。"""
import time

import yaml

from p2d_gsa.parameters import baseline_parameters
from p2d_gsa.sampling import load_design
from p2d_gsa.runner import run_design


def main():
    cfg = yaml.safe_load(open("configs/gsa_config.yaml", encoding="utf-8"))
    parameter_set = cfg["model"]["parameter_set"]
    _, baselines = baseline_parameters(parameter_set)
    ckpt = cfg["runtime"]["checkpoint_dir"]
    scenarios = list(cfg["scenarios"].keys())

    grand_t0 = time.perf_counter()
    summary = []
    for method in ("morris", "saltelli"):
        X = load_design(method, ckpt)
        for sc in scenarios:
            t0 = time.perf_counter()
            print(f"=== {method} / {sc} （{len(X)} 点）===", flush=True)
            path, total, succ, fail = run_design(
                method, sc, X, cfg, baselines, parameter_set)
            dt = time.perf_counter() - t0
            rate = fail / total if total else 0.0
            print(f"  完成 total={total} success={succ} fail={fail} "
                  f"fail_rate={rate:.4f} 用时={dt:.1f}s", flush=True)
            summary.append((method, sc, total, succ, fail, rate, dt))

    print("\n=== 全部汇总 ===", flush=True)
    tot_eval = tot_succ = tot_fail = 0
    for method, sc, total, succ, fail, rate, dt in summary:
        print(f"{method:8} {sc:20} n={total:5} succ={succ:5} fail={fail:3} "
              f"rate={rate:.4f} {dt:.0f}s", flush=True)
        tot_eval += total
        tot_succ += succ
        tot_fail += fail
    print(f"总仿真={tot_eval} 成功={tot_succ} 失败={tot_fail} "
          f"总失败率={tot_fail/max(tot_eval,1):.4f} "
          f"总用时={time.perf_counter()-grand_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
