# -*- coding: utf-8 -*-
"""小规模 runner 测试 + 断点续跑验证。"""
import yaml

from p2d_gsa.parameters import baseline_parameters, build_problem
from p2d_gsa.sampling import generate_saltelli
from p2d_gsa.runner import run_design


def main():
    cfg = yaml.safe_load(open("configs/gsa_config.yaml", encoding="utf-8"))
    _, base = baseline_parameters()
    X = generate_saltelli(build_problem(), cfg["sobol"])
    Xsmall = X[:20]
    # 用一个临时测试工况名，避免污染正式结果
    print("--- 第一次运行（应跑 20）---")
    p, tot, s, fl = run_design("test", "s1_discharge_1c", Xsmall, cfg, base)
    print("结果:", tot, s, fl)
    print("--- 第二次运行（断点续跑，应 0 todo）---")
    p2, tot2, s2, fl2 = run_design("test", "s1_discharge_1c", Xsmall, cfg, base)
    print("结果:", tot2, s2, fl2)


if __name__ == "__main__":
    main()
