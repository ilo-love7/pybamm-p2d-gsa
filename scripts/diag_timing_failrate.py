# -*- coding: utf-8 -*-
"""随机点耗时与失败率测试。"""
import time

import numpy as np
import pybamm
import yaml

pybamm.set_logging_level("ERROR")
from p2d_gsa.parameters import baseline_parameters, parameter_table
from p2d_gsa.model import build_model, run_one

cfg = yaml.safe_load(open("configs/gsa_config.yaml", encoding="utf-8"))
_, base = baseline_parameters()
parameter_table(base, cfg["perturbation"]).to_csv(
    "results/samples/parameter_space.csv", index=False, encoding="utf-8-sig")
m = build_model()
rng = np.random.default_rng(42)
n = 20

for sc in ("s1_discharge_1c", "s2_fastcharge_2c", "s3_cold_charge"):
    ts, fails = [], 0
    for _ in range(n):
        x = rng.random(24)
        t0 = time.perf_counter()
        try:
            run_one(x, cfg["scenarios"][sc], base, cfg["perturbation"], model=m)
            ts.append(time.perf_counter() - t0)
        except Exception:
            fails += 1
    ts = np.array(ts)
    print(f"{sc}: n={n} 失败={fails} 中位={np.median(ts):.2f}s "
          f"均值={ts.mean():.2f}s max={ts.max():.2f}s", flush=True)
