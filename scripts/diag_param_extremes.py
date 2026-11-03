# -*- coding: utf-8 -*-
"""逐参数扫描极端值，识别单独导致失败的参数。"""
import numpy as np
import pybamm
import yaml

pybamm.set_logging_level("ERROR")
from p2d_gsa.parameters import baseline_parameters, PARAM_SPECS
from p2d_gsa.model import build_model, run_one

cfg = yaml.safe_load(open("configs/gsa_config.yaml", encoding="utf-8"))
_, base = baseline_parameters()
m = build_model()
d = len(PARAM_SPECS)

for sc in ("s1_discharge_1c", "s2_fastcharge_2c", "s3_cold_charge"):
    print(f"=== {sc} ===", flush=True)
    for i, spec in enumerate(PARAM_SPECS):
        for val, tag in ((0.0, "lower"), (1.0, "upper")):
            x = np.full(d, 0.5)
            x[i] = val
            try:
                run_one(x, cfg["scenarios"][sc], base, cfg["perturbation"], model=m)
                ok = "OK"
            except Exception as e:
                ok = f"FAIL {type(e).__name__}"
            if ok != "OK":
                print(f"  {spec.short:16} {tag:5} {ok}", flush=True)
