# -*- coding: utf-8 -*-
"""探查 PyBaMM DFN + 集总热模型与参数。"""
import pybamm

print("=== 可选参数集（部分） ===")
for name in pybamm.parameter_sets:
    if name in ("Chen2020", "Chen2020_composite", "Ecker2015", "Marquis2019",
                "Mohtat2020", "NCA_Kim2011", "OKane2022", "ORegan2022"):
        print(" ", name)

print("\n=== 构建 DFN + Lumped thermal ===")
model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
print("模型名:", model.name)
print("变量数:", len(model.variables))
for v in ("Discharge capacity [A.h]", "Terminal voltage [V]",
          "Maximum temperature [K]", "Minimum negative particle surface concentration",
          "X-averaged negative particle surface potential [V]"):
    print("  含变量:", v, v in model.variables)
