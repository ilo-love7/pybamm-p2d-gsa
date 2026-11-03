# -*- coding: utf-8 -*-
"""3 工况基准冒烟测试。"""
import numpy as np
import pybamm

pybamm.set_logging_level("WARNING")
model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
params = pybamm.ParameterValues("Chen2020")

# 工况1：1C 放电
exp1 = pybamm.Experiment(["Discharge at 1C until 2.5 V"])
sim = pybamm.Simulation(model, parameter_values=params, experiment=exp1)
sol = sim.solve()
cap = sol["Discharge capacity [A.h]"].entries[-1]
t_end = sol["Time [s]"].entries[-1]
print(f"工况1 1C放电: 容量={cap:.4f} Ah 截止时间={t_end:.1f} s")

# 工况2：2C 快充 40 min（2400s），取最高温升
exp2 = pybamm.Experiment(["Charge at 2C for 2400 seconds"])
sim2 = pybamm.Simulation(model, parameter_values=params, experiment=exp2)
sol2 = sim2.solve()
T = sol2["X-averaged cell temperature [C]"].entries
print(f"工况2 2C快充: 最高温度={T.max():.2f} C 温升={T.max()-T[0]:.2f} C")

# 工况3：-10 C 充电，取负极表面电位最小值
p3 = pybamm.ParameterValues("Chen2020")
p3["Ambient temperature [K]"] = 263.15
p3["Initial temperature [K]"] = 263.15
exp3 = pybamm.Experiment(["Charge at 0.5C for 1800 seconds"])
sim3 = pybamm.Simulation(model, parameter_values=p3, experiment=exp3)
sol3 = sim3.solve()
phi = sol3["X-averaged negative electrode surface potential difference [V]"].entries
print(f"工况3 -10C充电: 负极表面电位最小值={phi.min():.4f} V")
