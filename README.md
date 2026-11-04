# 基于 PyBaMM 的 P2D 模型全局参数敏感性分析平台

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/ilo-love7/pybamm-p2d-gsa/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/pybamm-p2d-gsa/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-69%25-orange)](https://github.com/ilo-love7/pybamm-p2d-gsa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-DFN-orange)]()
[![No Deep Learning](https://img.shields.io/badge/deep%20learning-none-brightgreen)]()

*Global Parameter Sensitivity Analysis for a PyBaMM P2D (DFN) model — physics-based simulation & statistics, no deep learning.*

对 Doyle-Fuller-Newman（P2D/DFN）电化学模型进行**全局参数敏感性分析（GSA）**的
可复现平台。纯物理仿真 + 统计，**零深度学习、零模型训练**。采用两阶段策略：

1. **Morris 筛选**：以基本效应（elementary effects）的 μ\*、σ 快速识别重要参数；
2. **Sobol 方差分解**：基于 Saltelli 抽样计算一阶 S1 与总效应 ST，含 bootstrap
   95% 置信区间、收敛性检查，并给出不敏感参数白名单。

## 核心结果

- 共执行 **21468 次仿真**，成功 21380，**失败率 0.41%**（阈值 <5%），全部指标含
  bootstrap 95% 置信区间。
- 1C 放电容量由电极几何与活性材料分数主导（electrode_height ST=0.34）。
- 2C 快充温升由环境温度绝对主导（ambient_t ST=0.96）。
- 负极表面电位（析锂指示）由负极初始浓度绝对主导（c0_neg ST=0.91）。
- 全局不敏感参数白名单 17 个（ST<5%）。
- 交互效应量化（ST−S1）：冷充析锂为强耦合过程，c0_neg 交互量 0.29、
  electrode_height 交互占比 85%；快充温升几乎为纯主效应（ambient_t 交互占比 1%）。
- 方法交叉验证：Morris μ* 与 Sobol ST 秩高度一致，s1/s3 Spearman 分别为 0.96/0.91，
  强非线性 2C 工况（s2）两方法分歧更大（0.71）；跨工况共识重要参数为 ambient_t、
  c0_neg、electrode_height。
- 完整成对二阶 S2：独立二阶 Saltelli 设计（calc_second_order=true，6400×24，三工况
  共 19200 次仿真，失败率 0.08%–0.66%）。1C 放电容量最强交互为 electrode_height×
  c0_neg（S2=0.25）；冷充析锂为 neg_thickness×c0_neg（0.16）；2C 温升所有成对
  S2 均 <0.03，印证 ambient_t 近乎纯主效应。

## 模型与参数

- 模型：`pybamm.lithium_ion.DFN`，`thermal="lumped"` 集总热；
- 参数集：Chen2020（锂离子圆柱电芯）；
- 参数空间：24 个跨域标量参数（几何/孔隙/输运/动力学/热/初始浓度），统一乘法
  扰动 ±20%，其中负极初始浓度受最大浓度约束单独收窄上界。

## 3 个基准工况

| 工况 | 实验 | 输出指标 |
| --- | --- | --- |
| s1 | 1C 放电至 2.5V | 放电容量、截止时间 |
| s2 | 2C 快充 2400s | 最高温度 |
| s3 | 0.5C 充电 1800s | 负极表面电位最小值（析锂指示） |

## 目录

```
.
├── run_gsa.py              # 薄包装入口
├── configs/gsa_config.yaml
├── src/p2d_gsa/
│   ├── cli.py              # 包内统一入口（console 命令 p2d-gsa）
│   ├── config.py           # 多来源配置解析
│   ├── pipeline.py         # 全量分析编排
│   ├── parameters.py       # 24 参数空间与逐参数范围
│   ├── model.py            # DFN+热，单次仿真
│   ├── sampling.py         # Morris/Saltelli 抽样
│   ├── runner.py           # 并行 + 断点续跑 + 失败处理
│   ├── analysis.py         # S1/ST、bootstrap CI、收敛、白名单
│   ├── interactions.py     # ST−S1 交互量化与参数效应图谱
│   ├── cross_validation.py # Morris/Sobol 交叉验证与参数共识图谱
│   ├── second_order.py     # 完整成对二阶 S2 分析
│   └── plotting.py         # 报告配图
├── scripts/
└── results/                # 样本、CSV、figures、日志
```

## 安装与使用

```bash
# 作为包安装（提供 console 命令 p2d-gsa）
python -m pip install .
# 开发模式（含测试/ruff 等开发依赖）
python -m pip install -e ".[dev]"

# 以下命令用 p2d-gsa 或 python run_gsa.py 等价
p2d-gsa sample                 # 生成设计
p2d-gsa run                    # 跑仿真（断点续跑，可中断重跑）
p2d-gsa run --method morris    # 只跑 Morris
p2d-gsa analyze                # 分析、出图、白名单
p2d-gsa all                    # 一键全流程

# 交互效应量化（ST−S1，无需新仿真）
python scripts/run_interaction_analysis.py

# Morris/Sobol 交叉验证 + 参数共识图谱（fig13/14）
python scripts/run_cross_validation.py

# 完整成对二阶 S2：独立设计仿真（断点续跑）后分析（fig15/16）
python scripts/run_s2_simulation.py
python scripts/run_s2_analysis.py
```

## 验收口径

- 有效仿真样本 ≥5000（目标 10000），失败率 <5%；
- S1/ST 均含 bootstrap 95% 置信区间；
- 提供递增样本量的收敛检查；
- 白名单阈值 ST<5%；
- 所有阈值/规模集中在 config，禁止硬编码。

## 报告

完整方法、参数空间、21468 次仿真结果、参数图谱、白名单与复现指南见
[`reports/P2D全局参数敏感性分析平台报告-v1.0.md`](reports/P2D全局参数敏感性分析平台报告-v1.0.md)。

## License

[MIT](LICENSE)
