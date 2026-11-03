# 基于 PyBaMM 的 P2D 模型全局参数敏感性分析平台报告

**版本：v1.0　|　日期：2026-11-02　|　模型：DFN（P2D）+ Lumped Thermal　|　参数集：Chen2020**

---

## 1. 摘要

本平台基于 PyBaMM 的 Doyle–Fuller–Newman（DFN，即 P2D）电化学模型，对覆盖几何、孔隙/体积分数、输运、动力学、热、初始浓度 6 大类的 **24 个标量参数**开展全局参数敏感性分析（Global Sensitivity Analysis, GSA）。方法上采用 **Morris 轨迹筛选（定性/半定量）+ Sobol 方差分解（定量）** 两阶段设计，配套 3 个代表性工况（1C 放电、2C 快充、0.5C 全温区充电），共执行 **21468 次物理仿真**，成功 21380 次，**总失败率 0.41%**，所有一阶（S1）与全阶（ST）指标均给出 bootstrap 95% 置信区间，并完成收敛性分析与不敏感参数白名单（17 个）。

全部流程为**物理仿真 + 经典统计，零深度学习、零模型训练**；所有阈值、工况、抽样规模均由配置文件驱动，无硬编码，且支持并行与断点续跑，可一键复现。

核心结论：

- **1C 放电容量**由电极几何与活性材料分数主导（electrode_height ST=0.34、neg_thickness 0.23、neg_amf 0.22、c0_neg 0.14），符合"活性材料/可用锂总量决定容量"的物理预期。
- **2C 快充最高温度**由环境温度绝对主导（ambient_t ST=0.96），其余参数贡献均低于 5%。
- **负极表面电位（析锂指示量）**由负极初始浓度绝对主导（c0_neg ST=0.91），其次为负极厚度、电极高度、负极活性材料分数；低温冒烟工况下负极表面电位可低至 −0.64 V，符合强析锂预期。
- 电荷转移系数（neg_alpha、pos_alpha）与负极热导率（neg_thermal_k）在 ±20% 扰动范围内对全部主输出 **ST=0**，对所选工况不敏感。

---

## 2. 平台概述与方法学

### 2.1 两阶段 GSA 设计

| 阶段 | 方法 | 性质 | 作用 |
| --- | --- | --- | --- |
| 筛选 | Morris 轨迹法（Elementary Effects） | 半定量、低成本 | 用 μ*（平均绝对基本效应）衡量参数重要性量级，用 σ 衡量非线性/交互程度，快速识别重要参数 |
| 定量 | Sobol 方差分解 | 定量、基于方差 | 将输出方差分解为各参数的一阶效应 S1（独立贡献）与全阶效应 ST（含全部交互），并给出置信区间 |

两方法相互印证：Morris 散点图（μ* vs σ）与 Sobol ST 排序在全部工况下结论一致，增强了结果可信度。

### 2.2 敏感性指标定义

- **一阶指数 S1**：参数单独变化对输出方差的贡献比例。
- **全阶指数 ST**：参数及其参与的所有交互作用对输出方差的总贡献比例。ST ≥ S1，二者差值反映该参数的交互作用强度。
- **判定口径**：以 ST 为主输出指标；**ST < 0.05（5%）** 列入不敏感参数白名单。
- 负的 S1 小值为蒙特卡洛估计噪声（置信区间跨 0），不表示真实负效应。

### 2.3 不确定性量化

- 所有 S1/ST 均通过 **bootstrap 重采样（1000 次）** 估计 **95% 置信区间**（S1_conf、ST_conf）。
- 收敛分析在 4 个递增子样本（base N = 64、128、192、256）上重复计算，观察指标随样本量的稳定性。

---

## 3. 基准模型与参数空间

### 3.1 模型配置

- 模型：PyBaMM `DFN`（Doyle–Fuller–Newman，约 530 个变量）+ `Lumped` 集总热模型。
- 参数集：`Chen2020`（LG M50 21700 电芯参数）。
- 求解器容差 1e-6，单次仿真中位耗时约 0.85–0.91 s。
- 关键变量：
  - 温度：`X-averaged cell temperature [C]`（集总热下空间均匀）。
  - 析锂指示量：`X-averaged negative electrode surface potential difference [V]`，该值低于 0 时负极表面热力学上倾向析锂。

### 3.2 三个基准工况

| 工况 | 实验设置 | 主输出（聚合方式） | 基线冒烟结果 |
| --- | --- | --- | --- |
| s1_discharge_1c | Discharge at 1C until 2.5 V | 容量 capacity_ah（last）、截止时间 time_to_cutoff_s（last） | 容量 4.9474 Ah，截止 3562.2 s |
| s2_fastcharge_2c | Charge at 2C for 2400 s | 最高温度 max_temperature_c（max） | 34.14 ℃，温升 9.14 ℃ |
| s3_cold_charge | Charge at 0.5C for 1800 s | 负极表面电位最小值 min_neg_phi_v（min） | −0.6446 V（低温冒烟） |

说明：2C/低温充电会触发最大电压事件（Maximum voltage event）而提前终止，属正常保护行为而非错误；s3 的环境温度不再硬编码，而是纳入统一参数空间（ambient_t，238–358 K）一起扰动，保证 GSA 因子维度一致，低温析锂物理已由冒烟测试验证。

### 3.3 24 参数空间

从 Chen2020 的 93 个参数中挑选 24 个跨域**标量**参数，统一采用相对基线的乘法扰动（默认 0.8–1.2，即 ±20%），在归一化 [0,1] 空间抽样后线性映射回物理值。

| 类别 | 参数（短名） | 数量 |
| --- | --- | --- |
| 几何 geometry | neg_thickness, pos_thickness, sep_thickness, neg_particle_r, pos_particle_r, electrode_height | 6 |
| 孔隙/体积分数 porosity | neg_amf, pos_amf, neg_porosity, sep_porosity | 4 |
| 输运 transport | neg_diffusivity, pos_diffusivity, neg_conductivity, transference, neg_bruggeman | 5 |
| 动力学 kinetics | neg_alpha, pos_alpha | 2 |
| 热 thermal | h_transfer, neg_cp, neg_thermal_k, cooling_area, ambient_t | 5 |
| 初始浓度 initial | c0_elyte, c0_neg | 2 |

**c0_neg 安全上界处理（关键）**：逐参数极端值扫描发现，负极初始浓度 c0_neg 上界 ×1.2 会导致 t=0 求解失败。其基线值 29866 mol·m⁻³ 已达负极最大浓度 33133 mol·m⁻³ 的 90.14%，安全上界因子仅约 1.109。平台通过逐参数 `FACTOR_OVERRIDES` 将 c0_neg 单独收窄为 **(0.8, 1.10)**，其余参数保持 (0.8, 1.2)。修复后 20 个随机点三工况零失败。

---

## 4. 抽样设计

| 方法 | 配置 | 设计行数 | 构成 |
| --- | --- | --- | --- |
| Morris | num_levels=4，生成 60 条轨迹，optimal_trajectories=20 选最优 | **500** | 20 × (24+1) |
| Saltelli | base_n=256（2⁸，保证 Sobol 低差异序列收敛性），calc_second_order=False | **6656** | 256 × (24+2) |

Saltelli 采用交错（interleaved）布局：每个 base 点占一个 step=D+2=26 行的块，块内依次为 A、AB_0…AB_{23}、B；子样本提取按完整 base 点块切分，以保证嵌套子样本仍符合分析器的布局假设。

---

## 5. 仿真执行与失败率

- 执行方式：`ProcessPoolExecutor` 4 进程并行（Windows spawn 兼容，每 worker 独立构建模型），主进程单线程写 CSV，无写竞争。
- 断点续跑：读取已存在 index 跳过，单次失败不中断整体流程。
- 总仿真量：Morris 3×500 = 1500 + Saltelli 3×6656 = 19968 = **21468**。

| 方法 | 工况 | 失败/总数 | 失败率 |
| --- | --- | --- | --- |
| Morris | s1 | 1/500 | 0.20% |
| Morris | s2 | 20/500 | 4.00% |
| Morris | s3 | 6/500 | 1.20% |
| Saltelli | s1 | 7/6656 | 0.11% |
| Saltelli | s2 | 26/6656 | 0.39% |
| Saltelli | s3 | 28/6656 | 0.42% |
| **合计** | — | **88/21468** | **0.41%** |

总用时 4765.6 s（约 79 分钟）。失败率远低于 5% 的验收阈值。Morris s2 失败率相对较高（4%），源于轨迹设计在高倍率边界的采样更激进；Saltelli 因样本均匀分布，各工况失败率均显著更低。失败点在分析阶段以列均值/中位填充，不影响指标计算。

---

## 6. Morris 筛选结果

Morris μ* 排序与 Sobol ST 高度一致，且 σ 揭示了非线性/交互特征：

- **s1 容量**：electrode_height μ* 最高，其次 neg_amf、neg_thickness、c0_neg，均为决定材料总量的参数。
- **s3 负极电位**：c0_neg μ*=0.51 明显居首，electrode_height、neg_amf、neg_thickness 次之；这些参数同时具有较高 σ，表明其效应存在明显非线性/交互。
- **s2 温度**：ambient_t 单点突出，其余参数贴近原点。
- 大量参数（输运、动力学类）聚集在 μ*≈0、σ≈0 的左下角，为不敏感参数。

---

## 7. Sobol 方差分解结果（定量）

### 7.1 s1 · 1C 放电容量（前 6，含 95% CI）

| 参数 | S1 | S1_conf | ST | ST_conf |
| --- | --- | --- | --- | --- |
| electrode_height | 0.364 | 0.091 | **0.340** | 0.069 |
| neg_thickness | 0.154 | 0.071 | **0.229** | 0.052 |
| neg_amf | 0.167 | 0.076 | **0.220** | 0.048 |
| c0_neg | 0.114 | 0.058 | **0.141** | 0.030 |
| pos_thickness | 0.046 | 0.059 | 0.090 | 0.029 |
| pos_amf | 0.043 | 0.062 | 0.079 | 0.029 |

几何与活性材料参数的 ST 普遍高于 S1，说明它们通过电流密度等途径存在较强交互。

### 7.2 s2 · 2C 快充最高温度（前 3）

| 参数 | S1 | S1_conf | ST | ST_conf |
| --- | --- | --- | --- | --- |
| ambient_t | 0.948 | 0.143 | **0.959** | 0.127 |
| c0_neg | 0.035 | 0.036 | 0.042 | 0.008 |
| pos_amf | 0.008 | 0.016 | 0.013 | 0.015 |

环境温度通过散热温差近乎单独决定热平衡，是快充热管理的首要外部条件。

### 7.3 s3 · 负极表面电位最小值（前 4）

| 参数 | S1 | S1_conf | ST | ST_conf |
| --- | --- | --- | --- | --- |
| c0_neg | 0.626 | 0.170 | **0.913** | 0.156 |
| neg_thickness | 0.072 | 0.084 | 0.207 | 0.079 |
| electrode_height | 0.029 | 0.071 | 0.197 | 0.078 |
| neg_amf | 0.078 | 0.086 | 0.183 | 0.073 |

c0_neg 决定初始嵌锂态（SOC 起点），初始浓度越高、负极越接近饱和，表面电位越易降至 0 以下，是析锂风险的首要内在参数；其 ST（0.91）远高于 S1（0.63），表明与几何参数存在强交互。

---

## 8. 参数图谱（ST 矩阵）

全阶 ST 矩阵（参数 × 工况，完整数据见 `results/st_matrix_all.csv`，可视化见 `st_heatmap_all.png`）给出跨工况的参数影响全景：

| 参数 | s1 容量 | s2 温度 | s3 电位 |
| --- | --- | --- | --- |
| electrode_height | 0.340 | 0.009 | 0.197 |
| neg_thickness | 0.229 | 0.003 | 0.207 |
| neg_amf | 0.220 | 0.001 | 0.183 |
| c0_neg | 0.141 | 0.042 | **0.913** |
| pos_thickness | 0.090 | 0.002 | 0.020 |
| pos_amf | 0.079 | 0.013 | 0.010 |
| ambient_t | 0.000 | **0.959** | 0.009 |
| neg_alpha / pos_alpha / neg_thermal_k | 0 | 0 | 0 |
| 其余参数 | <0.01 | <0.01 | <0.01 |

注：同一工况 ST 之和可能大于 1，因全阶指标包含交互作用且部分几何参数经电流密度耦合而被重复计入，属正常现象。

---

## 9. 不敏感参数白名单

满足"在所有工况主输出上 ST < 0.05"的全局白名单共 **17 个**（见 `results/whitelist_global.csv`、`whitelist.png`）：

```
c0_elyte, cooling_area, h_transfer, neg_alpha, neg_bruggeman, neg_conductivity,
neg_cp, neg_diffusivity, neg_particle_r, neg_porosity, neg_thermal_k, pos_alpha,
pos_diffusivity, pos_particle_r, sep_porosity, sep_thickness, transference
```

工程含义：在 ±20% 标定不确定度内，这些参数对所选三类工况的主输出影响可忽略，可在后续模型精简、参数标定优先级排序或降阶建模中降低其权重；但应注意该结论**仅对本报告的工况与输出成立**，换用工况（如长循环老化、脉冲工况）可能改变结论。

---

## 10. 收敛性与置信区间

- **收敛性**：4 个子样本（N=64→256）上，主导参数（c0_neg、ambient_t、electrode_height 等）在 N=128 之后 ST 即趋于平稳，说明 base_n=256 足以稳定主要结论；次要参数在小样本下波动但始终接近 0。
- **置信区间**：主导参数 ST 的 95% CI 宽度适中且不跨 0（如 s3 c0_neg ST=0.91±0.16、s2 ambient_t ST=0.96±0.13），结论稳健；不敏感参数 CI 普遍跨 0，与其"无显著效应"一致。
- 过程中修复了两个收敛分析缺陷：① Saltelli 子样本按交错 base 点块正确切分；② 收敛分析先填充失败点 NaN，与主分析口径统一。

---

## 11. 物理结论与工程建议

1. **容量设计**：放电容量主要由电极几何（高度、厚度）与正负极活性材料分数、初始锂浓度决定，提升容量应优先优化活性材料负载与材料总量，而非动力学/输运微调。
2. **快充热安全**：快充温升几乎完全取决于环境温度，改善散热条件（环境/冷却）是控制快充温度最直接的手段；h_transfer、cooling_area 在当前集总热设置与扰动范围内影响有限。
3. **析锂风险**：负极初始浓度 c0_neg 是析锂指示量的首要内在因素，与负极厚度、电极高度、活性材料分数强交互；控制初始 SOC/负极嵌锂上限、优化负极几何是缓解析锂的关键。
4. **参数标定优先级**：动力学电荷转移系数与负极热导率在所选工况下不敏感，可降低标定优先级；资源应集中于几何、活性材料分数、初始浓度与环境温度。

---

## 12. 局限性

- 结论限定于 Chen2020 参数集、3 个单循环工况及其指定标量输出；未覆盖长循环老化、SEI 增长、脉冲/混合工况等。
- 集总热模型无法刻画空间温度梯度，热参数（h_transfer、cooling_area、neg_thermal_k）的影响可能被低估。
- 参数采用统一独立的 ±20% 乘法扰动，未引入参数间真实相关性；c0_neg 上界受物理可行性限制而收窄。
- 少量失败点（0.41%）以统计填充处理，虽比例极低但仍属近似。
- 本分析为物理仿真层面的敏感性结论，不直接等同于真实电芯的析锂检测准确率。

---

## 13. 复现指南

### 13.1 环境

默认 Python 3.13，依赖见 `requirements.txt`（numpy、scipy、pandas、scikit-learn、SALib、matplotlib、pybamm、casadi、pybammsolvers 等）。

```powershell
python -m pip install -r requirements.txt
```

### 13.2 一键流程（config 驱动）

```powershell
# 完整流程：抽样 -> 仿真(并行+断点续跑) -> 分析出图
python run_gsa.py all --config configs/gsa_config.yaml

# 或分步执行
python run_gsa.py sample                 # 生成 Morris/Saltelli 设计
python run_gsa.py run                    # 运行全部工况（可加 --method/--scenario 过滤）
python run_gsa.py analyze                # 计算指标、CI、收敛、白名单并出图
```

断点续跑：仿真中断后重新执行 `run` 命令即可自动跳过已完成 index。

### 13.3 脚本入口（与平台等价）

- `scripts/run_all_simulations.py`：全量仿真。
- `scripts/analyze_all.py`：全量分析与 22 张图生成（运行时设 `PYTHONPATH=src`）。

---

## 14. 产物文件清单

| 位置 | 内容 |
| --- | --- |
| `configs/gsa_config.yaml` | 模型、工况、抽样、统计、运行全部配置 |
| `src/p2d_gsa/` | parameters、model、sampling、runner、analysis、plotting 模块 |
| `results/samples/` | 设计矩阵 design_morris/saltelli.npy、参数空间表、6 个工况运行 CSV |
| `results/` | Morris/Sobol 指标 CSV、收敛 CSV、ST 矩阵、白名单（全局+分工况） |
| `results/figures/` | 22 张 PNG（Morris bar/scatter、Sobol ST/S1-vs-ST、收敛、ST 热力图、白名单） |
| `reports/` | 本报告 |
