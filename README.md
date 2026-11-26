# 面向铜熔炼配矿逆向设计的 TabPFN 代理模型

> A TabPFN surrogate model for inverse design of copper smelting charge blends

用 **TabPFN-3**（Nature 2025 的预训练表格基础模型）替代 FactSage 热力学计算：
输入 16 个配料/工况参数，预测 29 个冶金输出（冰铜、渣、气相成分与各相质量）；
再用 **SHAP** 解释模型、用 **NSGA-II** 在配比空间里求低渣含铜的配料方案。

**适用范围：固定工况。** 数据集为 3000 组、温度 1200 ℃ / 压力 1 atm / 供氧 21 g。
温度与氧气在数据中是常量，因此：

- 代理模型只学「配料 → 冶金结果」；
- 配矿优化只求 **11 种矿料的质量配比 + SiO₂ 熔剂比例（12 维）**，温度和氧气固定不变。

<div align="center">

| 环节 | 方法 | 产出 |
|---|---|---|
| 数据 | FactSage 8.3 平衡计算（1200 ℃ / 1 atm / 21 g O₂） | 16 输入 → 29 输出 |
| 代理模型 | TabPFN-3，逐目标拟合（in-context learning，无梯度下降） | R² / MAE / RMSE + 真实值-预测值散点 |
| 可解释性 | SHAP / shapiq（tabpfn-extensions） | 特征重要性、蜂群图、热力图 |
| 逆向设计 | NSGA-II（pymoo），12 维配比 | Pareto 前沿 + 人工方案对比 |

</div>

## 仓库说明

**本仓库只包含代码。** 数据集、FactSage 原始计算结果、训练好的模型权重
（`.ckpt` / `.tabpfn_fit`）和论文文档均不入库，见 [`.gitignore`](.gitignore)。
运行前请按下面的「数据准备」自行放置数据。

## 目录结构

```text
.
├── ml_pipeline/                    # 多步骤流水线（统一由 config.py 驱动）
│   ├── config.py                   #   唯一配置入口：数据集、模型、划分、SHAP
│   ├── common.py                   #   公共模块：路径解析、数据加载、绘图样式
│   ├── run_all.ps1                 #   一键按顺序跑多个步骤
│   ├── show_menu.py                #   run.bat 的菜单文字
│   ├── step1_train_tabpfn.py       #   ① TabPFN 逐目标训练 + 预测 + 评估
│   ├── step3_multiseed_evaluation.py  # ③ 多次随机划分重复评价
│   ├── step4_shap_all_targets.py   #   ④ 全目标 SHAP 特征重要性
│   ├── step5_shap_key_targets.py   #   ⑤ 四项关键指标 SHAP 论文图
│   ├── step5b_shap_signed.py       #   ⑤b 有符号 SHAP（蜂群图 / waterfall / 依赖图）
│   ├── step6_learning_curve.py     #   ⑥ 学习曲线（训练量 vs R²）
│   └── step7_make_figures.py       #   ⑦ 论文风格图表汇总
│
├── scripts/                        # 单文件脚本（每个脚本顶部集中放参数）
│   ├── training/
│   │   └── train_tabpfn.py         # TabPFN 训练/预测/评估（最小可用示例）
│   ├── analysis/
│   │   ├── shap_analysis.py        # SHAP 特征重要性
│   │   ├── plot_shap_key_targets_sci.py  # 四关键指标 SCI 风格 SHAP 图
│   │   ├── plot_learning_curve.py  # 学习曲线
│   │   └── predict_manual_blend.py # 人工配矿方案预测
│   └── optimization/
│       ├── nsga2_optimization.py   # ★ 配矿优化：固定 T/O2，12 维（11 配比 + SiO2 熔剂）
│       ├── calc_full_blend_composition.py  # 优化方案完整成分计算
│       ├── plot_optimization.py    # 优化结果图表
│       ├── plot_pareto_front_en.py # Pareto 前沿（英文版）
│       └── plot_manual_vs_optimized_en.py  # 人工 vs 优化对比（英文版）
│
└── tests/                          # 无需 GPU / 数据集的单元测试
    └── test_nsga2_feature_order.py
```

## 环境

Python 3.12 + CUDA 11.8（本项目实测环境）：

```bash
pip install -r requirements.txt
```

### TabPFN v3 是门控模型

先在 HuggingFace 上接受 TabPFN 许可，然后通过**环境变量**提供 token（不要写进代码）：

```powershell
# Windows PowerShell
$env:TABPFN_TOKEN = "tabpfn_sk_..."
$env:HF_ENDPOINT  = "https://hf-mirror.com"   # 国内加速，可省略
```

```bash
# Linux / macOS
export TABPFN_TOKEN=tabpfn_sk_...
export HF_ENDPOINT=https://hf-mirror.com
```

预训练权重默认 `model_path="auto"`：首次运行时自动下载到本地缓存（约 2.5 GB，不入库）。
若要复现特定权重，把 `ml_pipeline/config.py` 的 `MODEL["path"]` 或脚本顶部的
`MODEL_PATH` / `TABPFN_MODEL_PATH` 改成你的本地路径即可。

## 数据准备

仓库不含数据。请自行放置：

```text
data/
├── 3000_fixed_temp/
│   └── ml_dataset.csv      # 必需：3000 组固定工况数据（1200 ℃ / 1 atm / 21 g O2）
├── 矿源.xlsx                # 可选：11 种矿料的化学组成
└── 人工配矿.xlsx            # 可选：人工配矿记录（predict_manual_blend.py 用）
```

`ml_dataset.csv` 为 utf-8-sig 编码，共 **45 列**（16 输入 + 29 输出）、
每行一组 FactSage 平衡计算结果。

**16 个输入特征**

```text
T(℃)  P(atm)  Cu(g)  S(g)  Fe(g)  SiO2(g)  Pb(g)  Zn(g)  MgO(g)
Al2O3(g)  Ni(g)  Sb(g)  Bi(g)  CaO(g)  As(g)  O2(g)
```

（固定工况数据集里 `T(℃)`=1200、`P(atm)`=1、`O2(g)`=21 为常量，其余 13 项是配料质量。）

**29 个输出目标**

| 相 | 列名 | 单位 |
|---|---|---|
| 冰铜 | `matte_Cu_pct` `matte_Fe_pct` `matte_S_pct` `matte_Ni_pct` `matte_Zn_pct` `matte_Pb_pct` `matte_As_pct` | wt.% |
| 冰铜 | `matte_mass_g` | g |
| 渣 | `slag_Cu_pct` `slag_FeO_pct` `slag_SiO2_pct` `slag_Al2O3_pct` `slag_CaO_pct` `slag_MgO_pct` `slag_ZnO_pct` | wt.% |
| 渣 | `slag_FeO_SiO2_ratio` | 质量比 |
| 渣 | `slag_mass_g` | g |
| 气相 | `gas_SO2_mol` `gas_S2_mol` `gas_SO_mol` `gas_SSO_mol` `gas_SO3_mol` `gas_Zn_mol` `gas_PbS_mol` `gas_CuS_mol` `gas_Pb_mol` `gas_Sb_mol` `gas_As_mol` | mol 分数 |
| 气相 | `gas_mass_g` | g |

> 注：渣中 Cu 由 Cu₂O 换算为 Cu 元素 wt.%（×127.09/143.09）；气相成分是摩尔分数。

## 快速开始

### 最小示例：训练 + 预测 + 评估

```bash
python scripts/training/train_tabpfn.py
```

逐目标拟合 29 个 TabPFN 模型，输出 `evaluation_results.csv`（R²/MAE/RMSE）、
模型文件和真实值-预测值散点图。TabPFN 的 `fit()` 不做梯度下降，
只把训练数据编码进 Transformer 上下文，因此 3000 组样本分钟级即可完成。

```python
from tabpfn import TabPFNRegressor

reg = TabPFNRegressor(device="cuda", model_path="auto", ignore_pretraining_limits=True)
reg.fit(X_train, y_train)        # 无梯度下降，纯 in-context
y_pred = reg.predict(X_test)
```

### 完整流水线（训练 + SHAP + 出图）

```bash
# 默认跑步骤 1,7（训练 + 出图）
powershell -ExecutionPolicy Bypass -File ml_pipeline/run_all.ps1

# 只跑训练 + 四关键指标 SHAP
powershell -ExecutionPolicy Bypass -File ml_pipeline/run_all.ps1 -Steps 1,5

# 换结果目录
powershell -ExecutionPolicy Bypass -File ml_pipeline/run_all.ps1 -Run 复核
```

单步运行：

```bash
python ml_pipeline/step1_train_tabpfn.py        # 训练 + 预测
python ml_pipeline/step1_train_tabpfn.py --targets key   # 只跑 4 个关键目标
python ml_pipeline/step4_shap_all_targets.py    # SHAP
```

所有步骤的结果都落在 `<root>/<步骤>/<run>/` 下，`root` 与 `run` 由 `config.py` 决定。

### 配矿优化（固定温度与氧气，只求配比）

```bash
python scripts/optimization/nsga2_optimization.py     # NSGA-II，12 维
python scripts/optimization/calc_full_blend_composition.py   # 反算混合矿成分
python scripts/optimization/plot_pareto_front_en.py   # Pareto 前沿
python scripts/optimization/plot_manual_vs_optimized_en.py   # 人工 vs 优化
```

决策变量与约束：

```text
固定工况  T = 1200 ℃      O2 = 21 g          （T_DEFAULT / O2_DEFAULT）
决策变量  x[0:11]  11 种矿料的质量配比（内部归一化，sum = 1）
          x[11]    SiO2 熔剂比例（每 1 g 矿料额外添加的 g 数）
目标      min slag_Cu_pct（渣含铜）
          min |matte_Cu_pct - 60|（冰铜品位 60 wt%）
硬约束    slag_FeO_SiO2_ratio ∈ [1.15, 1.30]
          混合炉料 Cu/S/Fe/SiO2 落在训练数据覆盖范围内（避免外推）
```

参数全部集中在脚本顶部的「参数配置区」（`T_DEFAULT`、`O2_DEFAULT`、`FESIO2_LO`、
`FESIO2_HI`、`MATTE_CU_TARGET`、`COMP_RANGES`），改完直接重跑。

## 方法要点

### 1. 逐目标拟合的代理模型

29 个输出量纲差异极大（wt.%、质量 g、摩尔分数），回归性能与可分性都不同，
因此**每个目标单独训练一个 TabPFN 回归器**，而不是多输出联合建模。
`step1` 会为每个目标记录 R² / MAE / RMSE 并单独保存模型。

大样本预测会 OOM，`predict_batched()` 按 `BATCH_SIZE` 分批推理；
`ml_pipeline` 在显存不足时自动折半重试，下限为 `predict_batch_floor`。

### 2. SHAP 可解释性

`ml_pipeline/step4` 与 `scripts/analysis/shap_analysis.py` 用
`tabpfn_extensions.interpretability.shapiq` 计算 Shapley 值，再转成标准的
`shap.Explanation` 对象，从而复用 SHAP 的蜂群图 / waterfall / 依赖图。
`BUDGET` 参数控制精度：16 个特征精确计算需 65536 次前向（2¹⁶），256 为近似值。

四项关键冶金指标：`matte_Cu_pct`、`slag_Cu_pct`、`slag_FeO_pct`、`slag_SiO2_pct`。

> 固定工况下 `T(℃)`、`P(atm)`、`O2(g)` 是常量，它们在 SHAP 里的贡献恒为 0，
> 有效特征只有 13 项配料——这是数据集本身的属性，不是程序 bug。

### 3. NSGA-II 逆向设计

双目标（渣含铜 ↓、冰铜品位 → 60）+ 两个不等式约束（FeO/SiO₂ 区间、炉料成分覆盖范围）。
种群 100、最多 200 代、连续 30 代无改善则早停。Pareto 前沿上再按
「FeO/SiO₂ 最接近中值」挑一个代表解，用于和人工经验方案对比。

## 配置说明

`ml_pipeline/config.py` 是普通 Python 模块：

```python
ACTIVE_DATASET = "3000_fixed_temp"
RUN = "single_split_42"
```

| 配置段 | 作用 |
|---|---|
| `DATASETS` | 数据文件路径与结果根目录，可写同名子字典做专属覆盖 |
| `SPLIT` | 测试集比例、随机种子、多划分种子列表 |
| `MODEL` | token、镜像、权重路径、设备、集成数、预测批量 |
| `CACHE_ACCELERATION` | 各步骤是否启用 KV 缓存（`fit_with_cache`） |
| `SHAP` | 目标、背景样本数、被解释样本数、预算 |
| `LEARNING_CURVE` | 训练子集规模列表 |
| `MISC` | 中文字体候选 |

> ⚠️ `fit_with_cache` 会预计算训练上下文的 K/V，快很多，但**与默认的
> `fit_preprocessors` 不是数值等价的**（R² 在第 4–6 位小数上不同）。
> 要进论文表格的步骤（3、6）默认关闭，只有 SHAP 默认开启——详见 `config.py` 第 5 节。

## 测试

```bash
python -m unittest discover tests
```

`tests/test_nsga2_feature_order.py` 校验 `ratios_to_features()` 的列顺序与训练输入一致
（这类顺序错位不会报错，只会静默给出错误预测，所以单独加了测试）。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `GatedRepoError` / 401 | 未接受 TabPFN 许可，或未设置 `TABPFN_TOKEN` |
| 下载权重超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| CUDA OOM | 调小 `MODEL["predict_batch_size"]`（默认自动折半重试） |
| 中文变方块 | matplotlib 只用字体列表里**第一个可用**字体；`MISC["fonts"]` 把 `Microsoft YaHei` 放首位。下标 ₂ 连 YaHei 都没有，图里请写 mathtext `SiO$_2$` |
| Windows 下 GBK 编码错 | `PYTHONIOENCODING=utf-8`；`conda run -n <env> --no-capture-output` |
| `scripts/` 下脚本找不到路径 | 这些脚本用**相对项目根目录**的路径，必须在项目根目录运行 |
| 优化跑完没有可行解 | 放宽 `FESIO2_HI/LO` 或 `COMP_RANGES`，看脚本打印的约束诊断 |

## 引用

如果本仓库的代码对你有帮助，请引用 TabPFN 原文：

```bibtex
@article{tabpfn2025,
  title   = {Accurate predictions on small data with a tabular foundation model},
  author  = {Hollmann, Noah and others},
  journal = {Nature},
  year    = {2025}
}
```

本项目为作者博士研究课题的一部分，论文发表后此处补充完整引用信息。

## License

代码部分见 [LICENSE](LICENSE)。注意 TabPFN 模型权重遵循其自身的许可协议
（HuggingFace 门控模型），本仓库不包含权重。
