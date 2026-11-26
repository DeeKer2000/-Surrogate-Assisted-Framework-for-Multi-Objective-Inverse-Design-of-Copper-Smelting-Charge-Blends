# ml_pipeline — 统一实验流水线（固定工况 / 3000 组数据）

一套程序 + 一个配置文件，跑完代理模型的训练、评价、解释与出图。

**换数据集只改 `config.py` 一行，程序一行都不用动。**

原始的 `scripts/` 目录是单文件脚本，各自写死路径；本目录是改进后的统一版本：
所有路径、参数集中在 `config.py`，命令行可以临时覆盖。

> 本仓库面向**固定工况数据集**：3000 组、1200 ℃ / 1 atm / 21 g O₂。
> 温度和氧气在数据里是常量，模型只学「配料 → 冶金结果」。
> 配矿优化不在这里（它是 12 维固定工况问题），见 `scripts/optimization/nsga2_optimization.py`。

---

## 1. 快速开始

最省事的方式是一次跑完（默认步骤 `1,7`：训练 + 出图）：

```powershell
powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1
```

按需挑选步骤、或临时换结果目录：

```powershell
powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Steps 1,5
powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Run 复核
powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Root experiments/_试跑
```

等价的手工逐步命令（在项目根目录、已激活含 tabpfn 的环境、已设 `TABPFN_TOKEN`）：

```bash
conda run -n tabpfn --no-capture-output python ml_pipeline/step1_train_tabpfn.py
conda run -n tabpfn --no-capture-output python ml_pipeline/step5_shap_key_targets.py
```

> `--no-capture-output` 不能省，否则 `conda run` 会把输出缓存到进程结束才吐出来，
> 长任务看着像卡死。

---

## 2. 步骤一览

| 步骤 | 脚本 | 做什么 | 典型耗时（3000 组） |
|---|---|---|---|
| 1 | `step1_train_tabpfn.py` | 29 个目标逐个拟合 + 留出预测 + R²/MAE/RMSE | 1–2 h（GPU） |
| 3 | `step3_multiseed_evaluation.py` | 5 次随机划分重复评价（mean±SD） | 数小时 |
| 4 | `step4_shap_all_targets.py` | 全目标 SHAP 特征重要性 + 热力图 | 数小时 |
| 5 | `step5_shap_key_targets.py` | 四关键指标 SHAP 论文图 | 1–2 h |
| 5b | `step5b_shap_signed.py` | 有符号 SHAP（蜂群图 / waterfall / 依赖图） | 1–2 h |
| 6 | `step6_learning_curve.py` | 学习曲线（训练量 vs R²） | 1–3 h |
| 7 | `step7_make_figures.py` | 论文风格图（散点图、各目标 R² 条形图） | 秒级 |

> 基线算法对比不在本仓库：那些脚本依赖 XGBoost / LightGBM / CatBoost，
> 属于对比实验而非本方法本身。
>
> 固定工况下温度与氧气是常量，所以没有「温度 × 供氧响应面」「联合逆向求解」
> 「耗氧量反算」这类步骤——它们只有在工况可变时才有意义。

---

## 3. 配置

`config.py` 是普通 Python 模块，可直接写表达式、列表和注释。

```python
ACTIVE_DATASET = "3000_fixed_temp"
RUN = "single_split_42"      # 同一数据集下区分不同轮次
```

各段作用：

| 配置段 | 内容 |
|---|---|
| `DATASETS` | 数据集的 `data`（数据文件）与 `root`（结果根目录）；可写同名子字典做专属覆盖 |
| `SPLIT` | `test_size`、`random_state`、`seeds`（多划分用） |
| `MODEL` | token、镜像、权重路径、设备、`n_estimators`、预测批量 |
| `CACHE_ACCELERATION` | 各步骤是否用 KV 缓存 |
| `SHAP` | 目标、背景样本数、被解释样本数、`budget` |
| `LEARNING_CURVE` | 训练子集规模 |
| `MISC` | 字体候选 |

命令行覆盖（任何步骤脚本都认）：

```bash
python ml_pipeline/step1_train_tabpfn.py --run 复核 --root experiments/_试跑
python ml_pipeline/step1_train_tabpfn.py --config ml_pipeline/config_试验.py
```

### 结果目录

所有产出都在 `<root>/` 之下，按步骤分子目录：

```text
<root>/
├── tabpfn/<run>/
│   ├── evaluation_results.csv
│   ├── saved_models/<target>.tabpfn_fit
│   ├── predictions/<target>.npz
│   └── figures/
├── multiseed/<run>/
├── shap/<run>/
├── learning_curve/<run>/
└── figures/
```

---

## 4. KV 缓存（重要）

TabPFN 的 `fit_mode="fit_with_cache"` 会把训练上下文的 K/V 预先算好，
之后每次预测都省掉重新编码，多目标 / 多划分时快很多。

**但它与默认路径 `fit_preprocessors` 不是数值等价的**——两条引擎路径的注意力归约顺序不同。
实测（3000 组、seed 42、`matte_Cu_pct`）：

```text
fit_preprocessors   R² = 0.9998369968128022   ← 与归档结果逐位一致
fit_with_cache      R² = 0.9998353989154523   ← 差 1.6e-06
```

对较难的目标差距更明显（`slag_Cu_pct`：0.89015 vs 0.88995，第 4 位小数可见）。
因此 `CACHE_ACCELERATION` 里：

- 要进论文表格的步骤（3、6）**关闭**；
- SHAP（4、5）**开启**——它每个特征联盟都要预测一次，不开慢十几倍，
  而特征重要性排序对 1e-4 量级差异不敏感。

想加速可在命令行加 `--fit-mode cache`，但会改变表格数值，请自行确认是否可接受。

---

## 5. 注意

- 所有脚本假设**从项目根目录**运行（相对路径按项目根解析），不要 `cd ml_pipeline`。
- 数据文件是 `utf-8-sig` 编码；中文输出建议 `PYTHONIOENCODING=utf-8`。
- 显存不足时预测批量会自动折半重试，下限由 `predict_batch_floor` 控制。
- 修改 `common.py` 的 `INPUT_COLS` / `OUTPUT_COLS` 会同时影响所有步骤，
  且与已训练的模型文件不再匹配——改列顺序等于换数据集。
