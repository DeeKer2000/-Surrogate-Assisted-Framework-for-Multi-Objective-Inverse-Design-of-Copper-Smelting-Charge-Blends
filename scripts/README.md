# scripts — 单文件实验脚本（固定工况 / 3000 组数据）

每个脚本都可以单独运行，**参数集中在文件顶部的「参数配置区」**，改完直接跑，不用往下翻代码。

所有命令都在**项目根目录**执行（脚本内的相对路径按项目根解析）。

```bash
conda run -n tabpfn python scripts/training/train_tabpfn.py
```

> Windows 上含中文输出时加 `--env PYTHONIOENCODING=utf-8`，否则控制台按 GBK 解码会报错。

## training/

| 脚本 | 用途 |
|---|---|
| `train_tabpfn.py` | 29 个目标逐个拟合 TabPFN 回归器，留出法评估，保存模型与 R²/MAE/RMSE |

这是最小可用示例：读 CSV → 划分 → 逐目标 `fit`/`predict` → 评估 → 存模型 + 散点图。

## analysis/

| 脚本 | 用途 |
|---|---|
| `shap_analysis.py` | 用 shapiq 计算 Shapley 值并转成 `shap.Explanation`，输出特征重要性 CSV + 热力图 |
| `plot_shap_key_targets_sci.py` | 四项关键指标（`matte_Cu_pct`、`slag_Cu_pct`、`slag_FeO_pct`、`slag_SiO2_pct`）的 SCI 风格 SHAP 图 |
| `plot_learning_curve.py` | 固定测试集、嵌套训练子集规模下的 R² 曲线 |
| `predict_manual_blend.py` | 读人工配矿记录，取年度平均成分，预测其冶金结果（替代人工经验估算） |

## optimization/

| 脚本 | 用途 |
|---|---|
| `nsga2_optimization.py` | **配矿优化本体**：NSGA-II 双目标，12 维（11 矿料配比 + SiO₂ 熔剂） |
| `calc_full_blend_composition.py` | 由优化方案的配比反算混合炉料完整成分 |
| `plot_optimization.py` | 从优化结果 CSV 生成图表 |
| `plot_pareto_front_en.py` | Pareto 前沿（英文版，论文用） |
| `plot_manual_vs_optimized_en.py` | 人工经验方案 vs 优化方案对比（英文版，论文用） |

### 优化问题定义（固定温度与氧气）

```text
固定工况  T = 1200 ℃      O2 = 21 g        （T_DEFAULT / O2_DEFAULT）
决策变量  x[0:11]  11 种矿料的质量配比（内部归一化，sum = 1）
          x[11]    SiO2 熔剂比例（每 1 g 矿料额外添加的 g 数）
目标      min slag_Cu_pct
          min |matte_Cu_pct - 60|
硬约束    slag_FeO_SiO2_ratio ∈ [1.15, 1.30]
          混合炉料 Cu/S/Fe/SiO2 落在训练数据覆盖范围内（避免外推）
```

温度与氧气不进决策变量——本数据集里它们本来就是常量，只求配比。

### 脚本之间的依赖

```text
data/3000_fixed_temp/ml_dataset.csv
      ↓ train_tabpfn.py
experiments/3000_fixed_temp/tabpfn/<时间戳>/saved_models/*.tabpfn_fit
      ↓                    ↓
shap_analysis.py      nsga2_optimization.py
      ↓                    ↓
plot_shap_key_targets_sci.py   plot_optimization.py
                               calc_full_blend_composition.py
                               plot_pareto_front_en.py
                               plot_manual_vs_optimized_en.py
```

`predict_manual_blend.py` 与 `nsga2_optimization.py` 都直接复用
`train_tabpfn.py` 保存的 `.tabpfn_fit` 模型，因此必须先跑训练。
它们在脚本顶部用 `MODELS_DIR` / `MODEL_PATH` 两个常量指定要加载哪一次实验的模型。
