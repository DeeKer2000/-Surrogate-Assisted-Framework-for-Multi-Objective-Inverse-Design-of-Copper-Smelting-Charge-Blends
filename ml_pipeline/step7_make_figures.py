"""步骤7：生成论文风格图表并汇总本次运行的清单。

依赖步骤1（TabPFN 预测）的输出。

用法：
    python ml_pipeline/step7_make_figures.py

产出：
    <结果根目录>/figures/fig2_四项关键指标预测值真实值散点图.png/pdf/svg
    <结果根目录>/figures/fig2_散点图数据.csv
    <结果根目录>/figures/fig4_各目标R2条形图.png
    <结果根目录>/figures/fig5_各目标真实值预测值散点矩阵.png
    <结果根目录>/运行汇总.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import KEY_TARGETS, OUTPUT_COLS, TARGET_LABELS, banner, load_config, setup_console

setup_console()


def scatter_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    lower, upper = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    padding = max((upper - lower) * 0.045, 1e-8)
    return lower - padding, upper + padding


def plot_figure2(cfg, metrics: pd.DataFrame) -> pd.DataFrame:
    """图2：四项关键指标 真实值-预测值散点（单排四联）。"""
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.linewidth": 0.8,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    r2_map = metrics.set_index("target")["R2"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.15), constrained_layout=True)
    pooled: list[pd.DataFrame] = []

    # 逐目标试跑（--targets 只跑了一部分）时，缺哪个目标就跳过哪个面板，不中断出图。
    available = [t for t in KEY_TARGETS if (cfg.predictions_dir / f"{t}.npz").exists()]
    missing = [t for t in KEY_TARGETS if t not in available]
    if missing:
        print(f"注意: 缺少 {len(missing)} 个关键指标的预测文件，图2 只画已有的 "
              f"{len(available)} 个面板: {', '.join(missing)}", flush=True)
    if not available:
        raise FileNotFoundError(
            f"{cfg.predictions_dir} 里一个关键指标的预测文件都没有，"
            f"请先运行 step1_train_tabpfn.py --targets key"
        )

    for panel, (ax, target) in enumerate(zip(axes.flat, available)):
        archive = np.load(cfg.predictions_dir / f"{target}.npz")
        y_true = archive["y_true"].astype(float)
        y_pred = archive["y_pred"].astype(float)
        limits = scatter_limits(y_true, y_pred)

        ax.scatter(y_true, y_pred, s=9, alpha=0.30, color="#4F4A85",
                   edgecolors="none", rasterized=True)
        ax.plot(limits, limits, linestyle="--", linewidth=1.0, color="#777777")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{TARGET_LABELS.get(target, target)} (wt%)", fontweight="bold", pad=6)
        ax.set_xlabel("FactSage value")
        ax.set_ylabel("TabPFN-3 prediction")
        ax.grid(True, color="#E2E2E2", linewidth=0.55, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        ax.text(0.04, 0.95, rf"$R^2$ = {r2_map[target]:.5f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
        ax.text(-0.15, 1.08, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

        pooled.append(pd.DataFrame({
            "target": target,
            "test_index": archive["test_indices"].astype(int),
            "factsage_value": y_true,
            "tabpfn_prediction": y_pred,
        }))

    for ax in list(axes.flat)[len(available):]:      # 隐藏没用到的面板
        ax.set_visible(False)

    base = cfg.figures_dir / "fig2_四项关键指标预测值真实值散点图"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    data = pd.concat(pooled, ignore_index=True)
    data.to_csv(cfg.figures_dir / "fig2_散点图数据.csv", index=False, encoding="utf-8-sig")
    print(f"图2 已生成: {base.name}.png/pdf/svg", flush=True)
    return data


def copy_step1_figures(cfg) -> None:
    import shutil
    pairs = [
        ("metrics_r2_all_targets.png", "fig4_各目标R2条形图.png"),
        ("scatter_true_vs_pred.png", "fig5_各目标真实值预测值散点矩阵.png"),
    ]
    for source, target in pairs:
        origin = cfg.tabpfn_figures_dir / source
        if origin.exists():
            shutil.copy2(origin, cfg.figures_dir / target)
            print(f"汇总: {target}", flush=True)
        else:
            print(f"注意: 缺少 {origin}，请先运行步骤1。", flush=True)


def write_summary(cfg, metrics: pd.DataFrame) -> None:
    lines = [
        f"# {cfg.dataset} 运行汇总",
        "",
        "## 运行设置",
        "",
        f"- 配置文件：`{cfg.config_path}`",
        f"- 数据集：`{cfg.dataset}`（{cfg.dataset_label or '未填写说明'}）",
        f"- 数据文件：`{cfg.data_path}`",
        f"- 结果根目录：`{cfg.root}`",
        f"- 运行名称：`{cfg.run}`",
        f"- 划分：测试比例 {cfg.test_size}，随机种子 {cfg.random_state}，"
        f"训练 {int(metrics['n_train'].iloc[0])} / 测试 {int(metrics['n_test'].iloc[0])}",
        f"- 模型：TabPFN-3（`{cfg.model_label}`），逐目标独立拟合，"
        f"共 {len(metrics)}/{len(OUTPUT_COLS)} 个目标",
        "",
        "## 图表",
        "",
        "| 文件 | 对应论文 | 来源 |",
        "|---|---|---|",
        "| `figures/fig2_四项关键指标预测值真实值散点图.png/pdf/svg` | 图2 | 步骤1 留出预测 |",
        "| `figures/fig2_散点图数据.csv` | 图2 数据 | 步骤1 留出预测 |",
        "| `figures/fig4_各目标R2条形图.png` | 补充材料 | 步骤1 |",
        "| `figures/fig5_各目标真实值预测值散点矩阵.png` | 补充材料 | 步骤1 |",
        "| `shap/*/fig_shap_key_targets_bars.png/pdf/svg` | 图5 | 步骤5 四关键指标 SHAP 排序 |",
        "| `shap/*/fig_shap_key_targets_heatmap_13_features.png/pdf/svg` | 图6 | 步骤5 13 项炉料特征热力图 |",
        "| `learning_curve/*/fig_learning_curve.png` | 图4 | 步骤6 学习曲线 |",
        "| `optimization/figures/fig7_pareto_front_en.png/pdf/svg` | 图7 | scripts/optimization 配矿优化 |",
        "| `optimization/figures/fig8_manual_vs_optimized_en.png/pdf/svg` | 图8 | 优化 + 人工配矿预测 |",
        "",
        "## 四项关键指标（TabPFN-3）",
        "",
        "| 目标 | R² | RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    indexed = metrics.set_index("target")
    for target in KEY_TARGETS:
        if target in indexed.index:
            row = indexed.loc[target]
            lines.append(f"| {target} | {row['R2']:.5f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |")

    lines += [
        "",
        "## 复现命令",
        "",
        "```powershell",
    ]
    for step in ("step1_train_tabpfn", "step3_multiseed_evaluation",
                 "step4_shap_all_targets", "step5_shap_key_targets", "step5b_shap_signed",
                 "step6_learning_curve", "step7_make_figures"):
        lines.append(f"conda run -n tabpfn --no-capture-output python ml_pipeline/{step}.py")
    lines.append("# 配矿优化（固定 1200 ℃ / 21 g O2，12 维）")
    lines.append("conda run -n tabpfn --no-capture-output python scripts/optimization/nsga2_optimization.py")
    lines += [
        "```",
        "",
        "本文件与全部图表由 ml_pipeline 程序生成；程序不修改论文 Word 文件。",
    ]
    (cfg.root / "运行汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"运行汇总: {cfg.root / '运行汇总.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文风格图表")
    parser.add_argument("--no-copy-step1", action="store_true", help="不复制步骤1的图")
    known, _ = parser.parse_known_args()

    cfg = load_config()
    cfg.ensure_dirs()
    banner("步骤7  论文风格图表", cfg)

    metrics_path = cfg.tabpfn_dir / "evaluation_results.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少 {metrics_path}，请先运行 step1_train_tabpfn.py")
    metrics = pd.read_csv(metrics_path)
    metrics.to_csv(cfg.figures_dir / "各目标预测性能汇总.csv", index=False, encoding="utf-8-sig")

    average_path = cfg.baselines_dir / "per_algorithm_average.csv"
    if average_path.exists():
        print(f"注意: 发现 {average_path}，但本仓库不含基线对比脚本，图3 不做处理。", flush=True)

    plot_figure2(cfg, metrics)
    if not known.no_copy_step1:
        copy_step1_figures(cfg)

    write_summary(cfg, metrics)
    print(f"\n步骤7 完成。图表目录: {cfg.figures_dir}")


if __name__ == "__main__":
    main()
