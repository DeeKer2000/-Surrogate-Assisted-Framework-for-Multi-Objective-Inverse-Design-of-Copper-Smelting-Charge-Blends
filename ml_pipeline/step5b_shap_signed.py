"""步骤5b：四项关键指标的**有符号** SHAP 分析（蜂群图 / 有向条形图 / waterfall / 依赖图）。

与步骤5（step5_shap_key_targets.py）的区别：
    步骤5 只保留 mean |SHAP|（丢了方向），出的是排序柱状图和热力图；
    本步骤保留**逐样本的有符号 SHAP 值**，因此能回答"某成分升高把指标推向哪一边"，
    并输出 shap 原生的 beeswarm / waterfall / scatter（依赖图）等图。
    两者用的是同一套 SHAP 计算口径（shapiq 一阶 Shapley 值、同一划分、同一背景/解释样本），
    所以数字可以和步骤5 对齐。

用法：
    python ml_pipeline/step5b_shap_signed.py --dataset 3000_fixed_temp
    python ml_pipeline/step5b_shap_signed.py --dataset 3000_fixed_temp --targets matte_Cu_pct
    python ml_pipeline/step5b_shap_signed.py --dataset 3000_fixed_temp --keep-constant
    python ml_pipeline/step5b_shap_signed.py --dataset 3000_fixed_temp --sample 3 --budget 512

模型来源（--source，默认取 config 的 shap.source）：与步骤4/5 相同，cache 为默认（快），
archive 为兜底（解释已保存的模型，很慢）。

固定工况数据集（3000_fixed_temp）里 T/P/O2 是常量，它们的有符号 SHAP 恒为 0，
默认从图里剔除（--keep-constant 可保留），图注里应写明"其余 13 项炉料组成"。

产出（<结果根目录>/shap/<run>/key_targets_signed/）：
    shap_signed_values_<target>.npz/.csv  逐样本有符号 SHAP（含真实值/预测基线），兼作重绘缓存
    shap_signed_summary.csv               目标 × 特征：mean signed SHAP、mean |SHAP|、方向
    fig_shap_<target>_beeswarm.{png,svg,pdf}      蜂群图（有符号，按特征值着色）
    fig_shap_<target>_signed_bar.{png,svg,pdf}    平均有符号 SHAP（方向）条形图
    fig_shap_<target>_waterfall.{png,svg,pdf}     单个样本 waterfall（--sample 指定，默认 0）
    fig_shap_<target>_dependence_<feature>.{png,svg,pdf}  前 3 特征的依赖图
    fig_shap_key_targets_beeswarm.{png,svg,pdf}   四关键指标 2×2 蜂群图（论文图）

特性：有符号矩阵逐目标缓存，已存在则直接重绘不重算（改图样式时非常省时）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from common import (
    CHARGE_FEATURES, FEATURE_LABELS, INPUT_COLS, KEY_TARGETS, TARGET_LABELS, banner, human,
    load_config, load_dataset, setup_console, setup_env, split_data, style_paper_bars, timer,
)
import step4_shap_all_targets as step4          # 复用建模型 / 归档加载 / 解释器代码

warnings.filterwarnings("ignore")
setup_console()

SOURCES = ("cache", "archive")
POS_COLOR = "#B2182B"       # 正贡献
NEG_COLOR = "#2166AC"       # 负贡献
CYAN = "#0072B2"


# ---------------------------------------------------------------------------
#                              命令行
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", default=None,
                        help="逗号分隔的目标名（默认四项关键指标 matte_Cu_pct,slag_Cu_pct,slag_FeO_pct,slag_SiO2_pct）")
    parser.add_argument("--n-explain", dest="n_explain", type=int, default=None, help="被解释样本数")
    parser.add_argument("--n-background", dest="n_background", type=int, default=None, help="SHAP 背景样本数")
    parser.add_argument("--budget", type=int, default=None, help="SHAP 计算预算（16 个特征精确需 65536）")
    parser.add_argument("--source", choices=SOURCES, default=None, help="cache（默认，快）/ archive（慢）")
    parser.add_argument("--sample", type=int, default=0, help="waterfall 用第几个被解释样本（默认 0）")
    parser.add_argument("--top-features", dest="top_features", type=int, default=10,
                        help="有向条形图显示的 Top-N 特征（默认 10）")
    parser.add_argument("--n-dependence", dest="n_dependence", type=int, default=3,
                        help="画几张依赖图（前 N 个重要特征，默认 3）")
    parser.add_argument("--keep-constant", dest="keep_constant", action="store_true",
                        help="保留常量特征（固定工况数据集里 T/P/O2 的 SHAP 恒为 0，默认剔除）")
    parser.add_argument("--recompute", action="store_true", help="忽略缓存，重新计算 SHAP")
    known, _ = parser.parse_known_args()
    return known


# ---------------------------------------------------------------------------
#                              样式
# ---------------------------------------------------------------------------

def apply_sci_style() -> None:
    """SCI 期刊风格：Arial、矢量文字可编辑、8 pt 正文。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"图: {path_base.name}.pdf/.svg/.png", flush=True)


# ---------------------------------------------------------------------------
#                          SHAP 计算与缓存
# ---------------------------------------------------------------------------

def cache_paths(out_dir: Path, target: str) -> tuple[Path, Path]:
    return out_dir / f"shap_signed_values_{target}.npz", out_dir / f"shap_signed_values_{target}.csv"


def load_or_compute(cfg, target: str, x_train: np.ndarray, y_train: np.ndarray, x_background: np.ndarray,
                    x_explain: np.ndarray, y_explain: np.ndarray, budget: int, source: str,
                    out_dir: Path, recompute: bool):
    """取该目标的有符号 SHAP 矩阵：有缓存直接读，否则拟合并解释后落盘。"""
    npz_path, csv_path = cache_paths(out_dir, target)

    if npz_path.exists() and not recompute:
        blob = np.load(npz_path)
        print(f"    命中缓存 {npz_path.name}，跳过 SHAP 计算", flush=True)
        return blob["values"], blob["base_values"], blob["y_true"], blob["y_pred"]

    col = step4.OUTPUT_COLS.index(target)
    y_target = y_train[:, col].astype(np.float32)

    t0 = timer()
    reg = step4.build_model(cfg, target, x_train, y_target, source)
    t_ready = timer()
    explanation = step4.explain_model(reg, x_background, x_explain, budget)
    y_pred = reg.predict(x_explain)
    t_shap = timer()

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 0]
    base_values = np.asarray(explanation.base_values).reshape(-1)

    np.savez_compressed(npz_path, values=values, base_values=base_values,
                        y_true=y_explain, y_pred=y_pred)
    frame = pd.DataFrame(values, columns=[FEATURE_LABELS.get(c, c) for c in INPUT_COLS])
    frame.insert(0, "y_pred", y_pred)
    frame.insert(0, "y_true", y_explain)
    frame.insert(0, "sample", np.arange(len(values)))
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"    模型就绪 {human(t_ready - t0)} / SHAP {human(t_shap - t_ready)}；"
          f"缓存 {npz_path.name} + {csv_path.name}", flush=True)
    if source == "cache":
        del reg
        free_gpu()
    return values, base_values, y_explain, y_pred


def free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_explanation(values: np.ndarray, base_values: np.ndarray, data: np.ndarray,
                      keep: list[int]):
    """构造 shap.Explanation（带 data，beeswarm/scatter 需要用它着色）。"""
    import shap

    labels = [FEATURE_LABELS.get(INPUT_COLS[j], INPUT_COLS[j]) for j in keep]
    return shap.Explanation(
        values=values[:, keep],
        base_values=base_values,
        data=data[:, keep],
        feature_names=labels,
    )


# ---------------------------------------------------------------------------
#                              绘图
# ---------------------------------------------------------------------------

def plot_beeswarm(exp, target: str, out_dir: Path, max_display: int = 13) -> None:
    """有符号蜂群图：每个点是一个样本，横轴为 SHAP 值，颜色为特征取值。"""
    import shap

    fig = plt.figure(figsize=(3.9, 3.0))
    ax = fig.add_subplot(111)
    shap.plots.beeswarm(exp, max_display=max_display, ax=ax, show=False,
                        color_bar=True, s=13, plot_size=None, color_bar_label="Feature value")
    ax.set_xlabel(f"SHAP value for {TARGET_LABELS.get(target, target)} (wt%)")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    fig.subplots_adjust(left=0.24, right=0.92, top=0.94, bottom=0.18)
    save_figure(fig, out_dir / f"fig_shap_{file_label(target)}_beeswarm")
    plt.close(fig)


def plot_signed_bar(values: np.ndarray, keep: list[int], target: str, out_dir: Path,
                    top_features: int = 10) -> None:
    """平均有符号 SHAP 条形图（正=抬高指标，负=压低指标），按 mean |SHAP| 排序取 Top-N。"""
    mean_signed = values[:, keep].mean(axis=0)
    mean_abs = np.abs(values[:, keep]).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_features]
    labels = [FEATURE_LABELS.get(INPUT_COLS[keep[j]], INPUT_COLS[keep[j]]) for j in order]
    vals = mean_signed[order]

    y = np.arange(len(vals))[::-1]
    fig, ax = plt.subplots(figsize=(3.9, max(2.2, 0.3 * len(vals) + 0.9)))
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.68)
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Mean signed SHAP (wt%) — {TARGET_LABELS.get(target, target)}")
    span = max(abs(vals).max(), 1e-9)
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_paper_bars(ax, axis="x")     # 网格画在柱子上层（手稿图3 风格）
    for yy, val in zip(y, vals):
        offset = span * 0.03
        ax.text(val + (offset if val >= 0 else -offset), yy, f"{val:+.2f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=6.5)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.20)
    save_figure(fig, out_dir / f"fig_shap_{file_label(target)}_signed_bar")
    plt.close(fig)


def plot_waterfall(exp, target: str, out_dir: Path, sample: int = 0, max_display: int = 10) -> None:
    """单个样本的 waterfall：基线 → 预测值，逐步累加各特征的贡献。"""
    import shap

    idx = int(np.clip(sample, 0, len(exp.values) - 1))
    print(f"    waterfall 使用第 {idx} 个被解释样本", flush=True)
    shap.plots.waterfall(exp[idx], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(4.6, max(2.6, 0.32 * min(max_display, len(exp.feature_names)) + 1.4))
    # waterfall 的横轴标签由 shap 写成 "E[f(X)] = ..."（模型输出基线），保留它不要覆盖
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{TARGET_LABELS.get(target, target)}", fontweight="bold", y=1.02)
    save_figure(fig, out_dir / f"fig_shap_{file_label(target)}_waterfall")
    plt.close(fig)


def plot_dependence(exp, values: np.ndarray, keep: list[int], target: str, out_dir: Path,
                    n_dependence: int = 3) -> None:
    """前 N 个重要特征的依赖图（SHAP 值 vs 特征取值，颜色=最强交互特征）。"""
    import shap

    mean_abs = np.abs(values[:, keep]).mean(axis=0)
    order = np.argsort(-mean_abs)[:n_dependence]
    for rank, j in enumerate(order, start=1):
        name = INPUT_COLS[keep[j]]
        fig = plt.figure(figsize=(3.2, 2.6))
        ax = fig.add_subplot(111)
        try:
            shap.plots.scatter(exp[:, j], color=exp, ax=ax, show=False, dot_size=14)
        except Exception as exc:                      # 交互项估计失败时退回单色散点
            print(f"    依赖图 {name} 自动着色失败（{exc}），改用单色散点", flush=True)
            ax.scatter(exp.data[:, j], exp.values[:, j], s=14, color=CYAN, alpha=0.85)
            ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
            ax.set_xlabel(FEATURE_LABELS.get(name, name))
            ax.set_ylabel("SHAP value (wt%)")
        ax.set_title(f"{TARGET_LABELS.get(target, target)} — rank {rank}", fontweight="bold")
        fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
        save_figure(fig, out_dir / f"fig_shap_{file_label(target)}_dependence_{label_slug(name)}")
        plt.close(fig)


def plot_directional_bar(values: np.ndarray, data: np.ndarray, keep: list[int], target: str,
                         out_dir: Path, top_features: int = 10) -> None:
    """方向条形图：长度=mean|SHAP|（重要性），方向/颜色=效应方向。

    方向取"SHAP 值与该特征取值的相关系数"的符号（比 mean signed SHAP 稳健）：
    相关系数为正 → 该成分升高会抬高指标（红，画在右侧）；为负 → 压低指标（蓝，画在左侧）。
    强非线性时 mean signed SHAP 会正负抵消（例如 matte_Cu 的 S：mean|SHAP|=9.3、mean signed 仅 +1.05），
    所以幅值用 mean|SHAP|、方向单独用相关系数表达。
    """
    mean_abs = np.abs(values[:, keep]).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_features]
    labels, lengths, corrs = [], [], []
    for j in order:
        feature_idx = keep[j]
        col = data[:, feature_idx]
        corr = float(np.corrcoef(values[:, feature_idx], col)[0, 1]) if col.std() > 1e-12 else 0.0
        labels.append(FEATURE_LABELS.get(INPUT_COLS[feature_idx], INPUT_COLS[feature_idx]))
        lengths.append(mean_abs[j])
        corrs.append(corr if abs(corr) > 1e-3 else 0.0)

    signed = [ln if c >= 0 else -ln for ln, c in zip(lengths, corrs)]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(3.9, max(2.2, 0.3 * len(labels) + 0.9)))
    ax.barh(y, signed, color=[POS_COLOR if c >= 0 else NEG_COLOR for c in corrs],
            edgecolor="none", height=0.68)
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Mean |SHAP| (wt%), signed by effect direction — {TARGET_LABELS.get(target, target)}")
    span = max(max(lengths), 1e-9)
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_paper_bars(ax, axis="x")     # 网格画在柱子上层（手稿图3 风格）
    for yy, ln, corr, val in zip(y, lengths, corrs, signed):
        offset = span * 0.03
        ax.text(val + (offset if val >= 0 else -offset), yy, f"{ln:.2f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=6.5)
    ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.20)
    save_figure(fig, out_dir / f"fig_shap_{file_label(target)}_directional_bar")
    plt.close(fig)


def plot_directional_heatmap(records: list[dict], targets: list[str], out_dir: Path) -> None:
    """四关键指标 × 13 项炉料成分的方向热力图。

    颜色 = 行归一化后的 mean|SHAP| 再按效应方向取符号（红=成分升高抬高指标，蓝=压低指标）；
    格内数字 = 原始 mean|SHAP|。
    """
    if not records:
        return
    frame = pd.DataFrame(records)
    frame = frame[frame["feature"].isin(CHARGE_FEATURES)]
    if frame.empty:
        return
    present = [t for t in KEY_TARGETS if t in set(frame["target"])]
    if not present:
        return
    matrix = np.full((len(present), len(CHARGE_FEATURES)), np.nan)
    for i, target in enumerate(present):
        sub = frame[frame["target"] == target].set_index("feature")
        if sub.empty:
            continue
        row = sub["mean_abs_shap"].reindex(CHARGE_FEATURES).to_numpy(dtype=float)
        peak = np.nanmax(row) if np.isfinite(row).any() else 0.0
        for j, feature in enumerate(CHARGE_FEATURES):
            if feature not in sub.index:
                continue
            corr = float(sub.loc[feature, "shap_feature_corr"])
            sign = 1.0 if corr >= 0 else -1.0
            matrix[i, j] = sign * (row[j] / peak if peak > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(present) + 1.9))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CHARGE_FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in CHARGE_FEATURES], fontsize=7)
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([TARGET_LABELS.get(t, t) for t in present], fontsize=7)
    for i in range(len(present)):
        for j in range(len(CHARGE_FEATURES)):
            sub = frame[(frame["target"] == present[i]) & (frame["feature"] == CHARGE_FEATURES[j])]
            if sub.empty:
                continue
            raw = float(sub["mean_abs_shap"].iloc[0])
            corr = float(sub["shap_feature_corr"].iloc[0])
            arrow = ("\u2191" if corr >= 0 else "\u2193") if abs(corr) >= 0.2 else "\u2013"
            ax.text(j, i, f"{raw:.1f}\n{arrow}", ha="center", va="center", fontsize=6,
                    color="black" if abs(matrix[i, j]) < 0.6 else "white")
    ax.set_xticks(np.arange(-0.5, len(CHARGE_FEATURES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(present), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("Row-normalised mean |SHAP|,\nsigned by effect direction", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_title("Direction of charge-composition effects on key targets\n"
                 "(red: higher input raises the target; blue: lowers it; number = mean |SHAP|)",
                 fontweight="bold", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.16)
    save_figure(fig, out_dir / "fig_shap_key_targets_directional")
    plt.close(fig)


def plot_key_beeswarm_panel(explanations: dict, targets: list[str], out_dir: Path) -> None:
    """四关键指标 2×2 蜂群图（论文图，每图 Top-8 特征，不重复画色条）。"""
    import shap

    present = [t for t in targets if t in explanations]
    if len(present) < 2:
        return
    ncols = 2
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, target in zip(axes, present):
        shap.plots.beeswarm(explanations[target], max_display=8, ax=ax, show=False,
                            color_bar=False, s=11, plot_size=None)
        ax.set_title(TARGET_LABELS.get(target, target), fontweight="bold")
        ax.set_xlabel("SHAP value (wt%)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(axis="x", labelsize=6.5)
    for ax in axes[len(present):]:
        ax.set_visible(False)
    # 面板里没有逐图的色标，补一个共享色标说明颜色含义（与 shap 蜂群图同一套色带）
    try:
        from shap.plots import colors as shap_colors
        cmap = shap_colors.red_blue
    except Exception:
        cmap = "coolwarm"
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cax = fig.add_axes([0.955, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    cbar.set_label("Feature value", fontsize=7)
    cbar.set_ticks([])
    cbar.ax.text(1.9, 0.0, "Low", transform=cbar.ax.transAxes, va="bottom", ha="left", fontsize=6.5)
    cbar.ax.text(1.9, 1.0, "High", transform=cbar.ax.transAxes, va="top", ha="left", fontsize=6.5)
    fig.suptitle("Signed SHAP distribution for key metallurgical targets", fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.94, top=0.91, bottom=0.07, wspace=0.55, hspace=0.55)
    save_figure(fig, out_dir / "fig_shap_key_targets_beeswarm")
    plt.close(fig)


def file_label(target: str) -> str:
    return target.replace("_pct", "").lower()


def label_slug(name: str) -> str:
    return (name.replace("(g)", "").replace("(", "").replace(")", "")
            .replace("2", "2").replace("\u2103", "T"))


# ---------------------------------------------------------------------------
#                              主流程
# ---------------------------------------------------------------------------

def main() -> None:
    cli = parse_cli()
    cfg = load_config()
    setup_env(cfg)

    targets = cfg.resolve_targets(cli.targets) if cli.targets else list(KEY_TARGETS)
    n_explain = cfg.shap_n_explain if cli.n_explain is None else cli.n_explain
    n_background = cfg.shap_n_background if cli.n_background is None else cli.n_background
    budget = cfg.shap_budget if cli.budget is None else cli.budget
    source = (cli.source or cfg.shap_source).strip().lower()
    if source not in SOURCES:
        raise SystemExit(f"未知的 --source: {source}")

    out_dir = cfg.shap_dir / "key_targets_signed"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_sci_style()
    started = timer()

    banner(f"步骤5b  四项关键指标有符号 SHAP（{len(targets)} 个目标，模型来源={source}）", cfg)
    print(f"目标变量  : {', '.join(targets)}")
    print(f"背景样本  : 训练集前 {n_background} 行")
    print(f"解释样本  : 测试集前 {n_explain} 行")
    print(f"SHAP 预算 : {budget}")
    print(f"输出目录  : {out_dir}\n", flush=True)

    df, x, y = load_dataset(cfg)
    x_train, x_test, y_train, y_test, _, _ = split_data(cfg, x, y)
    x_background = x_train[:n_background].astype(np.float32)
    x_explain = x_test[:min(n_explain, len(x_test))].astype(np.float32)

    constant = [j for j in range(len(INPUT_COLS)) if x_train[:, j].std() <= 1e-12]
    keep = list(range(len(INPUT_COLS))) if cli.keep_constant else \
        [j for j in range(len(INPUT_COLS)) if j not in constant]
    if constant and not cli.keep_constant:
        print(f"常量特征（SHAP 恒为 0，已从图中剔除）: {', '.join(INPUT_COLS[j] for j in constant)}"
              f"\n", flush=True)

    explanations = {}
    summary_rows = []
    for order, target in enumerate(targets, start=1):
        if target not in step4.OUTPUT_COLS:
            print(f"[{order}/{len(targets)}] {target} 不在目标列表中，跳过", flush=True)
            continue
        col = step4.OUTPUT_COLS.index(target)
        if float(np.std(y_train[:, col])) < 1e-10:
            print(f"[{order}/{len(targets)}] {target} 训练目标为常数，跳过", flush=True)
            continue

        print(f"[{order}/{len(targets)}] {target} ...", flush=True)
        values, base_values, y_true, y_pred = load_or_compute(
            cfg, target, x_train, y_train, x_background, x_explain,
            y_test[:len(x_explain), col], budget, source, out_dir, cli.recompute,
        )
        exp = build_explanation(values, base_values, x_explain, keep)
        explanations[target] = exp

        mean_signed = values[:, keep].mean(axis=0)
        mean_abs = np.abs(values[:, keep]).mean(axis=0)
        for j, feature_idx in enumerate(keep):
            name = INPUT_COLS[feature_idx]
            corr = float(np.corrcoef(values[:, feature_idx], x_explain[:, feature_idx])[0, 1]) \
                if x_explain[:, feature_idx].std() > 1e-12 else 0.0
            summary_rows.append({
                "target": target, "feature": name,
                "mean_signed_shap": mean_signed[j], "mean_abs_shap": mean_abs[j],
                "shap_feature_corr": corr,
                "direction": "升高该成分→指标上升" if corr > 0.2 else
                             ("升高该成分→指标下降" if corr < -0.2 else "非单调/无一致方向"),
            })

        print(f"    sum|SHAP|={mean_abs.sum():.3f}  std(y)={y_true.std():.3f}  "
              f"比值={mean_abs.sum() / max(y_true.std(), 1e-12):.2f}（>3 说明解释器可能在离流形外推）",
              flush=True)
        print(f"    平均有符号 SHAP Top5: "
              + ", ".join(f"{INPUT_COLS[keep[j]]}={mean_signed[j]:+.2f}"
                          for j in np.argsort(-mean_abs)[:5]), flush=True)

        plot_beeswarm(exp, target, out_dir)
        plot_signed_bar(values, keep, target, out_dir, top_features=cli.top_features)
        plot_directional_bar(values, x_explain, keep, target, out_dir, top_features=cli.top_features)
        plot_waterfall(exp, target, out_dir, sample=cli.sample)
        plot_dependence(exp, values, keep, target, out_dir, n_dependence=cli.n_dependence)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary["rank_abs"] = summary.groupby("target")["mean_abs_shap"] \
            .rank(ascending=False, method="first").astype(int)
        summary = summary.sort_values(["target", "rank_abs"])
        summary.to_csv(out_dir / "shap_signed_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n有符号汇总表: {out_dir / 'shap_signed_summary.csv'}", flush=True)

    plot_key_beeswarm_panel(explanations, [t for t in KEY_TARGETS if t in explanations], out_dir)
    plot_directional_heatmap(summary_rows, targets, out_dir)
    print(f"\n步骤5b 完成，用时 {human(timer() - started)}。\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
