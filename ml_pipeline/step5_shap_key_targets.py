"""步骤5：四项关键指标的论文级 SHAP 图（单目标柱状图 + 2×2 综合柱状图 + 13 项炉料特征热力图）。

用法：
    python ml_pipeline/step5_shap_key_targets.py                              # 默认四项关键指标
    python ml_pipeline/step5_shap_key_targets.py --dataset 3000_fixed_temp
    python ml_pipeline/step5_shap_key_targets.py --targets matte_Cu_pct,slag_FeO_pct
    python ml_pipeline/step5_shap_key_targets.py --n-explain 50 --n-background 200 --budget 256
    python ml_pipeline/step5_shap_key_targets.py --source archive             # 兜底：解释已保存的模型（很慢）

模型来源（--source，默认取 config 的 shap.source）：
    cache   在进程内用与 step1 完全相同的划分与参数重拟合模型，并开启 KV 缓存
            （TabPFNRegressor(fit_mode="fit_with_cache")）后解释。
            已拟合模型无法保存 KV 缓存（tabpfn/inference.py::_raise_if_kv_cache_enabled_on_save_or_load），
            而加载来的模型无法带 KV 缓存，每个特征联盟都要重新编码训练上下文
            （约 30 s），单次完整解释要几十小时，因此默认走 cache。
    archive 加载 <结果根目录>/tabpfn/<run>/saved_models/<target>.tabpfn_fit（慢，仅作兜底）；
            归档解压到 <输出目录>/_archive_extract 下的临时子目录，加载完立即删除。

解释数据：背景样本取训练集前 n_background 行，被解释样本取测试集前 n_explain 行（与步骤4 一致）。

产出（<结果根目录>/shap/<run>/key_targets/）：
    shap_key_targets_importance.csv              目标 × 16 特征 的 mean |SHAP|（首列 target，兼作续跑缓存）
    shap_key_targets_importance_normalized.csv   逐目标按行最大值归一化
    shap_key_targets_ranked.csv                  长表：target / rank / feature / mean_abs_shap / normalized_importance
    fig_shap_<目标>_bar.{pdf,svg,png}            单目标 Top10 柱状图
    fig_shap_key_targets_bars.{pdf,svg,png}      2×2 综合柱状图（每图 Top8）
    fig_shap_key_targets_heatmap_13_features.{pdf,svg,png}   13 项炉料特征热力图（格内标注数值）

特性：逐目标落盘、断点续跑；重要性 CSV 已含全部目标时不再计算，直接用其重绘全部图表；
      打印每个目标的模型就绪时间、SHAP 耗时与 Top5 特征。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")                                   # 无 GUI 后端
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from common import (
    CHARGE_FEATURES, FEATURE_LABELS, INPUT_COLS, KEY_TARGETS, OUTPUT_COLS, ResumeTable,
    TARGET_LABELS, banner, human, load_config, load_dataset, setup_console, setup_env,
    split_data, style_paper_bars, timer,
)

warnings.filterwarnings("ignore")
setup_console()

SOURCES = ("cache", "archive")

#: 输出目录，由 main() 从 cfg.shap_key_dir 赋值（绘图函数沿用原脚本的模块级常量写法）
OUT_DIR: Path | None = None


def parse_cli() -> argparse.Namespace:
    """解析本脚本自己的参数（--config/--dataset/--run/--root 由 common.load_config 处理）。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", default=None,
                        help="逗号分隔的目标名（默认四项关键指标 matte_Cu_pct,slag_Cu_pct,slag_FeO_pct,slag_SiO2_pct）")
    parser.add_argument("--n-explain", dest="n_explain", type=int, default=None,
                        help="被解释的样本数（测试集前若干行）")
    parser.add_argument("--n-background", dest="n_background", type=int, default=None,
                        help="SHAP 背景样本数（训练集前若干行）")
    parser.add_argument("--budget", type=int, default=None, help="SHAP 计算预算（16 个特征精确需 65536）")
    parser.add_argument("--source", choices=SOURCES, default=None, help="cache（默认，快）/ archive（慢）")
    known, _ = parser.parse_known_args()
    return known


def main() -> None:
    global OUT_DIR

    cli = parse_cli()
    cfg = load_config()
    setup_env(cfg)
    cfg.ensure_dirs()

    targets = cfg.resolve_targets(cli.targets) if cli.targets else list(KEY_TARGETS)
    n_explain = cfg.shap_n_explain if cli.n_explain is None else cli.n_explain
    n_background = cfg.shap_n_background if cli.n_background is None else cli.n_background
    budget = cfg.shap_budget if cli.budget is None else cli.budget
    source = (cli.source or cfg.shap_source).strip().lower()
    if source not in SOURCES:
        raise SystemExit(f"未知的 --source: {source}（可选 {' / '.join(SOURCES)}）")

    OUT_DIR = cfg.shap_key_dir
    started = timer()
    apply_sci_style()

    banner(f"步骤5  四项关键指标 SHAP 论文图（{len(targets)} 个目标，模型来源={source}）", cfg)
    print(f"目标变量  : {', '.join(targets)}")
    print(f"背景样本  : 训练集前 {n_background} 行")
    print(f"解释样本  : 测试集前 {n_explain} 行")
    print(f"SHAP 预算 : {budget}")
    print(f"输出目录  : {OUT_DIR}\n", flush=True)

    table = ResumeTable(OUT_DIR / "shap_key_targets_importance.csv",
                        ["target", *INPUT_COLS], keys=["target"])
    missing = [t for t in targets if not table.has(target=t)]

    if not missing:
        print(f"shap_key_targets_importance.csv 已包含全部 {len(targets)} 个目标，"
              f"跳过 SHAP 计算，直接用已有结果重绘全部图表。", flush=True)
    else:
        if len(missing) < len(targets):
            print(f"断点续跑：{len(targets) - len(missing)} 个目标已有结果，"
                  f"本次只需计算 {len(missing)} 个。", flush=True)
        compute_targets(cfg, table, targets, n_explain, n_background, budget, source, started)

    importance = importance_from(table, targets)
    write_tables(importance, targets)
    plot_heatmap(importance, targets)
    for target in targets:
        plot_target_bar(importance, target)
    plot_key_target_bars(importance, targets)
    print(f"\n步骤5 完成（{len(importance)}/{len(targets)} 个目标），用时 {human(timer() - started)}。"
          f"\n输出目录: {OUT_DIR}")


def compute_targets(cfg, table: ResumeTable, targets: list[str], n_explain: int, n_background: int,
                    budget: int, source: str, started: float) -> None:
    """逐个目标计算 mean |SHAP| 并即时落盘。"""
    df, x, y = load_dataset(cfg)
    x_train, x_test, y_train, _, _, _ = split_data(cfg, x, y)
    x_background = x_train[:n_background].astype(np.float32)
    x_explain = x_test[:min(n_explain, len(x_test))].astype(np.float32)
    print(f"训练集 {len(x_train)}，测试集 {len(x_test)}；"
          f"实际背景 {len(x_background)} 行 / 解释 {len(x_explain)} 行\n", flush=True)

    durations: list[float] = []
    for order, target in enumerate(targets, start=1):
        tag = f"[{order:2d}/{len(targets)}] {target:<24s}"
        if table.has(target=target):
            print(f"{tag} 已完成，跳过", flush=True)
            continue

        col = OUTPUT_COLS.index(target)
        y_train_target = y_train[:, col]
        if float(np.std(y_train_target)) < 1e-10:
            print(f"{tag} 跳过：训练目标为常数，无法做 SHAP 解释", flush=True)
            continue

        print(f"{tag} 开始 ...", flush=True)
        t0 = timer()
        reg = build_model(cfg, target, x_train, y_train_target.astype(np.float32), source)
        t_ready = timer()
        mean_abs = compute_importance(reg, x_background, x_explain, budget)
        t_shap = timer()

        table.add(target=target, **{name: float(value) for name, value in zip(INPUT_COLS, mean_abs)})

        elapsed = timer() - t0
        durations.append(elapsed)
        avg = sum(durations) / len(durations)
        print(f"    模型就绪 {human(t_ready - t0)} / SHAP {human(t_shap - t_ready)} / "
              f"本目标合计 {human(elapsed)}", flush=True)
        print_importance(mean_abs)
        print(f"    已用 {human(timer() - started)}  "
              f"预计剩余 {human(avg * len([t for t in targets if not table.has(target=t)]))}",
              flush=True)
        if source == "cache":
            del reg
            free_gpu()

    if not durations:
        print("没有目标被计算（全部已有结果或全部跳过）。", flush=True)


def build_model(cfg, target: str, x_train: np.ndarray, y_train: np.ndarray, source: str):
    """按 --source 准备该目标的已拟合模型。"""
    if source == "archive":
        return load_archived_model(cfg, target, OUT_DIR / "_archive_extract")
    from tabpfn import TabPFNRegressor

    # 与 step1 完全相同的参数，只多开 KV 缓存：联盟之间共享同一份训练上下文编码
    reg = TabPFNRegressor(
        device=cfg.device, model_path=str(cfg.model_path),
        n_estimators=cfg.n_estimators, random_state=cfg.model_seed,
        ignore_pretraining_limits=True, fit_mode=cfg.fit_mode_for("shap"),
    )
    reg.fit(x_train, y_train)
    return reg


def load_archived_model(cfg, target: str, scratch_root: Path):
    """加载 <结果根目录>/tabpfn/<run>/saved_models/<target>.tabpfn_fit。

    沿用 scripts/analysis/plot_shap_key_targets_sci.py::load_tabpfn_model 的做法：解压归档、
    重建估计器、恢复已拟合属性；归档里的 init_params 存的是训练时的权重相对路径，
    这里统一换成配置里的权重路径。
    解压目录取结果目录下的 _archive_extract（不用系统临时目录：受限沙箱里临时目录不可写），
    目标加载完立即删除。
    注意：归档模型没有 KV 缓存，每个特征联盟都要重新编码上下文，非常慢。
    """
    import shutil

    import joblib
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import InferenceEngine, _extract_archive

    model_file = cfg.models_dir / f"{target}.tabpfn_fit"
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")

    work = scratch_root / target
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        _extract_archive(model_file, work)

        with (work / "init_params.json").open(encoding="utf-8") as handle:
            params = json.load(handle)
        params.pop("__class_name__", None)
        params["model_path"] = str(cfg.model_path)
        params["device"] = cfg.device
        precision = params.get("inference_precision")
        if isinstance(precision, str) and precision.startswith("torch."):
            params["inference_precision"] = getattr(torch, precision.split(".")[1])

        est = TabPFNRegressor(**params)
        est._initialize_model_variables()
        for key, value in joblib.load(work / "fitted_attrs.joblib").items():
            setattr(est, key, value)
        est.executor_ = InferenceEngine.load_state(work / "executor_state.joblib", est.models_)
        est.to(cfg.device)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:                                       # 顺手删掉空壳目录
            scratch_root.rmdir()
        except OSError:
            pass

    return est


def compute_importance(model, x_background: np.ndarray, x_explain: np.ndarray, budget: int) -> np.ndarray:
    """一阶 Shapley 值的 mean |SHAP|（沿用原脚本的 shapiq 调用方式）。"""
    from tabpfn_extensions.interpretability import (
        shapiq as tabpfn_shapiq,
        shapiq_to_shap_explanation,
    )

    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=model, data=x_background, index="SV", max_order=1,
    )
    explanation = shapiq_to_shap_explanation(
        explainer, x_explain, budget=budget, feature_names=list(INPUT_COLS),
    )
    values = np.asarray(explanation.values)
    if values.ndim == 3:                          # (n, d, 1) -> (n, d)
        values = values[:, :, 0]
    return np.abs(values).mean(axis=0)


def print_importance(mean_abs: np.ndarray, top: int = 5) -> None:
    """打印该目标的 Top-N 特征重要性。"""
    order = np.argsort(-mean_abs)
    print(f"    特征重要性 Top{top}:", flush=True)
    for j in order[:top]:
        print(f"      {INPUT_COLS[j]:10s} {mean_abs[j]:.4f}", flush=True)


def importance_from(table: ResumeTable, targets: list[str]) -> pd.DataFrame:
    """从续跑表里取出 目标 × 16 特征 的重要性表，行为 targets 顺序。"""
    frame = table.frame.set_index("target")
    present = [t for t in targets if t in frame.index]
    return frame.loc[present, INPUT_COLS].astype(float)


def write_tables(importance: pd.DataFrame, targets: list[str]) -> None:
    """导出重要性表、归一化表与长表排名（重要性表本身即续跑缓存，不再覆写）。"""
    normalized = importance.div(importance.max(axis=1).replace(0, 1), axis=0)
    normalized.to_csv(OUT_DIR / "shap_key_targets_importance_normalized.csv", encoding="utf-8-sig")

    rank_rows = []
    for target in targets:
        if target not in importance.index:
            continue
        for rank, (feature, value) in enumerate(
            importance.loc[target].sort_values(ascending=False).items(), start=1
        ):
            rank_rows.append({
                "target": target,
                "rank": rank,
                "feature": feature,
                "mean_abs_shap": value,
                "normalized_importance": normalized.loc[target, feature],
            })
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "shap_key_targets_ranked.csv",
                                   index=False, encoding="utf-8-sig")
    print(f"重要性表: {OUT_DIR / 'shap_key_targets_importance.csv'}")
    print(f"归一化表: {OUT_DIR / 'shap_key_targets_importance_normalized.csv'}")
    print(f"排名长表: {OUT_DIR / 'shap_key_targets_ranked.csv'}", flush=True)


# ---------------------------------------------------------------------------
#                              论文级绘图
# ---------------------------------------------------------------------------

def apply_sci_style() -> None:
    """SCI 期刊风格：Arial 字体、可编辑矢量文字（pdf.fonttype=42、svg.fonttype='none'）。"""
    mpl.rcParams.update(
        {
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
        }
    )


def save_figure(fig, path_base: Path) -> None:
    """同时导出 pdf / svg / png（png 600 dpi）。"""
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"图: {path_base.name}.pdf/.svg/.png", flush=True)


def plot_heatmap(importance: pd.DataFrame, targets: list[str]) -> None:
    """13 项炉料特征 × 关键目标 的归一化热力图（格内标注原始 mean |SHAP|）。"""
    target_order = [t for t in targets if t in importance.index]
    if not target_order:
        print("没有可用的 SHAP 结果，跳过绘图。", flush=True)
        return

    row_max = importance.loc[target_order].max(axis=1).replace(0, 1)
    norm = importance.loc[target_order].div(row_max, axis=0)

    feature_order = (
        importance.loc[target_order, CHARGE_FEATURES]
        .mean(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    norm = norm[feature_order]
    raw = importance.loc[target_order, feature_order]

    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    color_norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=1)
    im = ax.imshow(norm.values, cmap="viridis", norm=color_norm, aspect="auto")

    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in feature_order], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_order)))
    ax.set_yticklabels([TARGET_LABELS.get(t, t) for t in target_order])
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Prediction target")
    ax.set_title("SHAP feature importance for key metallurgical targets", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(feature_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)

    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            rgba = im.cmap(im.norm(norm.iat[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "#222222" if luminance > 0.58 else "white"
            ax.text(j, i, f"{raw.iat[i, j]:.2g}", ha="center", va="center",
                    fontsize=5.1, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized mean |SHAP|")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)

    fig.subplots_adjust(left=0.12, right=0.94, top=0.82, bottom=0.32)
    save_figure(fig, OUT_DIR / "fig_shap_key_targets_heatmap_13_features")
    plt.close(fig)


def _plot_bar_axis(ax, importance: pd.DataFrame, target: str, n_features: int = 8) -> None:
    """在给定坐标轴上画某个目标的 Top-N 水平柱状图。"""
    vals = importance.loc[target].sort_values(ascending=False).head(n_features)
    labels = [FEATURE_LABELS[c] for c in vals.index]
    y = np.arange(len(vals))[::-1]
    colors = ["#0072B2" if i == 0 else "#7AA6C2" for i in range(len(vals))]
    ax.barh(y, vals.values, color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(TARGET_LABELS.get(target, target), fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 网格线画在柱子上层（手稿图3 风格：线横穿柱体），不能压在柱子背后
    style_paper_bars(ax, axis="x")
    for yy, val in zip(y, vals.values):
        ax.text(val + vals.max() * 0.015, yy, f"{val:.2g}", va="center", fontsize=6.5)


def plot_target_bar(importance: pd.DataFrame, target: str) -> None:
    """单目标 Top10 柱状图（3.54 × 2.75 英寸，双栏半栏宽）。"""
    if target not in importance.index:
        return
    fig, ax = plt.subplots(figsize=(3.54, 2.75))
    _plot_bar_axis(ax, importance, target, n_features=10)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.18)
    file_label = target.replace("_pct", "").replace("_", "_").lower()
    save_figure(fig, OUT_DIR / f"fig_shap_{file_label}_bar")
    plt.close(fig)


def plot_key_target_bars(importance: pd.DataFrame, targets: list[str]) -> None:
    """关键目标综合柱状图（默认四项时即 2×2，每图 Top8）。"""
    targets = [t for t in targets if t in importance.index]
    if not targets:
        return
    ncols = 2
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 5.1 if nrows == 2 else 2.55 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, target in zip(axes, targets):
        _plot_bar_axis(ax, importance, target, n_features=8)
    for ax in axes[len(targets):]:
        ax.set_visible(False)
    fig.suptitle("SHAP importance for key metallurgical targets", fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.09, wspace=0.33, hspace=0.48)
    save_figure(fig, OUT_DIR / "fig_shap_key_targets_bars")
    plt.close(fig)


def free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
