"""步骤4：全部目标的 SHAP 特征重要性分析（逐目标解释图 + 重要性矩阵热力图）。

用法：
    python ml_pipeline/step4_shap_all_targets.py                              # 用 config.py 的设置
    python ml_pipeline/step4_shap_all_targets.py --dataset 3000_fixed_temp
    python ml_pipeline/step4_shap_all_targets.py --targets key                # 只跑四项关键指标
    python ml_pipeline/step4_shap_all_targets.py --targets matte_Cu_pct,slag_Cu_pct
    python ml_pipeline/step4_shap_all_targets.py --n-explain 50 --n-background 200 --budget 256
    python ml_pipeline/step4_shap_all_targets.py --source archive             # 兜底：解释已保存的模型（很慢）

模型来源（--source，默认取 config 的 shap.source）：
    cache   在进程内用与 step1 完全相同的划分与参数重拟合模型，并开启 KV 缓存
            （TabPFNRegressor(fit_mode="fit_with_cache")）后解释。
            已拟合模型无法保存 KV 缓存（tabpfn/inference.py::_raise_if_kv_cache_enabled_on_save_or_load），
            而加载来的模型无法带 KV 缓存，每个特征联盟都要重新编码一遍上下文
            （约 30 s），单次完整解释要几十小时，因此默认走 cache。
    archive 加载 <结果根目录>/tabpfn/<run>/saved_models/<target>.tabpfn_fit（慢，仅作兜底）；
            归档解压到 <输出目录>/_archive_extract 下的临时子目录，加载完立即删除。

解释数据：背景样本取训练集前 n_background 行，被解释样本取测试集前 n_explain 行
（与步骤5 的论文图完全一致，便于两处数字对齐）。

产出（<结果根目录>/shap/<run>/all_targets/）：
    shap_importance.csv        逐目标 mean |SHAP| 矩阵（首列 target，其余 16 列为输入特征）
    shap_heatmap.png           归一化 mean |SHAP| 热力图（目标 × 特征）
    <target>_summary.png       逐目标 SHAP summary 蜂群图
    <target>_bar.png           逐目标 SHAP 条形图
    <target>_waterfall.png     逐目标首个被解释样本的 waterfall 图

特性：逐目标落盘、断点续跑；shap_importance.csv 已含全部待分析目标时不再计算，直接用其重绘热力图；
      打印每个目标的模型就绪时间、SHAP 耗时与 Top5 特征。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import (
    INPUT_COLS, OUTPUT_COLS, ResumeTable, apply_cjk_fonts, banner, human,
    load_config, load_dataset, setup_console, setup_env, split_data, timer,
)

warnings.filterwarnings("ignore")
setup_console()

SOURCES = ("cache", "archive")


def parse_cli() -> argparse.Namespace:
    """解析本脚本自己的参数（--config/--dataset/--run/--root 由 common.load_config 处理）。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", default=None,
                        help="all = 全部 29 个目标，key = 四项关键指标，或逗号分隔的目标名")
    parser.add_argument("--n-explain", dest="n_explain", type=int, default=None,
                        help="被解释的样本数（测试集前若干行）")
    parser.add_argument("--n-background", dest="n_background", type=int, default=None,
                        help="SHAP 背景样本数（训练集前若干行）")
    parser.add_argument("--budget", type=int, default=None, help="SHAP 计算预算（16 个特征精确需 65536）")
    parser.add_argument("--source", choices=SOURCES, default=None, help="cache（默认，快）/ archive（慢）")
    known, _ = parser.parse_known_args()
    return known


def main() -> None:
    cli = parse_cli()
    cfg = load_config()
    setup_env(cfg)
    cfg.ensure_dirs()
    apply_cjk_fonts(cfg)

    targets = cfg.resolve_targets(cli.targets if cli.targets is not None else cfg.shap_targets)
    n_explain = cfg.shap_n_explain if cli.n_explain is None else cli.n_explain
    n_background = cfg.shap_n_background if cli.n_background is None else cli.n_background
    budget = cfg.shap_budget if cli.budget is None else cli.budget
    source = (cli.source or cfg.shap_source).strip().lower()
    if source not in SOURCES:
        raise SystemExit(f"未知的 --source: {source}（可选 {' / '.join(SOURCES)}）")

    out_dir = cfg.shap_all_dir
    started = timer()
    banner(f"步骤4  SHAP 特征重要性（{len(targets)}/{len(OUTPUT_COLS)} 个目标，模型来源={source}）", cfg)
    print(f"背景样本  : 训练集前 {n_background} 行")
    print(f"解释样本  : 测试集前 {n_explain} 行")
    print(f"SHAP 预算 : {budget}")
    print(f"输出目录  : {out_dir}\n", flush=True)

    table = ResumeTable(out_dir / "shap_importance.csv", ["target", *INPUT_COLS], keys=["target"])
    missing = [t for t in targets if not table.has(target=t)]

    if not missing:
        print(f"shap_importance.csv 已包含全部 {len(targets)} 个目标，跳过 SHAP 计算，"
              f"直接用已有结果重绘热力图。", flush=True)
    else:
        if len(missing) < len(targets):
            print(f"断点续跑：{len(targets) - len(missing)} 个目标已有结果，"
                  f"本次只需计算 {len(missing)} 个。", flush=True)
        run_targets(cfg, table, targets, n_explain, n_background, budget, source, started)

    matrix = matrix_from(table, targets)
    plot_heatmap(matrix, out_dir)
    print(f"\n步骤4 完成（{len(matrix)}/{len(targets)} 个目标），用时 {human(timer() - started)}。"
          f"\n输出目录: {out_dir}")


def run_targets(cfg, table: ResumeTable, targets: list[str], n_explain: int, n_background: int,
                budget: int, source: str, started: float) -> None:
    """逐目标计算 SHAP 并落盘。"""
    import shap

    df, x, y = load_dataset(cfg)
    x_train, x_test, y_train, _, _, _ = split_data(cfg, x, y)
    x_background = x_train[:n_background]
    x_explain = x_test[:min(n_explain, len(x_test))]
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
        reg = build_model(cfg, target, x_train, y_train_target, source)
        t_ready = timer()
        explanation = explain_model(reg, x_background, x_explain, budget)
        t_shap = timer()
        mean_abs = np.abs(np.asarray(explanation.values)).mean(axis=0)

        save_explanation_figures(shap, explanation, target, cfg.shap_all_dir)
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
        return load_archived_model(cfg, target, cfg.shap_all_dir / "_archive_extract")
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
    import json
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


def explain_model(reg, x_background: np.ndarray, x_explain: np.ndarray, budget: int):
    """用 shapiq 的一阶 Shapley 值构造 shap.Explanation（沿用原脚本的做法）。"""
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq, shapiq_to_shap_explanation

    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=reg, data=x_background, index="SV", max_order=1,
    )
    return shapiq_to_shap_explanation(
        explainer, x_explain, budget=budget, feature_names=list(INPUT_COLS),
    )


def save_explanation_figures(shap, explanation, target: str, out_dir: Path) -> None:
    """保存单个目标的 summary / bar / waterfall 三张图（样式与参数沿用原脚本）。"""
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{target}_summary.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{target}_bar.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{target}_waterfall.png", dpi=150)
    plt.close()


def print_importance(mean_abs: np.ndarray, top: int = 5) -> None:
    """打印该目标的 Top-N 特征重要性（条形长度沿用原脚本的 30 格）。"""
    order = np.argsort(-mean_abs)
    peak = float(mean_abs[order[0]]) if len(order) else 0.0
    print(f"    特征重要性 Top{top}:", flush=True)
    for j in order[:top]:
        bar = "█" * int(mean_abs[j] / peak * 30) if peak > 0 else ""
        print(f"      {INPUT_COLS[j]:10s} {mean_abs[j]:.4f}  {bar}", flush=True)


def matrix_from(table: ResumeTable, targets: list[str]) -> pd.DataFrame:
    """从续跑表里取出 目标 × 特征 的 mean |SHAP| 矩阵，行为 target 顺序。"""
    frame = table.frame.set_index("target")
    present = [t for t in targets if t in frame.index]
    return frame.loc[present, INPUT_COLS].astype(float)


def plot_heatmap(matrix: pd.DataFrame, out_dir: Path) -> None:
    """生成 SHAP 特征重要性热力图（特征 × 目标），样式与原脚本一致。"""
    import matplotlib.pyplot as plt

    if matrix.empty:
        print("没有可用的 SHAP 结果，跳过热力图。", flush=True)
        return

    shap_matrix = matrix.to_numpy(dtype=float)
    feature_names = list(matrix.columns)
    target_names = list(matrix.index)
    n_targets, n_features = shap_matrix.shape

    # 归一化：每行（每个目标）除以该行最大值，便于跨目标比较
    row_max = shap_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1                      # 避免除零
    norm_matrix = shap_matrix / row_max

    fig_w = max(6, n_features * 0.45 + 3)
    fig_h = max(4, n_targets * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(norm_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(target_names, fontsize=8)

    for i in range(n_targets):
        for j in range(n_features):
            text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{shap_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("归一化 mean |SHAP|", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("SHAP 特征重要性热力图", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"热力图: {out_dir / 'shap_heatmap.png'}（{n_targets} 个目标 × {n_features} 个特征）", flush=True)


def free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
