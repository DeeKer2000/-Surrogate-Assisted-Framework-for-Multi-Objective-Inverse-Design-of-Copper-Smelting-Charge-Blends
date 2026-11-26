"""步骤6：学习曲线（固定测试集，递增的嵌套训练子集，四项关键指标）。

用法：
    python ml_pipeline/step6_learning_curve.py                                  # 用 config.py 的规模序列
    python ml_pipeline/step6_learning_curve.py --dataset 3000_fixed_temp
    python ml_pipeline/step6_learning_curve.py --sizes 500,1000,1500
    python ml_pipeline/step6_learning_curve.py --targets matte_Cu_pct,slag_Cu_pct
    python ml_pipeline/step6_learning_curve.py --panel-label a                    # 拼版时加子图编号

说明：训练集已由 train_test_split 打乱，各规模一律取同一顺序的前 n 条，
      因此子集相互嵌套、测试集始终不变。按 (size, target) 断点续跑。

产出（<结果根目录>/learning_curve/<run>/）：
    learning_curve_detail.csv   每个 (size, target) 的 R2
    learning_curve_summary.csv  各训练规模的 mean / std
    fig_learning_curve.png      学习曲线图（Mean R² vs 训练集大小）
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from common import (
    KEY_TARGETS, OUTPUT_COLS, ResumeTable, apply_cjk_fonts, banner, human,
    load_config, load_dataset, normalize_fit_mode, parse_list, predict_batched,
    setup_console, setup_env, split_data, timer,
)

warnings.filterwarnings("ignore")
setup_console()

FIELDS = ["size", "target", "n_train", "n_test", "R2", "elapsed_seconds"]
#: cfg.learning_curve_sizes 为空时使用的默认规模序列
DEFAULT_SIZES = [100, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]
#: 左上角面板字母：只有把本图与其他子图拼成 a/b/c 版式时才需要，默认不加。
#: 需要时用 --panel-label a（例如 python ml_pipeline/step6_learning_curve.py --panel-label a）。
DEFAULT_PANEL_LABEL = ""
LINE_COLOR = "#484878"
REF_COLOR = "#A0A0A0"


def main() -> None:
    args = _cli_args()
    cfg = load_config()
    setup_env(cfg)
    cfg.ensure_dirs()
    apply_cjk_fonts(cfg)

    import torch
    from tabpfn import TabPFNRegressor

    targets = cfg.resolve_targets(args.targets) if args.targets else list(KEY_TARGETS)
    sizes = _resolve_sizes(args, cfg)
    panel_label = _resolve_panel_label(args)
    started = timer()

    # fit_mode：默认与步骤1一致（关闭 KV 缓存），学习曲线的绝对 R² 才能与步骤1对齐。
    # 加 --fit-mode cache 可加速，但数值会有 1e-4~1e-6 量级差异，见 config.py 第 5 节。
    fit_mode = normalize_fit_mode(getattr(args, "fit_mode", None), cfg.fit_mode_for("learning_curve"))
    if fit_mode == "fit_with_cache":
        print("提示: 已启用 KV 缓存加速，R² 与步骤1（默认路径）在 1e-4~1e-6 量级上略有差异。",
              flush=True)

    banner(f"步骤6  学习曲线（{len(sizes)} 个训练规模 × {len(targets)} 个目标）", cfg)
    print(f"训练规模    : {sizes}")
    print(f"目标变量    : {', '.join(targets)}")
    print(f"模型        : n_estimators={cfg.n_estimators}，model_seed={cfg.model_seed}，"
          f"fit_mode={fit_mode}")
    print(f"预测批量    : {cfg.predict_batch_size}（显存不足自动折半至 {cfg.predict_batch_floor}）")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("警告: 未检测到 CUDA，回退 CPU 会非常慢。", flush=True)

    _, x, y = load_dataset(cfg)
    x_tr, x_te, y_tr, y_te, _, _ = split_data(cfg, x, y)
    print(f"划分        : 训练集 {len(x_tr)}，测试集 {len(x_te)}（固定，seed={cfg.random_state}）")

    # 训练子集规模裁剪：不超过训练集行数，去重排序
    valid = sorted({s for s in sizes if 0 < s <= len(x_tr)})
    dropped = sorted({s for s in sizes if s > len(x_tr)})
    if dropped:
        print(f"注意: 规模 {dropped} 超过训练集 {len(x_tr)} 行，已忽略。")
    if not valid:
        raise ValueError(f"没有有效的训练规模（训练集 {len(x_tr)} 行）：{sizes}")
    if len(valid) < 2:
        print("注意: 只有 1 个有效训练规模，曲线只会有一个点。")
    sizes = valid

    total = len(sizes) * len(targets)
    table = ResumeTable(cfg.learning_curve_dir / "learning_curve_detail.csv", FIELDS,
                        keys=["size", "target"])
    durations = [float(v) for v in table.frame.get("elapsed_seconds", pd.Series(dtype=float)) if v == v]
    skipped: set[tuple[int, str]] = set()
    done = len([1 for s in sizes for t in targets if table.has(size=s, target=t)])
    if done:
        print(f"断点续跑    : 已有 {done}/{total} 条结果，跳过重复计算。")
    print(f"训练子集    : 取同一打乱顺序的前 n 条（相互嵌套）\n", flush=True)

    for size in sizes:
        for target in targets:
            col = OUTPUT_COLS.index(target)
            y_train = y_tr[:size, col]
            if np.std(y_train) < 1e-10:
                skipped.add((size, target))
                print(f"[跳过] size={size:<6d} {target:<22s} 训练子集为常数列", flush=True)
                continue
            if table.has(size=size, target=target):
                print(f"[跳过] size={size:<6d} {target:<22s} 已完成", flush=True)
                continue

            order = done + 1
            print(f"[{order:3d}/{total}] size={size:<6d} {target:<22s} 开始 ...", flush=True)
            t0 = timer()
            reg = TabPFNRegressor(
                device=cfg.device, model_path=str(cfg.model_path),
                n_estimators=cfg.n_estimators, random_state=cfg.model_seed,
                ignore_pretraining_limits=True, fit_mode=fit_mode,
            )
            reg.fit(x_tr[:size], y_train)
            t_fit = timer()
            y_pred = np.asarray(predict_batched(
                reg, x_te, cfg.predict_batch_size, cfg.predict_batch_floor), dtype=float)

            if not np.isfinite(y_pred).all():
                raise ValueError(f"预测值非有限: size={size}, target={target}")

            elapsed = timer() - t0
            durations.append(elapsed)
            r2 = float(r2_score(y_te[:, col], y_pred))
            table.add(size=size, target=target, n_train=size, n_test=len(x_te),
                      R2=r2, elapsed_seconds=elapsed)

            del reg
            gc.collect()
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            done += 1
            avg = sum(durations) / len(durations)
            print(f"[{done:3d}/{total}] size={size:<6d} {target:<22s} R2={r2:.6f}  "
                  f"拟合 {human(t_fit - t0)} / 预测 {human(timer() - t_fit)}  已用 {human(timer() - started)}  "
                  f"预计剩余 {human(avg * (total - done - len(skipped)))}", flush=True)

    publish(cfg, table, sizes, targets, skipped, panel_label)
    print(f"\n步骤6 完成（{done}/{total} 条），用时 {human(timer() - started)}。"
          f"\n结果目录: {cfg.learning_curve_dir}")


def _cli_args() -> argparse.Namespace:
    """解析本脚本专有参数；--config/--dataset/--run/--root 由 common.load_config 处理。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sizes", default=None, help="逗号分隔的训练子集规模（默认 config.py 的 LEARNING_CURVE['sizes']）")
    parser.add_argument("--targets", default=None,
                        help="all | key | 24 | 逗号分隔的目标名（默认四项关键指标）")
    parser.add_argument("--fit-mode", dest="fit_mode", default=None,
                        help="覆盖 fit_mode：fit_preprocessors（默认，与步骤1一致）/ fit_with_cache（快，数值略有差异）")
    parser.add_argument("--panel-label", dest="panel_label", default=None,
                        help="左上角面板字母（如 a）；默认不加，适合单图直接使用/投稿单面板")
    known, _ = parser.parse_known_args()
    return known


def _resolve_sizes(args, cfg) -> list[int]:
    """训练子集规模：命令行 > config.py > 默认序列。"""
    if args.sizes:
        sizes = [int(s) for s in parse_list(args.sizes)]
    else:
        sizes = list(cfg.learning_curve_sizes)
    if len(set(sizes)) != len(sizes):
        raise ValueError("训练子集规模不能重复。")
    return sizes


def _resolve_panel_label(args) -> str:
    """面板字母：命令行给了就用，否则默认空（不画字母）。"""
    raw = getattr(args, "panel_label", None)
    if raw is None:
        return DEFAULT_PANEL_LABEL
    return str(raw).strip()


def publish(cfg, table: ResumeTable, sizes: list[int], targets: list[str],
            skipped: set[tuple[int, str]], panel_label: str = "") -> None:
    """汇总各规模的 mean / std，并在结果齐全时绘制学习曲线。"""
    out_dir = cfg.learning_curve_dir
    detail = table.frame.copy()
    expected = {(s, t) for s in sizes for t in targets} - skipped
    missing = expected - set(zip(detail["size"], detail["target"]))

    summary = (table.frame.groupby("size")["R2"].agg(["mean", "std"]).reset_index())
    summary["n_targets"] = table.frame.groupby("size")["R2"].size().to_numpy()
    summary.to_csv(out_dir / "learning_curve_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print("学习曲线汇总（四项关键指标平均 R²）")
    print("=" * 74)
    for _, row in summary.iterrows():
        sd = f"{row['std']:.4f}" if row["std"] == row["std"] else "--"
        print(f"  训练规模 {int(row['size']):<6d}  mean R² = {row['mean']:.4f} ± {sd}"
              f"  （{int(row['n_targets'])} 个目标）")
    print("-" * 74)

    if missing:
        print(f"注意: 还有 {len(missing)} 条 (size, target) 结果未完成，本次不绘制学习曲线；"
              f"用同一命令续跑即可补全。", flush=True)
        return
    plot_curve(cfg, summary, targets, out_dir / "fig_learning_curve.png", panel_label)


def plot_curve(cfg, summary: pd.DataFrame, targets: list[str], out_path: Path,
               panel_label: str = "") -> None:
    """绘制 Mean R² - 训练集规模曲线（Arial、#484878 折线、最大规模处虚线参考）。

    panel_label 为空时不画左上角子图编号；拼版成 a/b/c 时传对应字母即可。
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", *cfg.fonts, "DejaVu Sans"]
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.0

    sizes = summary["size"].to_numpy(dtype=float)
    means = summary["mean"].to_numpy(dtype=float)
    n_targets = int(summary["n_targets"].iloc[0])
    is_key = set(targets) <= set(KEY_TARGETS)
    label = f"{n_targets}-Key Targets Mean R²" if is_key else f"{n_targets}-Target Mean R²"
    title = ("TabPFN Performance on 4 Key Targets" if is_key
             else f"TabPFN Performance on {n_targets} Targets")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sizes, means, "o-", color=LINE_COLOR, markersize=6, linewidth=1.5, label=label)

    # 参考线：最大训练规模处的平均 R²
    r2_max = float(means[-1])
    ax.axhline(r2_max, color=REF_COLOR, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(sizes[-1] + 0.02 * sizes[-1], r2_max - 0.002,
            f"Mean R²={r2_max:.4f} (n={int(sizes[-1])})", fontsize=8, color=REF_COLOR, va="top")
    ax.axvline(sizes[-1], color=REF_COLOR, linestyle=":", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Training Set Size (n)", fontsize=10)
    ax.set_ylabel("Mean R²", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(sizes)
    ax.set_xlim(0, sizes[-1] * 1.08)
    lo = min(0.75, float(np.floor((means.min() - 0.02) * 20) / 20))
    hi = max(1.01, float(np.ceil((means.max() + 0.02) * 20) / 20))
    ax.set_ylim(lo, hi)
    if (lo, hi) != (0.75, 1.01):
        print(f"注意: 数值超出默认纵轴 0.75–1.01，已调整为 {lo:.2f}–{hi:.2f}。")
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.15)
    if panel_label:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color="#272727", ha="left", va="bottom")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图表: {out_path}")


if __name__ == "__main__":
    main()
