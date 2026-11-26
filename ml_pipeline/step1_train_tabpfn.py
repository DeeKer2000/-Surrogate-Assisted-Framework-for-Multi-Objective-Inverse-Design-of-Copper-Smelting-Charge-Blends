"""步骤1：TabPFN-3 逐目标训练、预测与评估（29 个输出目标，单次划分）。

用法：
    python ml_pipeline/step1_train_tabpfn.py                    # 用 config.py 的设置
    python ml_pipeline/step1_train_tabpfn.py --dataset 3000_fixed_temp
    python ml_pipeline/step1_train_tabpfn.py --targets key      # 只跑四项关键指标（试跑）

特性：逐目标落盘、断点续跑、显存不足自动折半重试、打印耗时与预计剩余时间。

产出（<结果根目录>/tabpfn/<run>/）：
    evaluation_results.csv     29 个目标的 R2 / MAE / RMSE / MAE%
    predictions.csv            测试集真实值与预测值宽表
    split_indices.npz          划分行号（可复现）
    predictions/*.npz          逐目标的测试集索引、真实值、预测值
    saved_models/*.tabpfn_fit  逐目标已拟合模型（供后续复用）
    figures/                   各目标 R² 条形图、真实值-预测值散点矩阵
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import (
    OUTPUT_COLS, KEY_TARGETS, ResumeTable, apply_cjk_fonts, banner, human,
    load_config, load_dataset, predict_batched, setup_console, setup_env, split_data, timer,
)

warnings.filterwarnings("ignore")
setup_console()

FIELDS = ["target", "n_train", "n_test", "R2", "MAE", "RMSE", "mean_val", "MAE_pct", "elapsed_seconds"]


def main() -> None:
    cfg = load_config()
    setup_env(cfg)
    cfg.ensure_dirs()
    apply_cjk_fonts(cfg)

    import torch
    from tabpfn import TabPFNRegressor, save_fitted_tabpfn_model

    targets = cfg.resolve_targets(_cli_targets())
    started = timer()

    banner(f"步骤1  TabPFN-3 训练与预测（{len(targets)}/{len(OUTPUT_COLS)} 个目标）", cfg)
    print(f"划分      : 测试比例 {cfg.test_size}，随机种子 {cfg.random_state}")
    print(f"预测批量  : {cfg.predict_batch_size}（显存不足自动折半至 {cfg.predict_batch_floor}）")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("警告: 未检测到 CUDA，回退 CPU 会非常慢。", flush=True)

    df, x, y = load_dataset(cfg)
    x_tr, x_te, y_tr, y_te, train_idx, test_idx = split_data(cfg, x, y)
    np.savez_compressed(cfg.tabpfn_dir / "split_indices.npz",
                        train_indices=train_idx, test_indices=test_idx)
    print(f"训练集 {len(x_tr)}，测试集 {len(x_te)}", flush=True)

    # TabPFN 不做梯度下降，fit 就是一次性把训练集编码进上下文，没有 epoch 可报；
    # 真正耗时的是分批预测。这里把节奏说清楚，免得看起来像卡死。
    n_batches = max(1, -(-len(x_te) // max(cfg.predict_batch_size, 1)))
    print(f"节奏      : 每个目标 = 拟合 1 次（in-context 编码，无迭代过程）+ 预测 {n_batches} 批")
    print(f"            每批结束打印一行进度；跑完第 1 个目标后按实测速度给出剩余时间\n", flush=True)

    table = ResumeTable(cfg.tabpfn_dir / "evaluation_results.csv", FIELDS, keys=["target"])
    durations = [float(v) for v in table.frame.get("elapsed_seconds", pd.Series(dtype=float))]

    for order, target in enumerate(targets, start=1):
        tag = f"[{order:2d}/{len(targets)}] {target:<24s}"
        if table.has(target=target):
            print(f"{tag} 已完成，跳过", flush=True)
            continue

        col = OUTPUT_COLS.index(target)
        y_train, y_test = y_tr[:, col], y_te[:, col]
        print(f"{tag} 开始 ...", flush=True)
        t0 = timer()
        reg = None

        if np.std(y_train) < 1e-10:
            y_pred = y_test.copy()
            print("    常数列：预测值取真实值", flush=True)
        else:
            reg = TabPFNRegressor(
                device=cfg.device, model_path=str(cfg.model_path),
                n_estimators=cfg.n_estimators, random_state=cfg.model_seed,
                ignore_pretraining_limits=True,
            )
            n_batches = max(1, -(-len(x_te) // max(cfg.predict_batch_size, 1)))
            print(f"    [1/2] 拟合中（in-context 编码，没有迭代过程；大样本约需 30~60 秒）...", flush=True)
            reg.fit(x_tr, y_train)
            t_fit = timer()
            print(f"    [1/2] 拟合完成 {human(t_fit - t0)}"
                  f"，开始预测 {len(x_te)} 行（分 {n_batches} 批）...", flush=True)
            y_pred = predict_batched(
                reg, x_te, cfg.predict_batch_size, cfg.predict_batch_floor,
                progress=lambda i, n, rows, total, secs: print(
                    f"    [2/2] 预测批次 {i}/{n}  累计 {rows}/{total} 行  "
                    f"本批 {secs:.0f}s  已用 {human(timer() - t0)}", flush=True),
            )
            print(f"    [2/2] 预测完成 {human(timer() - t_fit)}"
                  f"（拟合 {human(t_fit - t0)} + 预测 {human(timer() - t_fit)}）", flush=True)

        # 先落盘预测，再保存模型：模型保存失败也不影响本次结果
        np.savez_compressed(cfg.predictions_dir / f"{target}.npz",
                            test_indices=test_idx, y_true=y_test,
                            y_pred=np.asarray(y_pred, dtype=float))

        save_error = None
        if reg is not None:
            try:
                save_fitted_tabpfn_model(reg, cfg.models_dir / f"{target}.tabpfn_fit")
            except Exception as exc:
                save_error = f"{type(exc).__name__}: {exc}"
            del reg
            torch.cuda.empty_cache()

        elapsed = timer() - t0
        durations.append(elapsed)
        mean_val = float(np.mean(y_test))
        mae = float(mean_absolute_error(y_test, y_pred))
        table.add(
            target=target, n_train=len(x_tr), n_test=len(x_te),
            R2=float(r2_score(y_test, y_pred)), MAE=mae,
            RMSE=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            mean_val=mean_val, MAE_pct=float(mae / abs(mean_val) * 100) if mean_val else np.nan,
            elapsed_seconds=elapsed,
        )

        done = len([t for t in targets if table.has(target=t)])
        avg = sum(durations) / len(durations)
        print(f"    R2={r2_score(y_test, y_pred):.4f}  MAE={mae:.4f}  用时 {human(elapsed)}  "
              f"已用 {human(timer() - started)}  预计剩余 {human(avg * (len(targets) - done))}", flush=True)
        if save_error:
            print(f"    [警告] 模型文件保存失败（指标与预测已保存，不影响结果）：{save_error}", flush=True)

    present = table.reindex("target", OUTPUT_COLS)["target"].tolist()
    write_summary(cfg, table.frame, present, test_idx, len(x_te))
    print(f"\n步骤1 完成（{len(present)}/{len(OUTPUT_COLS)} 个目标），"
          f"用时 {human(timer() - started)}。\n结果目录: {cfg.tabpfn_dir}")


def _cli_targets() -> str | None:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", default=None)
    known, _ = parser.parse_known_args()
    return known.targets


def write_summary(cfg, metrics: pd.DataFrame, present: list[str], test_idx, n_test: int) -> None:
    """打印汇总、导出预测宽表、绘制条形图与散点矩阵。"""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 74)
    print("评估结果汇总")
    print("=" * 74)
    header = f"{'目标变量':<26s}{'R2':>9s}{'MAE':>11s}{'RMSE':>11s}{'MAE%':>9s}"
    print(header)
    print("-" * 74)
    for _, r in metrics.iterrows():
        print(f"{r['target']:<26s}{r['R2']:9.4f}{r['MAE']:11.4f}{r['RMSE']:11.4f}{r['MAE_pct']:8.1f}%")
    print("-" * 74)
    print(f"{'平均':<26s}{metrics['R2'].mean():9.4f}")
    r2_map = metrics.set_index("target")["R2"]
    for key in KEY_TARGETS:
        if key in r2_map.index:
            r = metrics.set_index("target").loc[key]
            print(f"  [关键指标] {key:<20s} R2={r['R2']:.5f}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}")

    wide = pd.DataFrame({"test_index": test_idx})
    for target in present:
        archive = np.load(cfg.predictions_dir / f"{target}.npz")
        wide[f"{target}__true"] = archive["y_true"]
        wide[f"{target}__pred"] = archive["y_pred"]
    wide.to_csv(cfg.tabpfn_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n预测明细: {cfg.tabpfn_dir / 'predictions.csv'}")

    # 各目标 R² 条形图
    order = metrics.sort_values("R2", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.30 * len(order) + 1.2)))
    ax.barh(order["target"], order["R2"],
            color=["#C44E52" if v < 0.9 else "#4C72B0" for v in order["R2"]],
            edgecolor="#2b2b2b", linewidth=0.4)
    ax.axvline(0.9, color="#888888", linestyle=":", linewidth=0.9)
    ax.set_xlim(min(0, order["R2"].min() * 1.05), 1.01)
    ax.set_xlabel("R$^2$")
    # 标题里的上标用 mathtext（R$^2$），不要用字面量 R²：
    # SimHei 缺 U+00B2，而 matplotlib 不会逐字符回退，字面量会画成豆腐块。
    ax.set_title(f"TabPFN-3 各目标 R$^2$（{cfg.dataset}，单次 {int((1 - cfg.test_size) * 100)}:"
                 f"{int(cfg.test_size * 100)} 划分，seed={cfg.random_state}）")
    ax.grid(axis="x", alpha=0.3)
    for ypos, value in enumerate(order["R2"]):
        ax.text(value + 0.005 if value >= 0 else value - 0.005, ypos, f"{value:.4f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(cfg.tabpfn_figures_dir / "metrics_r2_all_targets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 各目标真实值-预测值散点矩阵
    n_cols = 5
    n_rows = int(np.ceil(len(present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, target in zip(axes, present):
        archive = np.load(cfg.predictions_dir / f"{target}.npz")
        yt, yp = archive["y_true"], archive["y_pred"]
        ax.scatter(yt, yp, alpha=0.35, s=8, color="steelblue", edgecolors="none")
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, linewidth=0.9)
        ax.set_title(f"{target}  R$^2$={r2_map[target]:.4f}", fontsize=8.5)
        ax.set_xlabel("FactSage 真实值", fontsize=8)
        ax.set_ylabel("TabPFN 预测值", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axes[len(present):]:
        ax.set_visible(False)
    fig.suptitle(f"TabPFN-3 真实值 vs 预测值（{cfg.dataset}，测试集 {n_test} 组）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(cfg.tabpfn_figures_dir / "scatter_true_vs_pred.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"图表: {cfg.tabpfn_figures_dir}")


if __name__ == "__main__":
    main()
