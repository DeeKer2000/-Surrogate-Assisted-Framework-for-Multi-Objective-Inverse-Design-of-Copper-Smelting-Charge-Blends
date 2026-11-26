"""步骤3：TabPFN-3 多随机划分重复评价（24 项基础组分目标，论文表3 / 表S1）。

用法：
    python ml_pipeline/step3_multiseed_evaluation.py                       # 种子 42-46 × 24 个目标
    python ml_pipeline/step3_multiseed_evaluation.py --dataset 3000_fixed_temp
    python ml_pipeline/step3_multiseed_evaluation.py --targets key --seeds 42,43
    python ml_pipeline/step3_multiseed_evaluation.py --max-pairs 4         # 只新跑 4 对再退出
    python ml_pipeline/step3_multiseed_evaluation.py --targets all         # 换成全部 29 个目标

特性：按 (seed, target) 断点续跑，每完成一对立即落盘预测与指标；
      汇总表发布前校验种子×目标组合完整、指标有限、无重复行。

产出（<结果根目录>/multiseed/<run>/）：
    per_seed_metrics.csv                 每个 (seed, target) 的 R2 / RMSE / MAE
    seed_<seed>/split_indices.npz        该随机种子的训练/测试行号
    seed_<seed>/<target>_predictions.npz 测试集行号、真实值、预测值
    target_mean_sd.csv                   逐目标跨种子的均值 / 标准差 / 极值
    table_s1_24_targets_mean_sd.csv      表S1：24 项目标的 mean ± sd
    table3_key_targets_mean_sd.csv       表3：四项关键指标的 mean ± sd
    table3_key_targets_formatted.csv     表3："mean ± sd" 文本格式
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import (
    KEY_TARGETS, OUTPUT_COLS, ResumeTable, banner, human, load_config,
    load_dataset, normalize_fit_mode, parse_list, predict_batched, setup_console,
    setup_env, split_data, timer,
)

warnings.filterwarnings("ignore")
setup_console()

FIELDS = ["seed", "target", "n_train", "n_test", "R2", "RMSE", "MAE", "elapsed_seconds"]
TABLE_COLUMNS = [
    "target", "n_seeds", "R2_mean", "R2_sd",
    "RMSE_mean", "RMSE_sd", "MAE_mean", "MAE_sd",
]


def main() -> None:
    args = _cli_args()
    cfg = load_config()
    setup_env(cfg)
    cfg.ensure_dirs()

    import torch
    from tabpfn import TabPFNRegressor

    targets = cfg.resolve_targets(args.targets or "24")
    if len(set(targets)) != len(targets):
        raise ValueError("目标变量不能重复。")
    seeds = [int(s) for s in parse_list(args.seeds)] if args.seeds else list(cfg.seeds)
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("至少要两个互不相同的划分随机种子。")

    # fit_mode：默认与步骤1/2 一致（关闭 KV 缓存），保证表格数值口径统一。
    # 加 --fit-mode cache 可加速，但与步骤1 的预测不是数值等价的，见 config.py 第 5 节。
    fit_mode = normalize_fit_mode(args.fit_mode, cfg.fit_mode_for("multiseed"))
    if fit_mode == "fit_with_cache":
        print("提示: 已启用 KV 缓存加速，结果与步骤1（默认路径）在 1e-4~1e-6 量级上略有差异。",
              flush=True)

    total = len(seeds) * len(targets)
    started = timer()

    banner(f"步骤3  多随机划分重复评价（{len(seeds)} 个种子 × {len(targets)} 个目标 = {total} 对）", cfg)
    print(f"划分随机种子: {seeds}")
    print(f"模型        : n_estimators={cfg.n_estimators}，model_seed={cfg.model_seed}，"
          f"fit_mode={fit_mode}")
    print(f"预测批量    : {cfg.predict_batch_size}（显存不足自动折半至 {cfg.predict_batch_floor}）")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("警告: 未检测到 CUDA，回退 CPU 会非常慢。", flush=True)

    _, x, y = load_dataset(cfg)
    print(f"目标        : {', '.join(targets)}\n", flush=True)

    table = ResumeTable(cfg.multiseed_dir / "per_seed_metrics.csv", FIELDS, keys=["seed", "target"])
    durations = [float(v) for v in table.frame.get("elapsed_seconds", pd.Series(dtype=float)) if v == v]
    done = len([1 for s in seeds for t in targets if table.has(seed=s, target=t)])
    if done:
        print(f"断点续跑: 已有 {done}/{total} 对结果，跳过重复计算。\n", flush=True)

    new_pairs = 0
    for seed in seeds:
        # 划分行号先落盘，便于复核与复现
        x_tr, x_te, y_tr, y_te, train_idx, test_idx = split_data(cfg, x, y, seed=seed)
        seed_dir = cfg.multiseed_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        indices_file = seed_dir / "split_indices.npz"
        if indices_file.exists():
            with np.load(indices_file) as archive:
                np.testing.assert_array_equal(archive["train_indices"], train_idx)
                np.testing.assert_array_equal(archive["test_indices"], test_idx)
        else:
            np.savez_compressed(indices_file, train_indices=train_idx, test_indices=test_idx)

        for target in targets:
            col = OUTPUT_COLS.index(target)
            if table.has(seed=seed, target=target):
                print(f"[跳过] seed={seed} {target:<22s} 已完成", flush=True)
                continue

            order = done + 1
            print(f"[{order:3d}/{total}] seed={seed} {target:<22s} 开始 ...", flush=True)
            t0 = timer()
            y_train, y_test = y_tr[:, col], y_te[:, col]

            reg = TabPFNRegressor(
                device=cfg.device, model_path=str(cfg.model_path),
                n_estimators=cfg.n_estimators, random_state=cfg.model_seed,
                ignore_pretraining_limits=True, fit_mode=fit_mode,
            )
            reg.fit(x_tr, y_train)
            t_fit = timer()
            y_pred = np.asarray(predict_batched(
                reg, x_te, cfg.predict_batch_size, cfg.predict_batch_floor), dtype=float)
            print(f"           拟合 {human(t_fit - t0)} / 预测 {human(timer() - t_fit)}", flush=True)

            if not np.isfinite(y_pred).all():
                raise ValueError(f"预测值非有限: seed={seed}, target={target}")

            np.savez_compressed(seed_dir / f"{target}_predictions.npz",
                                test_indices=test_idx, y_true=y_test, y_pred=y_pred)

            elapsed = timer() - t0
            durations.append(elapsed)
            r2 = float(r2_score(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            table.add(seed=seed, target=target, n_train=len(x_tr), n_test=len(x_te),
                      R2=r2, RMSE=rmse, MAE=mae, elapsed_seconds=elapsed)

            del reg
            gc.collect()
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            done += 1
            new_pairs += 1
            avg = sum(durations) / len(durations)
            print(f"[{done:3d}/{total}] seed={seed} {target:<22s} R2={r2:.6f}  RMSE={rmse:.6g}  "
                  f"MAE={mae:.6g}  用时 {human(elapsed)}  已用 {human(timer() - started)}  "
                  f"预计剩余 {human(avg * (total - done))}", flush=True)

            if args.max_pairs and new_pairs >= args.max_pairs and done < total:
                print(f"\n试跑结束（本轮新增 {new_pairs} 对，累计 {done}/{total}）。"
                      f"再次运行同一命令即可续跑。\n结果目录: {cfg.multiseed_dir}", flush=True)
                return

    summarize(cfg, seeds, targets)
    print(f"\n步骤3 完成（{total} 对），用时 {human(timer() - started)}。\n结果目录: {cfg.multiseed_dir}")


def _cli_args() -> argparse.Namespace:
    """解析本脚本专有参数；--config/--dataset/--run/--root 由 common.load_config 处理。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", default=None,
                        help="all | key | 24 | 逗号分隔的目标名（默认 24 项基础组分）")
    parser.add_argument("--seeds", default=None, help="逗号分隔的划分随机种子（默认 config.py 的 SPLIT['seeds']）")
    parser.add_argument("--max-pairs", type=int, default=None, help="本轮最多新跑多少对 (seed, target)")
    parser.add_argument("--fit-mode", dest="fit_mode", default=None,
                        help="覆盖 fit_mode：fit_preprocessors（默认，与步骤1一致）/ fit_with_cache（快，数值略有差异）")
    known, _ = parser.parse_known_args()
    return known


def summarize(cfg, seeds: list[int], targets: list[str]) -> None:
    """完整性校验通过后发布逐目标汇总表与论文表3 / 表S1。"""
    out_dir = cfg.multiseed_dir
    detail = pd.read_csv(out_dir / "per_seed_metrics.csv")

    if detail.duplicated(["seed", "target"]).any():
        raise ValueError("per_seed_metrics.csv 存在重复的 (seed, target) 行，不发布汇总表。")
    expected = {(s, t) for s in seeds for t in targets}
    actual = set(zip(detail["seed"], detail["target"]))
    if actual != expected or len(detail) != len(expected):
        raise ValueError(
            f"种子×目标组合不完整（应有 {len(expected)} 对，实际 {len(actual)} 对），"
            f"不发布汇总表；请用同一命令续跑补全。")
    if not np.isfinite(detail[["R2", "RMSE", "MAE"]].to_numpy()).all():
        raise ValueError("指标中存在非有限值，不发布汇总表。")

    rows = []
    for target in targets:
        group = detail.loc[detail["target"] == target]
        row = {"target": target, "n_seeds": len(group)}
        for metric in ("R2", "RMSE", "MAE"):
            row.update({
                f"{metric}_mean": float(group[metric].mean()),
                f"{metric}_sd": float(group[metric].std(ddof=1)),
                f"{metric}_min": float(group[metric].min()),
                f"{metric}_max": float(group[metric].max()),
            })
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "target_mean_sd.csv", index=False, encoding="utf-8-sig")

    # 表S1：全部目标；表3：四项关键指标（若本次未跑该目标则跳过）
    summary[TABLE_COLUMNS].to_csv(
        out_dir / "table_s1_24_targets_mean_sd.csv", index=False, encoding="utf-8-sig")
    present_keys = [t for t in KEY_TARGETS if t in set(summary["target"])]
    if present_keys:
        key_summary = summary.set_index("target").loc[present_keys].reset_index()
        key_summary[TABLE_COLUMNS].to_csv(
            out_dir / "table3_key_targets_mean_sd.csv", index=False, encoding="utf-8-sig")
        formatted = key_summary[["target"]].copy()
        for metric in ("R2", "RMSE", "MAE"):
            decimals = 5 if metric == "R2" else 4
            formatted[metric] = [
                f"{row[f'{metric}_mean']:.{decimals}f} ± {row[f'{metric}_sd']:.{decimals}f}"
                for _, row in key_summary.iterrows()
            ]
        formatted.to_csv(
            out_dir / "table3_key_targets_formatted.csv", index=False, encoding="utf-8-sig")
    else:
        print("注意: 本次未评价任何关键指标目标，跳过表3。", flush=True)

    print("\n" + "=" * 74)
    print("多划分汇总（跨种子 mean ± sd）")
    print("=" * 74)
    print(f"{'目标变量':<24s}{'R2_mean':>10s}{'R2_sd':>10s}{'RMSE_mean':>12s}{'MAE_mean':>12s}")
    print("-" * 74)
    for _, r in summary.iterrows():
        print(f"{r['target']:<24s}{r['R2_mean']:10.5f}{r['R2_sd']:10.5f}"
              f"{r['RMSE_mean']:12.4g}{r['MAE_mean']:12.4g}")
    print("-" * 74)
    per_seed = detail.groupby("seed")["R2"].mean()
    print(f"种子平均 R²: 均值 {per_seed.mean():.5f}  标准差 {per_seed.std(ddof=1):.5f}"
          f"  （{len(per_seed)} 个种子）")
    print(f"\n表格: {out_dir / 'table_s1_24_targets_mean_sd.csv'}")


if __name__ == "__main__":
    main()
