"""
训练数据量影响实验：固定测试集下不同嵌套训练子集规模的 TabPFN 预测性能
输出：学习曲线图（R² vs 训练集大小）

用法：
    python scripts/analysis/plot_learning_curve.py                 # 完整重跑（训练 + 出图）
    python scripts/analysis/plot_learning_curve.py --replot-only    # 只读已有 summary CSV 重画图，不重训
"""
import argparse
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', buffering=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── 路径 ──
DATA_PATH = r'data\3000_fixed_temp\ml_dataset.csv'
OUTPUT_DIR = r'experiments\3000_fixed_temp\learning_curve'
FIGNAME = 'fig_learning_curve.png'
# 左上角面板字母（子图编号）：仅拼版成 a/b/c 时使用，单图留空即可
PANEL_LABEL = ''

import os
TABPFN_MODEL_PATH = os.environ.get('TABPFN_MODEL_PATH', 'auto')   # 'auto' = 自动下载

INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'
]
OUTPUT_COLS = [
    'matte_Cu_pct', 'slag_Cu_pct', 'slag_FeO_pct', 'slag_SiO2_pct',
]

# 嵌套训练子集规模
TRAIN_SIZES = [100, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]


def run_experiment() -> pd.DataFrame:
    """逐训练量实验，返回各规模的 mean/std 汇总。"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from tabpfn import TabPFNRegressor

    # ── 加载数据 ──
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    X = df[INPUT_COLS].values
    Y = df[OUTPUT_COLS].values

    # 固定的 80/20 划分：2400 组训练样本，600 组测试样本
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    N_train = len(X_train)
    print(f"总训练集: {N_train}, 测试集: {len(X_test)}")

    # ── 逐训练量实验 ──
    results = []  # [{size, target, R2}]

    for size in TRAIN_SIZES:
        print(f"\n{'='*50}")
        print(f"训练集大小: {size}")
        print(f"{'='*50}")

        # train_test_split 已随机打乱训练集；按同一顺序取前 size 条，
        # 使各规模训练子集相互嵌套，而测试集始终保持不变。
        X_sub = X_train[:size]
        Y_sub = Y_train[:size]

        for i, target in enumerate(OUTPUT_COLS):
            y_train = Y_sub[:, i]
            y_test = Y_test[:, i]
            if np.std(y_train) < 1e-10:
                continue

            reg = TabPFNRegressor(
                device='cuda',
                model_path=TABPFN_MODEL_PATH,
                ignore_pretraining_limits=True
            )
            reg.fit(X_sub, y_train)
            y_pred = reg.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            results.append({'size': size, 'target': target, 'R2': r2})

            print(f"  [{i+1}/{len(OUTPUT_COLS)}] {target}: R²={r2:.4f}")

    # ── 汇总（4个关键冶金目标） ──
    df_res = pd.DataFrame(results)
    df_avg = df_res.groupby('size')['R2'].agg(['mean', 'std']).reset_index()

    # 保存结果
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / 'learning_curve_detail.csv', index=False, encoding='utf-8-sig')
    df_avg.to_csv(out_dir / 'learning_curve_summary.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("结果汇总")
    print("="*50)
    for _, row in df_avg.iterrows():
        print(f"  size={int(row['size']):4d}:  mean R²={row['mean']:.4f} ±{row['std']:.4f}")

    return df_avg


def plot_from_summary(df_avg: pd.DataFrame, out_path: Path) -> None:
    """按汇总表绘图（Mean R² vs 训练集大小）。PANEL_LABEL 为空时不画子图编号。"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4))

    sizes = df_avg['size'].values
    means = df_avg['mean'].values
    stds = df_avg['std'].values

    # 主曲线（4 个关键指标平均 R²）
    ax.plot(sizes, means, 'o-', color='#484878',
            markersize=6, linewidth=1.5, label='4-Key Targets Mean R²')

    # 参考线：最大训练规模处的平均 R²
    r2_max = means[-1]
    ax.axhline(r2_max, color='#A0A0A0', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(sizes[-1] + 50, r2_max - 0.002, f'Mean R²={r2_max:.4f} (n={int(sizes[-1])})',
            fontsize=8, color='#A0A0A0', va='top')

    # 标注最大规模处
    ax.axvline(sizes[-1], color='#A0A0A0', linestyle=':', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Training Set Size (n)', fontsize=10)
    ax.set_ylabel('Mean R²', fontsize=10)
    ax.set_title('TabPFN Performance on 4 Key Targets', fontsize=11, fontweight='bold')
    _ = ax.set_xticks(sizes)
    ax.set_ylim(0.75, 1.01)
    ax.set_xlim(0, sizes[-1] * 1.08)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    ax.grid(alpha=0.15)

    # 标签（PANEL_LABEL 为空则不画）
    if PANEL_LABEL:
        ax.text(-0.12, 1.05, PANEL_LABEL, transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='#272727', ha='left', va='bottom')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='学习曲线：训练 + 绘图')
    parser.add_argument('--replot-only', action='store_true',
                        help='不重训模型，直接用已有的 learning_curve_summary.csv 重画图')
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        summary_path = out_dir / 'learning_curve_summary.csv'
        if not summary_path.exists():
            raise SystemExit(f"找不到 {summary_path}，请先完整跑一次（不加 --replot-only）。")
        df_avg = pd.read_csv(summary_path, encoding='utf-8-sig')
        print(f"读取汇总: {summary_path}（{len(df_avg)} 个训练规模，未重训模型）")
    else:
        df_avg = run_experiment()

    plot_from_summary(df_avg, out_dir / FIGNAME)


if __name__ == '__main__':
    main()
