"""
读取人工配矿数据，用 TabPFN 预测年度平均配料的冶金结果（slag_Cu, matte_Cu, FeO/SiO2）。
用于替换论文中的人工经验估算值。
用法: conda run -n tabpfn python scripts/analysis/predict_manual_blend.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ---- 配置 ----
EXCEL_PATH = PROJECT_ROOT / 'data' / '人工配矿.xlsx'
MODELS_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'tabpfn' / '20260622_2247' / 'saved_models'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'optimization'

# 固定工艺参数（与 nsga2_optimization.py 一致）
T_DEFAULT = 1200.0
P_DEFAULT = 1.0
O2_DEFAULT = 21.0
TOTAL_CHARGE = 100.0

# 目标变量
TARGETS = ['slag_Cu_pct', 'matte_Cu_pct', 'slag_FeO_SiO2_ratio']


def load_and_average_blend(path):
    """读取人工配矿 Excel，返回年度平均配料成分（wt%）。"""
    df = pd.read_excel(path)
    # 过滤出有效数据行（有 Cu 值且 Cu 为数值的行）
    df = df[pd.to_numeric(df['Cu (%)'], errors='coerce').notna()].copy()
    # 选取需要的列并重命名
    col_map = {
        'Cu (%)': 'Cu', 'S (%)': 'S', 'Fe (%)': 'Fe', 'SiO2 (%)': 'SiO2',
        'CaO (%)': 'CaO', 'MgO (%)': 'MgO', 'As (%)': 'As', 'Sb (%)': 'Sb',
        'Bi (%)': 'Bi', 'Al2O3 (%)': 'Al2O3', 'Ni (%)': 'Ni', 'Pb (%)': 'Pb',
        'Zn (%)': 'Zn',
    }
    df = df[list(col_map.keys())].rename(columns=col_map)
    # 确保所有列为数值类型
    df = df.apply(pd.to_numeric, errors='coerce')
    # 计算年度平均
    avg = df.mean()
    print(f"有效数据行数: {len(df)}")
    return avg


def blend_to_features(blend_series):
    """将配料成分（wt% series）转换为模型输入特征向量（16维）。"""
    features = np.array([
        T_DEFAULT, P_DEFAULT,
        blend_series['Cu'],      # Cu(g)
        blend_series['S'],       # S(g)
        blend_series['Fe'],      # Fe(g)
        blend_series['SiO2'],    # SiO2(g)
        blend_series['Pb'],      # Pb(g)
        blend_series['Zn'],      # Zn(g)
        blend_series['MgO'],     # MgO(g)
        blend_series['Al2O3'],   # Al2O3(g)
        blend_series['Ni'],      # Ni(g)
        blend_series['Sb'],      # Sb(g)
        blend_series['Bi'],      # Bi(g)
        blend_series['CaO'],     # CaO(g)
        blend_series['As'],      # As(g)
        O2_DEFAULT,              # O2(g)
    ], dtype=np.float32)
    return features


def load_model(name):
    """加载单个 TabPFN 模型。"""
    import tempfile, zipfile, json, joblib
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import _extract_archive, InferenceEngine

    # 预训练权重：默认 'auto'（由 tabpfn 自动下载/走缓存），
    # 也可用环境变量 TABPFN_MODEL_PATH 指向本地 ckpt
    MODEL_PATH_ABS = os.environ.get('TABPFN_MODEL_PATH', 'auto')

    path = MODELS_DIR / f'{name}.tabpfn_fit'
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _extract_archive(path, tmp)
        with (tmp / "init_params.json").open() as f:
            params = json.load(f)
        params.pop("__class_name__")
        params["model_path"] = MODEL_PATH_ABS
        params["device"] = 'cuda'
        est = TabPFNRegressor(**params)
        est._initialize_model_variables()
        fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
        for key, value in fitted_attrs.items():
            setattr(est, key, value)
        est.executor_ = InferenceEngine.load_state(
            tmp / "executor_state.joblib", est.models_)
        est.to('cuda')
    return est


def main():
    print("=" * 60)
    print("人工配矿数据预测（年度平均 → 冶金结果）")
    print("=" * 60)

    # 1. 读取年度平均配料成分
    print("\n[1/4] 读取人工配矿数据...")
    avg_blend = load_and_average_blend(EXCEL_PATH)
    print("\n年度平均配料成分 (wt%):")
    for k, v in avg_blend.items():
        print(f"  {k}: {v:.4f}")

    # 2. 转换为模型输入特征
    print("\n[2/4] 转换为模型输入特征...")
    features = blend_to_features(avg_blend)
    print("  特征向量: ")
    for col, val in zip(
        ['T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
         'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
         'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'],
        features
    ):
        print(f"    {col}: {val:.4f}")

    # 3. 加载模型并预测
    print("\n[3/4] 加载 TabPFN 模型并预测...")
    results = {}
    for target in TARGETS:
        print(f"  预测: {target}...")
        model = load_model(target)
        pred = model.predict(features.reshape(1, -1))[0]
        results[target] = float(pred)
        print(f"    = {pred:.4f}")

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("预测结果（年度平均人工配料 → 冶金输出）")
    print("=" * 60)
    print(f"  slag_Cu_pct (渣含铜):       {results['slag_Cu_pct']:.4f} wt%")
    print(f"  matte_Cu_pct (冰铜品位):    {results['matte_Cu_pct']:.2f} wt%")
    print(f"  slag_FeO_SiO2_ratio (Fe/SiO2): {results['slag_FeO_SiO2_ratio']:.3f}")
    print()

    # 与之前估算值的对比
    print("与之前估算值的对比:")
    print(f"  slag_Cu:   估算 0.6500 → 实际预测 {results['slag_Cu_pct']:.4f}")
    print(f"  matte_Cu:  估算 58.00 → 实际预测 {results['matte_Cu_pct']:.2f}")
    print(f"  FeO/SiO2:  估算 1.350 → 实际预测 {results['slag_FeO_SiO2_ratio']:.3f}")

    # 保存结果
    output_path = OUTPUT_DIR / 'manual_blend_prediction.csv'
    pd.DataFrame([results]).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()